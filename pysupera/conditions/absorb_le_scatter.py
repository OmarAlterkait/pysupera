from collections import defaultdict
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np

from .base import PartitionConditionBase
from ..diagnostics import MergeOutcome
from ..utils import SemanticType

if TYPE_CHECKING:
    from ..partitioner import ParticlePartitioner
    from ..data import Particle


class AbsorbLEScatter(PartitionConditionBase):
    """
    Absorb each ``kLEScatter`` particle into at most one touching non-``kLEScatter`` neighbour.

    This condition targets low-energy scatter products
    (``sem_type == SemanticType.kLEScatter``) that are spatially adjacent to
    any other semantic class.  Each ``kLEScatter`` particle is allowed to
    merge with at most one partner to avoid over-clustering.

    Run :class:`CombineLEScatters` **before** this condition so that
    transitively connected LEScatter chains are consolidated into one particle
    prior to absorption.

    Pipeline
    --------
    1. ``get_candidates`` builds all pairs between ``SemanticType.kLEScatter``
       particles and particles with a different semantic type –
       O(n_kLEScatter × n_other).
    2. The base partitioner tests proximity on those candidates.
    3. ``post_filter`` enforces the one-merge-per-kLEScatter constraint on
       the touching pairs returned by the proximity check.
    """

    @property
    def name(self) -> str:
        return "AbsorbLEScatter"

    @property
    def description(self) -> str:
        return "Absorb kLEScatter particles into touching non-kLEScatter neighbours"

    def get_candidates(self, partitioner: 'ParticlePartitioner') -> List[Tuple[int, int]]:
        """
        Build all pairs between ``kLEScatter`` particles and non-``kLEScatter`` particles.

        Partitions the full particle list into two sub-lists in O(n), then
        returns their Cartesian product in O(n_kLEScatter × n_other), avoiding
        the full O(n²) enumeration.

        Parameters
        ----------
        partitioner : ParticlePartitioner
            Active partitioner providing ``particles`` and ``diagnostics``.

        Returns
        -------
        list of tuple[int, int]
            Candidate pairs expressed as ``(kLEScatter_id, other_id)`` using
            particle IDs (not positional indices).
        """
        candidates = self._build_candidates(partitioner)
        self._contact = {}      # particle-level path: no contact measured
        partitioner.diagnostics.record_candidates(
            f"{self.name}: Stage 1 Candidate Building", candidates
        )
        return candidates

    def post_filter(self,
                    partitioner: 'ParticlePartitioner',
                    touching_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Allow each ``kLEScatter`` particle to merge with at most one partner.

        Groups *touching_pairs* by their ``kLEScatter`` member and keeps one
        touching non-``kLEScatter`` neighbour for each group: the one holding
        its nearest ancestor, else its nearest relative, else by interaction,
        contact, distance and creation time (see
        :meth:`_select_one_merge_per_sem5`).
        Surplus candidates are discarded; when diagnostics are enabled each
        rejection is recorded with reason ``ALREADY_MERGED_WITH_OTHER``.

        Parameters
        ----------
        partitioner : ParticlePartitioner
            Active partitioner providing ``particle_lookup`` and
            ``diagnostics``.
        touching_pairs : list of tuple[int, int]
            Pairs that passed the proximity check, as produced by
            ``ParticlePartitioner._batch_check_proximity_optimized``.

        Returns
        -------
        list of tuple[int, int]
            At most one pair per ``kLEScatter`` particle.
        """
        return self._select_one_merge_per_sem5(partitioner, touching_pairs)

    def get_rep_candidates(
            self,
            partitioner: 'ParticlePartitioner',
            rep_lookup: Dict[int, 'Particle']) -> List[Tuple[int, int]]:
        """
        Return directed pairs pairing each ``kLEScatter`` representative with
        every non-``kLEScatter`` representative.

        Pair ordering: ``(le_scatter_rep_id, other_rep_id)`` — the
        ``kLEScatter`` partition merges INTO the non-``kLEScatter`` partition,
        whose representative survives.

        Parameters
        ----------
        partitioner : ParticlePartitioner
        rep_lookup : dict[int, Particle]
            Maps original particle ID → current representative Particle.

        Returns
        -------
        list of tuple[int, int]
            ``(le_scatter_rep_id, other_rep_id)``.
        """
        #: (LE rep id, other rep id) -> (contact, closest distance), for
        #: post_filter; empty until a pass finds touching pairs.
        self._contact = {}
        # Deduplicate representatives
        unique_reps = self.unique_reps(rep_lookup)
        le_reps    = [r for r in unique_reps if r.sem_type == SemanticType.kLEScatter]
        other_reps = [r for r in unique_reps if r.sem_type != SemanticType.kLEScatter]

        if not le_reps or not other_reps:
            partitioner.diagnostics.record_candidates(
                f"{self.name}: Rep Candidate Building", []
            )
            return []

        D = partitioner.checker.D

        # Global point-cloud KDTree: build separate LE and other point arrays
        # labelled by rep index, then call query_ball_tree(D) once to find all
        # (LE_point, other_point) pairs within D in O(N log N).
        # LE scatter clouds are always small (few hits/voxels), so the output
        # size is bounded by n_le_pts × local_density near touching boundaries.
        def _stack(reps):
            clouds   = []
            rep_tags = []
            for k, rep in enumerate(reps):
                parts = rep._cloud_parts
                c = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
                if len(c) == 0:
                    continue
                clouds.append(c[:, :3].astype(np.float32))
                rep_tags.append(np.full(len(c), k, dtype=np.int32))
            if not clouds:
                return None, None
            return np.concatenate(rep_tags), np.concatenate(clouds)

        le_tags, le_pts = _stack(le_reps)
        ot_tags, ot_pts = _stack(other_reps)

        if le_tags is None or ot_tags is None:
            partitioner.diagnostics.record_candidates(
                f"{self.name}: Rep Candidate Building", []
            )
            return []

        from scipy.spatial import cKDTree
        # Every (LE point, other point) pair within D, as two index arrays.
        m = cKDTree(le_pts).sparse_distance_matrix(
            cKDTree(ot_pts), D, output_type="ndarray")
        pi = m["i"].astype(np.int64)
        pj = m["j"].astype(np.int64)

        # Map to rep pairs, and keep per pair what the selection may need:
        # how many LE points lie within D of the other rep (its contact),
        # and the closest distance.  The list order decides nothing.
        n_ot = len(other_reps)
        rk = le_tags[pi].astype(np.int64) * n_ot + ot_tags[pj]
        if not len(rk):
            partitioner.diagnostics.record_candidates(
                f"{self.name}: Rep Candidate Building", []
            )
            return []
        order = np.argsort(rk, kind="stable")
        rk, pi, dist = rk[order], pi[order], np.asarray(m["v"])[order]
        start = np.flatnonzero(np.r_[True, rk[1:] != rk[:-1]])
        pairs = rk[start]
        dmin = np.minimum.reduceat(dist, start)
        # an LE point counts once per pair, however many points it is near
        inv = np.repeat(np.arange(len(pairs)), np.diff(np.r_[start, len(rk)]))
        contact = np.bincount(np.unique(inv * len(le_pts) + pi) // len(le_pts),
                              minlength=len(pairs))
        le_ids = np.array([r.id for r in le_reps])[pairs // n_ot]
        ot_ids = np.array([r.id for r in other_reps])[pairs % n_ot]
        candidates = list(zip(le_ids.tolist(), ot_ids.tolist()))
        # only an LE rep with more than one partner is ever ranked
        multi = np.flatnonzero(np.r_[False, le_ids[1:] == le_ids[:-1]]
                               | np.r_[le_ids[:-1] == le_ids[1:], False])
        for k in multi.tolist():
            self._contact[candidates[k]] = (int(contact[k]), float(dmin[k]))
        partitioner.diagnostics.record_candidates(
            f"{self.name}: Rep Candidate Building", candidates
        )
        return candidates

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_candidates(self, partitioner: 'ParticlePartitioner') -> List[Tuple[int, int]]:
        """
        Form the Cartesian product of ``kLEScatter`` and non-``kLEScatter`` sub-populations.

        Makes two O(n) passes over ``partitioner.particles`` to collect the
        two sub-lists, then iterates their product in
        O(n_kLEScatter × n_other).

        Parameters
        ----------
        partitioner : ParticlePartitioner
            Active partitioner providing ``particles``.

        Returns
        -------
        list of tuple[int, int]
            All ``(kLEScatter_particle_id, other_particle_id)`` combinations.
        """
        sem5_particles  = [p for p in partitioner.particles if p.sem_type == SemanticType.kLEScatter]
        other_particles = [p for p in partitioner.particles if p.sem_type != SemanticType.kLEScatter]

        return [
            (sem5_p.id, other_p.id)
            for sem5_p in sem5_particles
            for other_p in other_particles
        ]

    def _select_one_merge_per_sem5(self,
                                   partitioner: 'ParticlePartitioner',
                                   touching_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Keep one touching partner per ``kLEScatter`` partition.

        Groups pairs by their ``kLEScatter`` member.  When it touches more
        than one partition, the partner is the first of these that leaves
        one (:meth:`_partner_keys`):

        1. the partition holding its nearest ancestor -- its parent's, else
           its grandparent's, and so on: a low-energy piece goes where it
           came from, so a delta ray goes to its track;
        2. the partition holding its nearest relative, by the steps up to a
           common ancestor and then down from it;
        3. a partition of its own interaction;
        4. the most contact: LE points within D of the partner;
        5. the closest point;
        6. the earliest creation time of the partner's representative;
        7. the smallest representative id (an exact time tie: products of
           one Geant4 step, or primaries of one vertex).

        Every key is a property of the event -- genealogy, geometry, Geant4
        truth, the reader's ids -- so the choice does not depend on the
        order anything is listed in.  Records diagnostic rejections for the
        others.

        Parameters
        ----------
        partitioner : ParticlePartitioner
            Active partitioner providing ``particle_lookup``,
            ``rep_lookup`` and ``diagnostics``.
        touching_pairs : list of tuple[int, int]
            All pairs that passed the proximity check.

        Returns
        -------
        list of tuple[int, int]
            Filtered list with at most one entry per ``kLEScatter`` particle.
        """
        sem5_merges: dict = defaultdict(list)

        for id1, id2 in touching_pairs:
            p1 = partitioner.particle_lookup[id1]
            p2 = partitioner.particle_lookup[id2]

            if p1.sem_type == SemanticType.kLEScatter:
                sem5_merges[p1.id].append((id1, id2))
            elif p2.sem_type == SemanticType.kLEScatter:
                sem5_merges[id2].append((id1, id2))

        selected = []
        contact = getattr(self, "_contact", None) or {}

        for sem5_id in sorted(sem5_merges):
            pairs = sem5_merges[sem5_id]
            if len(pairs) == 1:
                selected.append(pairs[0])
                continue
            partner = lambda pr: pr[0] if pr[0] != sem5_id else pr[1]
            keys = self._partner_keys(partitioner, sem5_id,
                                      [partner(pr) for pr in pairs], contact)
            selected_pair = min(pairs, key=lambda pr: keys[partner(pr)])
            selected.append(selected_pair)

            if partitioner.diagnostics.enabled:
                selected_partner_id = partner(selected_pair)
                for pair in pairs:
                    if pair == selected_pair:
                        continue
                    partitioner.diagnostics.record(
                        pair[0], pair[1],
                        MergeOutcome.ALREADY_MERGED_WITH_OTHER,
                        f"SemID-5 particle {sem5_id} already merged with "
                        f"particle {selected_partner_id} "
                        f"(rejecting {partner(pair)})",
                        stage=f"{self.name}: Stage 3 Selection",
                    )

        return selected

    @staticmethod
    def _partner_keys(partitioner: 'ParticlePartitioner', le_id: int,
                      partners: List[int], contact: dict) -> dict:
        """
        The sort key of each partner of the ``kLEScatter`` partition
        *le_id*; the smallest wins (see :meth:`_select_one_merge_per_sem5`).

        The genealogy is that of the partition's representative *le_id*.  An
        ancestor or relative counts for the partition that holds it now.
        """
        lookup = partitioner.particle_lookup
        reps = getattr(partitioner, "rep_lookup", None) or {}

        def rep_of(pid):
            r = reps.get(pid)
            return r.id if r is not None else pid

        targets = set(partners)
        # The representative's parent chain, nearest first.
        chain, cur, seen = [le_id], lookup.get(le_id), {le_id}
        while cur is not None:
            par = cur.parent_id
            if par is None or par in seen or par not in lookup:
                break
            chain.append(par)
            seen.add(par)
            cur = lookup[par]

        # 1. nearest ancestor
        kin = {}
        for depth, a in enumerate(chain[1:], 1):
            q = rep_of(a)
            if q in targets and q not in kin:
                kin[q] = (0, depth, 0)
        # 2. nearest relative: from each ancestor in turn (the piece itself
        #    first), the nearest descendants that sit in a partner
        if not kin:
            children = AbsorbLEScatter._children(partitioner)
            for up, a in enumerate(chain):
                level, down, seen_d = [a], 0, {a}
                while level and not kin:
                    down += 1
                    nxt = []
                    for x in level:
                        for c in children.get(x, ()):
                            if c in seen_d:
                                continue
                            seen_d.add(c)
                            nxt.append(c)
                            q = rep_of(c)
                            if q in targets and q not in kin:
                                kin[q] = (1, up, down)
                    level = nxt
                if kin:
                    break

        own = reps.get(le_id) or lookup.get(le_id)
        own_int = getattr(own, "_interaction_id", None)
        keys = {}
        for q in partners:
            rq = reps.get(q) or lookup.get(q)
            n, d = contact.get((le_id, q), (0, np.inf))
            t = getattr(rq, "creation_time", np.nan)
            keys[q] = (kin.get(q, (2, 0, 0)),
                       0 if getattr(rq, "_interaction_id", None) == own_int else 1,
                       -n, d,
                       t if t == t else np.inf,
                       q)
        return keys

    @staticmethod
    def _children(partitioner: 'ParticlePartitioner') -> dict:
        """Particle id -> its children's ids, made once per partitioner."""
        ch = getattr(partitioner, "_le_children", None)
        if ch is None:
            ch = defaultdict(list)
            for p in partitioner.particle_lookup.values():
                par = p.parent_id
                if par is not None and par != p.id:
                    ch[par].append(p.id)
            partitioner._le_children = ch
        return ch
