"""
Condition: CombineLEScatters
=============================
Collapse all ``kLEScatter`` particles that are transitively connected through
spatial proximity into a single consolidated ``kLEScatter`` particle before the
main ``AbsorbLEScatter`` stage.

Motivation
----------
``AbsorbLEScatter`` makes *pairwise* merge decisions: it absorbs a
``kLEScatter`` particle into the first touching non-``kLEScatter`` neighbour it
finds.  When a chain exists — e.g. Shower A touches LEScatter B which touches
LEScatter C, but A does not directly touch C — the transitively connected
particles B and C are not guaranteed to end up in the same partition.

Running this condition **before** ``AbsorbLEScatter`` resolves the chain by
first grouping B and C into one combined ``kLEScatter``, which then touches A
and is absorbed as a whole.

Algorithm
---------
Proximity graph is built over ``kLEScatter`` particles only.  Connected
components (found iteratively by the partition-level incremental engine)
define the merged groups.

Representative selection
------------------------
Within each LEScatter group the particle **created earliest** (Geant4 start
time, ``creation_time``) is chosen as the representative (i.e. its scalar
attributes — id, pdg, sem_type, ancestry — survive), and on equal times the
smaller id.  Point clouds of all group members are concatenated onto the
representative.

The representative matters beyond its id: ``AbsorbLEScatter`` and the
instance building read the group's genealogy from it.  An ancestor is
created before its descendants, so a photon, not its photoelectron,
represents a group holding both, and the group follows the photon into its
shower.  The choice is made once per group of touching pieces
(:meth:`CombineLEScatters.post_filter`), not pair by pair: merging pairs one
after another would let the order they are applied in decide which piece
ends up representing the group.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np

from .base import PartitionConditionBase
from ..utils import SemanticType

if TYPE_CHECKING:
    from ..partitioner import ParticlePartitioner
    from ..data import Particle


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _rep_key(partitioner: 'ParticlePartitioner', pid: int) -> tuple:
    """
    Sort key of a group member as its group's representative; the smallest
    wins: the earliest Geant4 creation time (``creation_time``), then the
    smallest id.  An ancestor is made before its descendants, so of a
    lineage the ancestor represents the group -- which is what the
    genealogy read from the representative later needs: the instance a
    photon's group joins follows the photon, not its photoelectron.
    """
    t = getattr(partitioner.particle_lookup[pid], "creation_time", np.nan)
    return (t if t == t else np.inf, pid)


# ---------------------------------------------------------------------------
# Condition class
# ---------------------------------------------------------------------------

class CombineLEScatters(PartitionConditionBase):
    """
    Merge all mutually touching ``kLEScatter`` particles into groups before
    the ``AbsorbLEScatter`` absorption stage.

    This condition should be placed in the condition pipeline **before**
    ``AbsorbLEScatter`` so that transitive LEScatter chains are collapsed
    into a single particle prior to shower absorption.

    Only ``kLEScatter`` × ``kLEScatter`` pairs are considered; all other
    semantic types are ignored.

    Pipeline
    --------
    1. ``get_candidates`` returns all unordered ``kLEScatter`` pairs for the
       initial (particle-level) proximity pass.
    2. ``get_rep_candidates`` returns the touching pairs among current
       ``kLEScatter`` representatives, re-evaluated on every convergence
       pass.
    3. ``post_filter`` merges every group of touching LEScatter pieces into
       one representative chosen for the whole group (:func:`_rep_key`).
    """

    @property
    def name(self) -> str:
        return "CombineLEScatters"

    @property
    def description(self) -> str:
        return (
            "Consolidate transitively touching kLEScatter particles into one "
            "before shower absorption"
        )

    # ------------------------------------------------------------------
    # PartitionConditionBase interface
    # ------------------------------------------------------------------

    def post_filter(self,
                    partitioner: 'ParticlePartitioner',
                    touching_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge each group of touching ``kLEScatter`` pieces into one
        representative.

        The groups are the connected components of *touching_pairs*, so
        they do not depend on the pairs' order.  Each group's
        representative is its member with the smallest :func:`_rep_key`,
        and every other member is merged into it.

        Returns
        -------
        list of tuple[int, int]
            ``(member_id, representative_id)`` for every member that is not
            its group's representative.
        """
        root: Dict[int, int] = {}

        def find(x):
            root.setdefault(x, x)
            while root[x] != x:
                root[x] = root[root[x]]
                x = root[x]
            return x

        for a, b in touching_pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                root[max(ra, rb)] = min(ra, rb)
        groups: Dict[int, List[int]] = {}
        for x in root:
            groups.setdefault(find(x), []).append(x)
        merges = []
        for members in groups.values():
            rep = min(members, key=lambda m: _rep_key(partitioner, m))
            merges.extend((m, rep) for m in sorted(members) if m != rep)
        return sorted(merges)

    def get_candidates(
            self, partitioner: 'ParticlePartitioner') -> List[Tuple[int, int]]:
        """
        Return all unordered pairs of ``kLEScatter`` particles.

        Used only in the legacy particle-level proximity path.  In the
        default partition-level path, :meth:`get_rep_candidates` is called
        instead.

        Parameters
        ----------
        partitioner : ParticlePartitioner

        Returns
        -------
        list of tuple[int, int]
            All ``(le_id_a, le_id_b)`` combinations, ``a < b`` by list order.
        """
        le = [p for p in partitioner.particles
              if p.sem_type == SemanticType.kLEScatter]
        candidates = [
            (le[i].id, le[j].id)
            for i in range(len(le))
            for j in range(i + 1, len(le))
        ]
        partitioner.diagnostics.record_candidates(
            f"{self.name}: Stage 1 Candidate Building", candidates
        )
        return candidates

    def get_rep_candidates(
            self,
            partitioner: 'ParticlePartitioner',
            rep_lookup: Dict[int, 'Particle']) -> List[Tuple[int, int]]:
        """
        Return the touching pairs among the current unique ``kLEScatter``
        representatives.

        Called on every convergence pass so that newly consolidated
        representatives (with grown point clouds) are re-evaluated for
        further merges.

        The pairs carry no direction: :meth:`post_filter` chooses each
        group's representative.

        Parameters
        ----------
        partitioner : ParticlePartitioner
        rep_lookup : dict[int, Particle]
            Maps original particle ID → current representative Particle.

        Returns
        -------
        list of tuple[int, int]
            ``(rep_id_a, rep_id_b)`` for all unique pairs.
        """
        # Collect unique kLEScatter representatives from the live partition
        # state.  Using unique_reps() rather than iterating partitioner.particles
        # ensures that already-absorbed particles are never re-inspected:
        # an absorbed particle's rep_lookup entry points to its absorber, which
        # is deduplicated away by unique_reps(), leaving only active reps.
        unique_le = [
            rep
            for rep in self.unique_reps(rep_lookup)
            if rep.sem_type == SemanticType.kLEScatter
        ]

        if len(unique_le) < 2:
            partitioner.diagnostics.record_candidates(
                f"{self.name}: Rep Candidate Building", []
            )
            return []

        D = partitioner.checker.D

        # Global point-cloud KDTree: concatenate all LE rep clouds into one
        # array labelled by rep index, then call query_pairs(D) once.  This
        # finds all touching (rep_i, rep_j) pairs in O(N log N) — strictly
        # better than O(n_reps²) centroid pre-filter followed by per-pair
        # cloud checks, regardless of whether point clouds are voxelized.
        def _xyz(rep):
            parts = rep._cloud_parts
            return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

        clouds   = [_xyz(r) for r in unique_le]
        nonempty = [(k, c) for k, c in enumerate(clouds) if len(c) > 0]

        if len(nonempty) < 2:
            partitioner.diagnostics.record_candidates(
                f"{self.name}: Rep Candidate Building", []
            )
            return []

        rep_tags = np.repeat(
            [k for k, _ in nonempty],
            [len(c) for _, c in nonempty],
        ).astype(np.int32)
        pts = np.concatenate([c[:, :3] for _, c in nonempty], axis=0).astype(np.float32)

        from scipy.spatial import KDTree
        pt_pairs = KDTree(pts).query_pairs(r=D, output_type='ndarray')

        # Map point-pairs → rep-pairs, drop same-rep and deduplicate.
        if len(pt_pairs) == 0:
            partitioner.diagnostics.record_candidates(
                f"{self.name}: Rep Candidate Building", []
            )
            return []

        ri = rep_tags[pt_pairs[:, 0]]
        rj = rep_tags[pt_pairs[:, 1]]
        mask = ri != rj
        if not mask.any():
            partitioner.diagnostics.record_candidates(
                f"{self.name}: Rep Candidate Building", []
            )
            return []

        # Canonical (min, max) ordering → unique set of rep-pairs.
        rep_pairs = np.unique(
            np.sort(np.stack([ri[mask], rj[mask]], axis=1), axis=1), axis=0
        )

        candidates: List[Tuple[int, int]] = []
        for i, j in rep_pairs:
            candidates.append((unique_le[i].id, unique_le[j].id))

        partitioner.diagnostics.record_candidates(
            f"{self.name}: Rep Candidate Building", candidates
        )
        return candidates
