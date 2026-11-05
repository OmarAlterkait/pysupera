"""Tests for partitioning conditions."""
import numpy as np
import pytest

from pysupera.conditions.photon_decay import PhotonDecay
from pysupera.conditions.touching_em_shower import TouchingEMShower
from pysupera.conditions.combine_le_scatters import CombineLEScatters
from pysupera.conditions.absorb_le_scatter import AbsorbLEScatter
from pysupera.utils import SemanticType
from tests.conftest import (
    make_particle, cloud,
    PT_PRIMARY, PT_TRACK, PT_IONIZATION, PT_COMPTON, PT_PHOTON, PT_DELTA,
    PT_CONVERSION, PT_PHOTO_ELEC,
)


def _part(*particles, D=5.0):
    """Build a ParticlePartitioner for the given particles."""
    from tests.conftest import make_partitioner
    return make_partitioner(list(particles), D=D)


def _find_partition(partitions, pid):
    """Return the partition that contains the particle with the given id."""
    for part in partitions:
        if any(p.id == pid for p in part):
            return part
    raise KeyError(pid)


def _same_partition(partitions, pid_a, pid_b):
    """True iff pid_a and pid_b are in the same partition."""
    return _find_partition(partitions, pid_a) is _find_partition(partitions, pid_b)


# ============================================================================
# PhotonDecay
# ============================================================================

class TestPhotonDecay:
    def test_electron_child_of_photon_merges(self):
        photon = make_particle(1, PT_PHOTON, pdg=22,
                               pc=np.zeros((0, 3), dtype=np.float32),
                               parent_id=1, ancestor_id=1)
        electron = make_particle(2, PT_CONVERSION, pdg=11, parent_pdg=22,
                                  parent_id=1, ancestor_id=1,
                                  offset=(0, 0, 0))
        prt = _part(photon, electron)
        parts = prt.partition(PhotonDecay(), verbose=False)
        assert _same_partition(parts, 1, 2)

    def test_positron_child_of_photon_merges(self):
        photon = make_particle(1, PT_PHOTON, pdg=22,
                               pc=np.zeros((0, 3), dtype=np.float32),
                               parent_id=1, ancestor_id=1)
        positron = make_particle(2, PT_CONVERSION, pdg=-11, parent_pdg=22,
                                  parent_id=1, ancestor_id=1,
                                  offset=(0.1, 0, 0))
        prt = _part(photon, positron)
        parts = prt.partition(PhotonDecay(), verbose=False)
        assert _same_partition(parts, 1, 2)

    def test_both_decay_products_in_same_partition(self):
        photon = make_particle(1, PT_PHOTON, pdg=22,
                               pc=np.zeros((0, 3), dtype=np.float32),
                               parent_id=1, ancestor_id=1)
        electron = make_particle(2, PT_CONVERSION, pdg=11, parent_pdg=22,
                                  parent_id=1, ancestor_id=1)
        positron = make_particle(3, PT_CONVERSION, pdg=-11, parent_pdg=22,
                                  parent_id=1, ancestor_id=1,
                                  offset=(0.2, 0, 0))
        prt = _part(photon, electron, positron)
        parts = prt.partition(PhotonDecay(), verbose=False)
        assert _same_partition(parts, 1, 2)
        assert _same_partition(parts, 1, 3)

    def test_compton_electron_stays_its_own_particle(self):
        """A shower begins at the pair production: a Compton electron is not
        merged into its photon, however close or far."""
        photon = make_particle(1, PT_PHOTON, pdg=22,
                               pc=np.zeros((0, 3), dtype=np.float32),
                               parent_id=1, ancestor_id=1)
        compton = make_particle(2, PT_COMPTON, pdg=11, parent_pdg=22,
                                parent_id=1, ancestor_id=1, n_pts=2)
        photo = make_particle(3, PT_PHOTO_ELEC, pdg=11, parent_pdg=22,
                              parent_id=1, ancestor_id=1, n_pts=2,
                              offset=(30, 0, 0))
        photon.sem_type = SemanticType.kLEScatter   # a tiny photon blob
        prt = _part(photon, compton, photo)
        parts = prt.partition(PhotonDecay(), verbose=False)
        assert not _same_partition(parts, 1, 2)
        assert not _same_partition(parts, 1, 3)
        # no pair, no shower: the photon is not promoted
        assert prt.rep_lookup[1].sem_type == SemanticType.kLEScatter

    def test_conversion_promotes_the_photon_and_leaves_compton_out(self):
        photon = make_particle(1, PT_PHOTON, pdg=22,
                               pc=np.zeros((0, 3), dtype=np.float32),
                               parent_id=1, ancestor_id=1)
        compton = make_particle(2, PT_COMPTON, pdg=11, parent_pdg=22,
                                parent_id=1, ancestor_id=1, n_pts=2,
                                offset=(80, 0, 0))
        pair = make_particle(3, PT_CONVERSION, pdg=11, parent_pdg=22,
                             parent_id=1, ancestor_id=1, n_pts=20)
        photon.sem_type = SemanticType.kLEScatter
        prt = _part(photon, compton, pair)
        parts = prt.partition_combined([PhotonDecay()], verbose=False)
        assert _same_partition(parts, 1, 3)
        assert not _same_partition(parts, 1, 2)
        assert prt.rep_lookup[1].sem_type == SemanticType.kShower

    def test_non_photon_parent_no_merge(self):
        # parent is a track (parent_pdg=13, not 22) → no PhotonDecay merge
        track = make_particle(1, PT_TRACK, pdg=13,
                               parent_id=1, ancestor_id=1,
                               offset=(0, 0, 0))
        electron = make_particle(2, PT_PRIMARY, pdg=11, parent_pdg=13,
                                  parent_id=1, ancestor_id=1,
                                  offset=(50, 0, 0))  # far away
        prt = _part(track, electron)
        parts = prt.partition(PhotonDecay(), verbose=False)
        assert not _same_partition(parts, 1, 2)


# ============================================================================
# TouchingEMShower
# ============================================================================

class TestTouchingEMShower:
    def test_touching_parent_child_electrons_merge(self):
        parent = make_particle(1, PT_PRIMARY, pdg=11,
                                parent_id=1, ancestor_id=1,
                                offset=(0, 0, 0))
        child = make_particle(2, PT_COMPTON, pdg=11,
                               parent_id=1, ancestor_id=1,
                               offset=(0.05, 0, 0))  # very close → touching
        prt = _part(parent, child)
        parts = prt.partition(TouchingEMShower(), verbose=False)
        assert _same_partition(parts, 1, 2)

    def test_different_ancestor_no_merge(self):
        parent = make_particle(1, PT_PRIMARY, pdg=11,
                                parent_id=1, ancestor_id=1,
                                offset=(0, 0, 0))
        # child has different ancestor_id → TouchingEMShower filter rejects
        child = make_particle(2, PT_COMPTON, pdg=11,
                               parent_id=1, ancestor_id=99,   # different ancestor
                               offset=(0.05, 0, 0))
        prt = _part(parent, child)
        parts = prt.partition(TouchingEMShower(), verbose=False)
        assert not _same_partition(parts, 1, 2)

    def test_non_touching_parent_child_not_merged(self):
        parent = make_particle(1, PT_PRIMARY, pdg=11,
                                parent_id=1, ancestor_id=1,
                                offset=(0, 0, 0))
        child = make_particle(2, PT_COMPTON, pdg=11,
                               parent_id=1, ancestor_id=1,
                               offset=(100, 0, 0))  # far away
        prt = _part(parent, child, D=5.0)
        parts = prt.partition(TouchingEMShower(), verbose=False)
        assert not _same_partition(parts, 1, 2)

    def test_track_particle_not_included_as_candidate(self):
        # A track (pdg=13) parent should not produce EM shower candidates
        parent = make_particle(1, PT_TRACK, pdg=13,
                                parent_id=1, ancestor_id=1,
                                offset=(0, 0, 0))
        child = make_particle(2, PT_COMPTON, pdg=11,
                               parent_id=1, ancestor_id=1,
                               offset=(0.05, 0, 0))
        prt = _part(parent, child)
        parts = prt.partition(TouchingEMShower(), verbose=False)
        assert not _same_partition(parts, 1, 2)


# ============================================================================
# CombineLEScatters
# ============================================================================

class TestCombineLEScatters:
    def test_two_touching_le_scatters_merge(self):
        le1 = make_particle(1, PT_IONIZATION, pdg=11,
                             parent_id=1, ancestor_id=1,
                             offset=(0, 0, 0))
        le2 = make_particle(2, PT_IONIZATION, pdg=11,
                             parent_id=2, ancestor_id=2,
                             offset=(0.05, 0, 0))  # touching
        assert le1.sem_type == SemanticType.kLEScatter
        assert le2.sem_type == SemanticType.kLEScatter
        prt = _part(le1, le2)
        parts = prt.partition(CombineLEScatters(), verbose=False)
        assert _same_partition(parts, 1, 2)

    def test_non_touching_le_scatters_not_merged(self):
        le1 = make_particle(1, PT_IONIZATION, pdg=11, offset=(0, 0, 0))
        le2 = make_particle(2, PT_IONIZATION, pdg=11, offset=(100, 0, 0))
        prt = _part(le1, le2, D=5.0)
        parts = prt.partition(CombineLEScatters(), verbose=False)
        assert not _same_partition(parts, 1, 2)

    def test_shower_not_absorbed_by_combine_le(self):
        # CombineLEScatters only looks at LE-LE pairs → shower unaffected
        le = make_particle(1, PT_IONIZATION, pdg=11, offset=(0, 0, 0))
        shower = make_particle(2, PT_PRIMARY, pdg=11, offset=(0.05, 0, 0))
        assert shower.sem_type == SemanticType.kShower
        prt = _part(le, shower)
        parts = prt.partition(CombineLEScatters(), verbose=False)
        assert not _same_partition(parts, 1, 2)

    def test_transitive_chain_all_merged(self):
        # A touches B, B touches C, A does not touch C → expect 1 partition
        le_a = make_particle(1, PT_IONIZATION, pdg=11, offset=(0,    0, 0))
        le_b = make_particle(2, PT_IONIZATION, pdg=11, offset=(0.05, 0, 0))
        le_c = make_particle(3, PT_IONIZATION, pdg=11, offset=(0.10, 0, 0))
        prt = _part(le_a, le_b, le_c, D=0.1)   # D just enough to link neighbors
        parts = prt.partition(CombineLEScatters(), verbose=False)
        # All three should be in one partition
        assert _same_partition(parts, 1, 2)
        assert _same_partition(parts, 2, 3)

    def test_representative_is_the_earliest_created(self):
        # the photon, not its larger photoelectron, represents the group
        photon = make_particle(1, PT_IONIZATION, pdg=22, n_pts=3, offset=(0, 0, 0))   # LE
        elec   = make_particle(2, PT_PHOTO_ELEC, pdg=11, parent_id=1, ancestor_id=1,
                               n_pts=15, offset=(0.05, 0, 0))
        photon.creation_time, elec.creation_time = 10.0, 12.0
        prt = _part(elec, photon)
        prt.partition(CombineLEScatters(), verbose=False)
        assert prt.get_representative(2).id == prt.get_representative(1).id == 1

    def test_representative_on_equal_times_is_the_smallest_id(self):
        le1 = make_particle(1, PT_IONIZATION, pdg=11, n_pts=3,  offset=(0,    0, 0))
        le2 = make_particle(2, PT_IONIZATION, pdg=11, n_pts=15, offset=(0.05, 0, 0))
        le1.creation_time = le2.creation_time = 4.0
        prt = _part(le2, le1)
        prt.partition(CombineLEScatters(), verbose=False)
        assert prt.get_representative(2).id == 1


# ============================================================================
# AbsorbLEScatter
# ============================================================================

class TestAbsorbLEScatter:
    def test_le_scatter_absorbed_by_touching_shower(self):
        shower = make_particle(1, PT_PRIMARY, pdg=11, offset=(0,    0, 0))
        le     = make_particle(2, PT_IONIZATION, pdg=11, offset=(0.05, 0, 0))
        assert shower.sem_type == SemanticType.kShower
        assert le.sem_type     == SemanticType.kLEScatter
        prt = _part(shower, le)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        assert _same_partition(parts, 1, 2)

    def test_isolated_le_scatter_stays_singleton(self):
        le = make_particle(1, PT_IONIZATION, pdg=11, offset=(0, 0, 0))
        # No other particle nearby
        prt = _part(le)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        assert len(parts) == 1
        assert len(parts[0]) == 1

    def test_le_scatter_not_close_to_shower_not_absorbed(self):
        shower = make_particle(1, PT_PRIMARY, pdg=11, offset=(0,   0, 0))
        le     = make_particle(2, PT_IONIZATION, pdg=11, offset=(100, 0, 0))
        prt = _part(shower, le, D=5.0)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        assert not _same_partition(parts, 1, 2)

    def test_le_scatter_absorbed_by_at_most_one_neighbor(self):
        # Two showers close to one LE scatter → LE absorbed into exactly one
        shower_a = make_particle(1, PT_PRIMARY, pdg=11, offset=(0,    0, 0))
        shower_b = make_particle(2, PT_PRIMARY, pdg=11, offset=(0.2,  0, 0))
        le       = make_particle(3, PT_IONIZATION, pdg=11, offset=(0.1, 0, 0))
        prt = _part(shower_a, shower_b, le)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        # All three may end up together (via le), but le is in exactly one partition
        le_part = _find_partition(parts, 3)
        assert le_part is not None
        # shower_a and shower_b may or may not be together; le belongs to one
        assert len(le_part) >= 2

    def test_le_scatter_absorbed_into_track(self):
        track = make_particle(1, PT_TRACK, pdg=13, offset=(0, 0, 0))
        le    = make_particle(2, PT_IONIZATION, pdg=11, offset=(0.05, 0, 0))
        assert track.sem_type == SemanticType.kTrack
        prt = _part(track, le)
        parts = prt.partition(AbsorbLEScatter(), verbose=False)
        assert _same_partition(parts, 1, 2)


class TestAbsorbLEScatterChoice:
    """
    Which partner an LE piece touching several partitions goes to.  Each
    test puts the rule under test against a later one that would choose
    the other partner.  D = 0.6 throughout, so the partners (0.8 or more
    apart) never touch each other.
    """
    D = 0.6

    def absorb(self, *ps):
        prt = _part(*ps, D=self.D)
        prt.partition(AbsorbLEScatter(), verbose=False)
        return prt

    def test_nearest_ancestor_beyond_the_parent(self):
        # the LE piece's parent is far away; its grandparent, a track, touches
        track  = make_particle(1, PT_TRACK, pdg=13, pc=cloud([(x, 0, 0) for x in range(11)]))
        parent = make_particle(2, PT_PRIMARY, pdg=11, parent_id=1, ancestor_id=1,
                               pc=cloud([(100, 0, 0)]))
        le     = make_particle(3, PT_IONIZATION, pdg=11, parent_id=2, ancestor_id=1,
                               pc=cloud([(5, 0.5, 0)]))
        other  = make_particle(4, PT_PRIMARY, pdg=11, pc=cloud([(5, 0.8, 0)]))   # closer
        prt = self.absorb(track, parent, le, other)
        assert prt.get_representative(3).id == 1

    def test_nearest_relative_when_no_ancestor_touches(self):
        top     = make_particle(1, PT_PRIMARY, pdg=22, pc=cloud([(100, 0, 0)]))
        parent  = make_particle(2, PT_PRIMARY, pdg=22, parent_id=1, ancestor_id=1,
                                pc=cloud([(200, 0, 0)]))
        le      = make_particle(3, PT_IONIZATION, pdg=11, parent_id=2, ancestor_id=1,
                                pc=cloud([(0, 0, 0)]))
        sibling = make_particle(4, PT_PRIMARY, pdg=11, parent_id=2, ancestor_id=1,
                                pc=cloud([(0.5, 0, 0)]))
        cousin  = make_particle(5, PT_PRIMARY, pdg=11, parent_id=1, ancestor_id=1,
                                pc=cloud([(-0.5, 0, 0)]))
        stranger = make_particle(6, PT_PRIMARY, pdg=11, pc=cloud([(0, 0.3, 0)]))  # closest
        prt = self.absorb(top, parent, le, sibling, cousin, stranger)
        assert prt.get_representative(3).id == 4

    def test_own_interaction_before_geometry(self):
        le   = make_particle(1, PT_IONIZATION, pdg=11, interaction_id=0, pc=cloud([(0, 0, 0)]))
        same = make_particle(2, PT_PRIMARY, pdg=11, interaction_id=0, pc=cloud([(0.5, 0, 0)]))
        near = make_particle(3, PT_PRIMARY, pdg=11, interaction_id=1, pc=cloud([(0, 0.3, 0)]))
        prt = self.absorb(le, same, near)
        assert prt.get_representative(1).id == 2

    def test_most_contact_before_closest_point(self):
        le   = make_particle(1, PT_IONIZATION, pdg=11,
                             pc=cloud([(0, 0, 0), (1, 0, 0), (2, 0, 0)]))
        wide = make_particle(2, PT_PRIMARY, pdg=11, pc=cloud([(x, 0.5, 0) for x in range(3)]))
        near = make_particle(3, PT_PRIMARY, pdg=11, pc=cloud([(0, -0.3, 0)]))
        prt = self.absorb(le, wide, near)
        assert prt.get_representative(1).id == 2

    def test_closest_point_on_equal_contact(self):
        le   = make_particle(1, PT_IONIZATION, pdg=11, pc=cloud([(0, 0, 0)]))
        far  = make_particle(2, PT_PRIMARY, pdg=11, pc=cloud([(0.5, 0, 0)]))
        near = make_particle(3, PT_PRIMARY, pdg=11, pc=cloud([(0, 0.4, 0)]))
        prt = self.absorb(le, far, near)
        assert prt.get_representative(1).id == 3

    def test_earliest_creation_then_smallest_id(self):
        def scene(t2, t3):
            le = make_particle(1, PT_IONIZATION, pdg=11, pc=cloud([(0, 0, 0)]))
            a  = make_particle(2, PT_PRIMARY, pdg=11, pc=cloud([(0.5, 0, 0)]))
            b  = make_particle(3, PT_PRIMARY, pdg=11, pc=cloud([(-0.5, 0, 0)]))
            a.creation_time, b.creation_time = t2, t3
            return self.absorb(le, a, b).get_representative(1).id
        assert scene(5.0, 2.0) == 3
        assert scene(2.0, 5.0) == 2
        assert scene(2.0, 2.0) == 2


def _random_scene(seed, n=80):
    """
    LE pieces and showers on a coarse grid (so many distances tie), with a
    random genealogy and two interactions.
    """
    rng = np.random.default_rng(seed)
    ps = []
    for i in range(n):
        le = i % 3 != 0
        par = int(rng.integers(0, i)) if i and rng.random() < 0.7 else i
        k = int(rng.integers(1, 4 if le else 12))
        pc = (rng.integers(0, 8, (k, 3)) * 0.5).astype(np.float32)
        p = make_particle(i, PT_IONIZATION if le else PT_PRIMARY, pdg=11,
                          parent_id=par, interaction_id=int(rng.integers(0, 2)), pc=pc)
        p.creation_time = float(rng.integers(0, 3))
        ps.append(p)
    return ps


class TestLEOrderIndependence:
    """The same event, its particles listed in another order, partitions alike."""

    @pytest.mark.parametrize("seed", range(6))
    def test_combine_then_absorb(self, seed):
        from pysupera.partitioner import ParticlePartitioner

        def run(order):
            ps = _random_scene(seed)
            prt = ParticlePartitioner([ps[i] for i in order], distance_threshold=0.6,
                                      backend="cpu-single", enable_diagnostics=False)
            prt.partition_combined([CombineLEScatters(), AbsorbLEScatter()], verbose=False)
            return {i: prt.get_representative(i).id for i in range(len(ps))}

        n = len(_random_scene(seed))
        ref = run(range(n))
        rng = np.random.default_rng(100 + seed)
        assert len(set(ref.values())) < n          # there were merges
        for _ in range(3):
            assert run(rng.permutation(n)) == ref

    @pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 2, 0), (1, 0, 2)])
    def test_combined_le_group_has_one_representative_in_any_order(self, order):
        # A touches B, B touches C, A does not touch C; C is created first
        a = make_particle(1, PT_IONIZATION, pdg=11, pc=cloud([(0, y, 0) for y in range(5)]))
        b = make_particle(2, PT_IONIZATION, pdg=11, pc=cloud([(0.5, y, 0) for y in range(2)]))
        c = make_particle(3, PT_IONIZATION, pdg=11, pc=cloud([(1.0, y, 3) for y in range(4)]))
        c.point_cloud[0] = (1.0, 0, 0)
        a.creation_time, b.creation_time, c.creation_time = 5.0, 3.0, 1.0
        ps = [a, b, c]
        prt = _part(*[ps[i] for i in order], D=0.6)
        prt.partition(CombineLEScatters(), verbose=False)
        assert {prt.get_representative(i).id for i in (1, 2, 3)} == {3}


# ============================================================================
# PhotonDecay: re-attaching the photon's own split-off pieces
# ============================================================================

from pysupera.conditions.photon_decay import _is_own_fragment   # noqa: E402


class _Stub:
    def __init__(self, pid, pdg, parent_id, parent_pdg, g4):
        self.id, self.pdg = pid, pdg
        self.parent_id, self.parent_pdg = parent_id, parent_pdg
        self.geant4_id = g4


class TestIsOwnFragment:
    """
    A split-off piece IS the parent photon, so it shares its geant4_id.  A
    fluorescence X-ray is a distinct particle and must not match, or two
    genuinely separate objects get merged.
    """

    def test_split_off_piece_matches(self):
        parent = _Stub(1, 22, 1, 0, 500)
        child = _Stub(9, 22, 1, 22, 500)
        assert _is_own_fragment(child, parent)

    def test_fluorescence_photon_does_not_match(self):
        parent = _Stub(1, 22, 1, 0, 500)
        child = _Stub(9, 22, 1, 22, 501)       # its own Geant4 track
        assert not _is_own_fragment(child, parent)

    def test_electron_child_does_not_match(self):
        parent = _Stub(1, 22, 1, 0, 500)
        assert not _is_own_fragment(_Stub(9, 11, 1, 22, 500), parent)

    def test_missing_parent_does_not_match(self):
        assert not _is_own_fragment(_Stub(9, 22, 1, 22, 500), None)

    def test_missing_geant4_id_does_not_match(self):
        class NoG4:
            id, pdg, parent_id, parent_pdg = 9, 22, 1, 22
        assert not _is_own_fragment(NoG4(), _Stub(1, 22, 1, 0, 500))
