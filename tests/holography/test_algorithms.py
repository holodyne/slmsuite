"""
Unit tests for slmsuite.holography.algorithms module.
"""
import pytest
import numpy as np
import logging
from types import SimpleNamespace

from slmsuite.holography.algorithms import (
    Hologram, SpotHologram, MultiplaneHologram, FeedbackHologram
)
from slmsuite.holography.algorithms._header import cp_affine_transform
from slmsuite.holography.analysis import Affine

try:
    import cupy as cp
except ImportError:
    cp = None

logger = logging.getLogger(__name__)


def _np(array):
    """Return numpy array regardless of whether input is numpy or cupy."""
    return array.get() if hasattr(array, "get") else array


class TestHologram:

    def test_dtype(self, subtests):
        with subtests.test("float32"):
            h = Hologram(target=np.zeros((64, 64), dtype=np.float32))
            assert h.dtype == np.float32
            assert h.dtype_complex == np.complex64

        with subtests.test("float64"):
            h = Hologram(target=np.zeros((64, 64), dtype=np.float64), dtype=np.float64)
            assert h.dtype == np.float64
            assert h.dtype_complex == np.complex128

    def test_shape(self, subtests):
        with subtests.test("slm_shape defaults to computational shape"):
            h = Hologram(target=np.zeros((64, 128)))
            assert h.slm_shape == (64, 128)
            assert h.shape == (64, 128)

        with subtests.test("slm_shape can differ from computational shape"):
            h = Hologram(target=np.zeros((64, 64)), slm_shape=(32, 32))
            assert h.slm_shape == (32, 32)
            assert h.shape == (64, 64)

        with subtests.test("phase shape matches slm_shape not computational shape"):
            h = Hologram(target=np.zeros((64, 64)), slm_shape=(32, 32))
            assert h.get_phase().shape == (32, 32)

    def test_unpad_slice_tracks_shape_changes(self, subtests):
        """
        `_unpad_slice` memoizes `toolbox.unpad(shape, slm_shape)` for the GS hot path.
        It is keyed on the shapes rather than computed once, because subclasses adjust
        them after `__init__` -- `CompressedSpotHologram` sets `shape = slm_shape`. A
        compute-once cache would go stale there, silently and only for that subclass.
        """
        h = Hologram(target=np.zeros((64, 64)), slm_shape=(32, 32))

        with subtests.test("matches unpad"):
            from slmsuite.holography import toolbox
            assert h._unpad_slice == toolbox.unpad(h.shape, h.slm_shape)

        with subtests.test("repeated access is cached"):
            assert h._unpad_slice is h._unpad_slice

        with subtests.test("invalidates when the shapes change"):
            padded = h._unpad_slice
            h.slm_shape = (64, 64)
            assert h._unpad_slice != padded
            assert h._unpad_slice == (0, 64, 0, 64)

    def test_raises(self, subtests):
        with subtests.test("shape mismatch"):
            with pytest.raises(ValueError):
                Hologram(target=np.zeros((64, 64)), phase=np.zeros((64, 64)), amp=np.ones((32, 32)))

        with subtests.test("invalid method"):
            h = Hologram(target=np.zeros((64, 64)))
            with pytest.raises(ValueError, match="Unrecognized method"):
                h.optimize(method="INVALID", maxiter=1, verbose=False)

        with subtests.test("invalid stat group"):
            h = Hologram(target=np.zeros((64, 64)))
            with pytest.raises(ValueError):
                h.optimize(method="GS", maxiter=1, verbose=False, stat_groups=["INVALID_GROUP"])

    def test_target_normalization(self, subtests):
        with subtests.test("target L2 normalized on construction"):
            raw = np.random.rand(64, 64).astype(np.float32) + 0.1
            h = Hologram(target=raw)
            assert np.isclose(float(np.sum(_np(h.target) ** 2)), 1.0, rtol=1e-4)

        with subtests.test("set_target L2 normalizes"):
            h = Hologram(target=np.zeros((64, 64)))
            h.set_target(np.ones((64, 64)) * 5.0)
            assert np.isclose(float(np.sum(_np(h.target) ** 2)), 1.0, rtol=1e-4)

    def test_phase(self, subtests):
        with subtests.test("phase range after construction"):
            h = Hologram(target=np.zeros((64, 64)))
            phase = h.get_phase()
            assert phase.min() >= 0.0
            assert phase.max() <= 2 * np.pi + 1e-5

        with subtests.test("MRAF no NaN in phase"):
            target = np.full((64, 64), np.nan, dtype=np.float32)
            target[20, 20] = 1.0
            target[40, 40] = 1.0
            h = Hologram(target=target)
            h.optimize(method="GS", maxiter=10, verbose=False)
            assert not np.any(np.isnan(h.get_phase())), "MRAF optimization produced NaN in phase"

    def test_iter(self, subtests):
        with subtests.test("increments with optimize"):
            h = Hologram(target=np.zeros((64, 64)))
            h.optimize(method="GS", maxiter=10, verbose=False)
            assert h.iter == 10

        with subtests.test("consecutive optimize accumulates"):
            h = Hologram(target=np.zeros((64, 64)))
            h.optimize(method="GS", maxiter=5, verbose=False)
            h.optimize(method="GS", maxiter=5, verbose=False)
            assert h.iter == 10

    def test_reset(self, subtests):
        with subtests.test("clears iter and stats"):
            h = Hologram(target=np.zeros((64, 64)))
            h.optimize(method="GS", maxiter=5, verbose=False)
            h.reset()
            assert h.iter == 0
            assert h.stats == {"method": [], "flags": {}, "stats": {}}

        with subtests.test("reset_phase randomizes phase"):
            h = Hologram(target=np.zeros((64, 64)))
            phase_before = h.get_phase().copy()
            h.reset_phase()
            assert not np.allclose(phase_before, h.get_phase()), \
                "reset_phase() should produce a different random phase"

    def test_stats(self, subtests):
        with subtests.test("length matches iterations"):
            target = np.zeros((64, 64))
            target[32, 32] = 1.0
            h = Hologram(target=target)
            N = 10
            h.optimize(method="GS", maxiter=N, verbose=False, stat_groups=["computational"])
            comp = h.stats["stats"]["computational"]
            assert len(comp["efficiency"]) == N
            assert len(comp["uniformity"]) == N
            assert len(comp["std_err"]) == N

        with subtests.test("values are finite"):
            target = np.zeros((64, 64))
            target[20, 30] = 1.0
            target[40, 50] = 1.0
            h = Hologram(target=target)
            h.optimize(method="GS", maxiter=10, verbose=False, stat_groups=["computational"])
            comp = h.stats["stats"]["computational"]
            assert all(np.isfinite(v) for v in comp["efficiency"])
            assert all(np.isfinite(v) for v in comp["uniformity"])
            assert all(np.isfinite(v) for v in comp["std_err"])

    def test_gs_convergence(self, subtests):
        with subtests.test("single spot efficiency > 0.9"):
            target = np.zeros((64, 64))
            target[16, 48] = 1.0
            h = Hologram(target=target)
            h.optimize(method="GS", maxiter=40, verbose=False, stat_groups=["computational"])
            eff = h.stats["stats"]["computational"]["efficiency"][-1]
            assert eff > 0.9, f"Single-spot GS efficiency {eff:.4f} should exceed 0.9"

        with subtests.test("farfield peak at target location"):
            target = np.zeros((64, 64))
            r, c = 20, 44
            target[r, c] = 1.0
            h = Hologram(target=target)
            h.optimize(method="GS", maxiter=40, verbose=False)
            ff = np.abs(h.get_farfield())
            peak = np.unravel_index(np.argmax(ff), ff.shape)
            assert peak == (r, c), f"GS farfield peak at {peak}, expected ({r}, {c})"

        with subtests.test("efficiency improves over iterations"):
            target = np.zeros((64, 64))
            for r, c in [(13, 17), (30, 44), (50, 10), (10, 50)]:
                target[r, c] = 1.0
            h = Hologram(target=target)
            h.optimize(method="GS", maxiter=20, verbose=False, stat_groups=["computational"])
            effs = h.stats["stats"]["computational"]["efficiency"]
            assert effs[-1] > effs[0], "GS efficiency should improve over iterations"

        with subtests.test("WGS-Leonardo std_err decreases"):
            target = np.zeros((64, 64))
            for r, c in [(13, 17), (30, 44), (50, 10), (10, 50)]:
                target[r, c] = 1.0
            h = Hologram(target=target)
            h.optimize(method="WGS-Leonardo", maxiter=30, verbose=False, stat_groups=["computational"])
            errs = h.stats["stats"]["computational"]["std_err"]
            assert errs[-1] <= errs[1], "WGS-Leonardo std_err should decrease from iteration 1 to end"

    @pytest.mark.parametrize("method", ["WGS-Leonardo", "WGS-Kim", "WGS-Nogrette"])
    def test_wgs_uniformity_improves_over_wgs_iterations(self, method):
        target = np.zeros((64, 64))
        for r, c in [(13, 17), (30, 44), (50, 10), (10, 50), (32, 32)]:
            target[r, c] = 1.0
        h = Hologram(target=target)
        h.optimize(method=method, maxiter=30, verbose=False, stat_groups=["computational"])
        unis = h.stats["stats"]["computational"]["uniformity"]
        assert unis[-1] >= unis[1], f"{method} uniformity should not decrease from iteration 1 to end"

    @pytest.mark.parametrize("method", ["GS", "WGS-Leonardo", "WGS-Kim", "WGS-Nogrette"])
    def test_gs_speed(self, random_seed, method, benchmark):
        target = np.zeros((1024, 1024))

        rng = np.random.default_rng(random_seed)
        for i in range(20):
            test_point = (rng.integers(0, 1024), rng.integers(0, 1024))
            target[test_point] = 1
        hologram = Hologram(target=target)
        benchmark(hologram.optimize, method=method, maxiter=20, verbose=False, stat_groups=[])

    @pytest.mark.gpu
    @pytest.mark.parametrize("method", ["GS", "WGS-Leonardo", "WGS-Kim", "WGS-Nogrette"])
    def test_gs_speed_gpu(self, random_seed, method, benchmark, has_cupy):
        import cupy as cp
        target = cp.zeros((1024, 1024))

        rng = np.random.default_rng(random_seed)
        for i in range(20):
            test_point = (rng.integers(0, 1024), rng.integers(0, 1024))
            target[test_point] = 1
        hologram = Hologram(target=target)
        benchmark(hologram.optimize, method=method, maxiter=20, verbose=False, stat_groups=[])

    def test_padded_shape(self, subtests):
        with subtests.test("padding_order=0 returns exact input"):
            for slm_shape in [(128, 128), (100, 200), (64, 64)]:
                assert Hologram.get_padded_shape(slm_shape, padding_order=0, square_padding=False) == slm_shape

        with subtests.test("padding_order=1 at least input size"):
            for slm_shape in [(100, 100), (128, 128), (720, 1280)]:
                padded = Hologram.get_padded_shape(slm_shape, padding_order=1, square_padding=False)
                assert padded[0] >= slm_shape[0]
                assert padded[1] >= slm_shape[1]

        with subtests.test("results are powers of two by default"):
            for slm_shape in [(128, 128), (100, 200), (720, 1280)]:
                padded = Hologram.get_padded_shape(slm_shape)
                assert np.log2(padded[0]) % 1 == 0, f"Height {padded[0]} is not a power of 2"
                assert np.log2(padded[1]) % 1 == 0, f"Width {padded[1]} is not a power of 2"

        with subtests.test("padding_order=2 not smaller than order=1"):
            slm_shape = (128, 128)
            padded1 = Hologram.get_padded_shape(slm_shape, padding_order=1, square_padding=False)
            padded2 = Hologram.get_padded_shape(slm_shape, padding_order=2, square_padding=False)
            assert padded2[0] >= padded1[0]
            assert padded2[1] >= padded1[1]

        with subtests.test("square padding produces equal dimensions"):
            padded = Hologram.get_padded_shape((128, 256), square_padding=True)
            assert padded[0] == padded[1]

        with subtests.test("no square padding pads each dim independently"):
            padded = Hologram.get_padded_shape((128, 256), square_padding=False)
            assert padded[0] >= 128
            assert padded[1] >= 256

        with subtests.test("result never smaller than input"):
            for slm_shape in [(64, 64), (100, 100), (720, 1280), (512, 512)]:
                padded = Hologram.get_padded_shape(slm_shape)
                assert padded[0] >= slm_shape[0]
                assert padded[1] >= slm_shape[1]


class TestSpotHologram:

    def test_spot_hologram(self, subtests):
        with subtests.test("spot places power at correct pixel"):
            shape = (64, 64)
            spot_knm = np.array([[32.0], [32.0]])
            h = SpotHologram(shape=shape, spot_vectors=spot_knm, basis="knm")
            target = _np(h.target)
            assert target[32, 32] > 0
            rest = target.copy()
            rest[32, 32] = 0.0
            assert np.all(np.nan_to_num(rest) == 0)

        with subtests.test("len equals number of spots"):
            shape = (64, 64)
            N = 7
            spots = np.array([[10.0 + 5 * i for i in range(N)],
                              [10.0 + 5 * i for i in range(N)]])
            h = SpotHologram(shape=shape, spot_vectors=spots, basis="knm")
            assert len(h) == N

        with subtests.test("uniform spot amplitude gives equal target pixel powers"):
            shape = (64, 64)
            spots = np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
            h = SpotHologram(shape=shape, spot_vectors=spots, basis="knm")
            target = _np(h.target)
            pixel_powers = [float(target[int(spots[1, i]), int(spots[0, i])] ** 2) for i in range(3)]
            assert np.allclose(pixel_powers, pixel_powers[0], rtol=1e-4), \
                f"Uniform spot amplitudes should give equal target pixel powers: {pixel_powers}"

        with subtests.test("spot out of bounds raises"):
            shape = (64, 64)
            spots = np.array([[100.0], [100.0]])
            with pytest.raises(ValueError, match="[Bb]ounds|bounds"):
                SpotHologram(shape=shape, spot_vectors=spots, basis="knm")

        with subtests.test("GS efficiency > 0.5"):
            shape = (64, 64)
            spots = np.array([[16.0, 48.0], [16.0, 48.0]])
            h = SpotHologram(shape=shape, spot_vectors=spots, basis="knm")
            h.optimize(method="GS", maxiter=30, verbose=False, stat_groups=["computational"])
            eff = h.stats["stats"]["computational"]["efficiency"][-1]
            assert eff > 0.5, f"SpotHologram GS efficiency {eff:.3f} should exceed 0.5"


class TestMultiplaneHologram:

    def test_multiplane_directs_power_to_all_child_targets(self, subtests):
        """
        Two child Holograms target distinct spots. After MultiplaneHologram GS
        optimization, the composite phase should produce farfield power at BOTH
        target locations—something neither child alone would achieve, exercising
        the weighted nearfield summation in _farfield2nearfield.
        """
        shape = (64, 64)
        amp = np.ones(shape, dtype=np.float32) / np.sqrt(np.prod(shape))

        # Child A targets a spot in the top-left quadrant.
        target_a = np.zeros(shape, dtype=np.float32)
        spot_a = (16, 16)
        target_a[spot_a] = 1.0

        # Child B targets a spot in the bottom-right quadrant.
        target_b = np.zeros(shape, dtype=np.float32)
        spot_b = (48, 48)
        target_b[spot_b] = 1.0

        h_a = Hologram(target=target_a, amp=amp)
        h_b = Hologram(target=target_b, amp=amp)
        mph = MultiplaneHologram([h_a, h_b])

        mph.optimize(method="GS", maxiter=40, verbose=False)

        # The children share phase, so the composite farfield is the same for both.
        # Check that the single farfield has significant power at BOTH target spots.
        ff = np.abs(h_a.get_farfield())
        peak_power = ff.max()

        with subtests.test("child A target has significant farfield power"):
            power_a = ff[spot_a]
            assert power_a > 0.3 * peak_power, (
                f"Power at spot A = {power_a:.4f} vs peak {peak_power:.4f}"
            )

        with subtests.test("child B target has significant farfield power"):
            power_b = ff[spot_b]
            assert power_b > 0.3 * peak_power, (
                f"Power at spot B = {power_b:.4f} vs peak {peak_power:.4f}"
            )

        with subtests.test("children share phase (multiplane composition)"):
            # The core invariant: both children reference the SAME phase array.
            assert h_a.phase is h_b.phase

        with subtests.test("len matches number of children"):
            assert len(mph) == 2

        with subtests.test("set_target is forbidden on the meta hologram"):
            with pytest.raises(RuntimeError):
                mph.set_target(np.zeros(shape))

        with subtests.test("non-hologram child is rejected"):
            with pytest.raises(ValueError):
                MultiplaneHologram([h_a, "not a hologram"])

        with subtests.test("nested MultiplaneHologram is rejected"):
            with pytest.raises(ValueError):
                MultiplaneHologram([mph, h_a])


class TestGSEquivalence:
    """
    End-to-end guard for the batched-routine optimization: GS from a fixed starting phase
    must land on the same answer as the legacy per-child code path.
    """

    @staticmethod
    def _run_multiplane(shape, slm_shape, iterations=8):
        amp = np.ones(slm_shape, dtype=np.float32) / np.sqrt(np.prod(slm_shape))
        children = []
        for spot in ((shape[0] // 4, shape[1] // 4), (shape[0] // 2, 3 * shape[1] // 8)):
            target = np.zeros(shape, dtype=np.float32)
            target[spot] = 1.0
            children.append(Hologram(target=target, amp=amp, slm_shape=slm_shape))
        mph = MultiplaneHologram(children)
        mph.reset_phase(random_phase=0)
        mph.optimize(method="GS", maxiter=iterations, verbose=False)
        return mph

    @pytest.mark.parametrize(
        "shape, slm_shape",
        [((64, 64), (64, 64)), ((64, 64), (32, 32))],
        ids=["unpadded", "padded"],
    )
    def test_multiplane_matches_unbatched_path(self, shape, slm_shape, monkeypatch, subtests):
        reference = self._run_multiplane(shape, slm_shape)

        # Force the legacy path: the per-child routine loop.
        monkeypatch.setattr(
            MultiplaneHologram, "_can_batch_routines", lambda self, mraf: False
        )
        legacy = self._run_multiplane(shape, slm_shape)

        with subtests.test("phase"):
            assert np.allclose(_np(reference.phase), _np(legacy.phase), atol=1e-4)
        with subtests.test("child amp_ff"):
            for i, (a, b) in enumerate(zip(reference.holograms, legacy.holograms)):
                assert np.allclose(_np(a.amp_ff), _np(b.amp_ff), atol=1e-5), f"child {i}"

    def test_batched_routines_decline_for_wgs_and_mraf(self, subtests):
        """
        The batched amplitude replacement collapses every child onto one branch, so it
        must decline whenever the children would not agree on that branch.
        """
        mph = self._run_multiplane((64, 64), (64, 64), iterations=1)
        plain = [{"mraf_enabled": False}] * len(mph)

        with subtests.test("plain GS batches"):
            mph.flags["method"] = "GS"
            assert mph._can_batch_routines(plain)

        with subtests.test("WGS declines"):
            mph.flags["method"] = "WGS-Leonardo"
            assert not mph._can_batch_routines(plain)

        with subtests.test("fixed phase declines"):
            mph.flags["method"] = "GS"
            mph.flags["fixed_phase"] = True
            assert not mph._can_batch_routines(plain)
            mph.flags["fixed_phase"] = False

        with subtests.test("MRAF declines"):
            assert not mph._can_batch_routines([{"mraf_enabled": True}] * len(mph))


class TestResetWeightsIsNotRepeated:
    """
    `MultiplaneHologram.reset()` used to reset every child's weights twice: once via
    `super().reset()` (which calls the multiplane `reset_weights` override, itself a loop
    over all children) and again via each child's own `reset()`. Silent and easy to
    reintroduce, so it is pinned by count.
    """

    @staticmethod
    def _multiplane(shape=(64, 64), n=3):
        amp = np.ones(shape, dtype=np.float32) / np.sqrt(np.prod(shape))
        children = []
        for i in range(n):
            target = np.zeros(shape, dtype=np.float32)
            target[8 * (i + 1), 8 * (i + 1)] = 1.0
            children.append(Hologram(target=target, amp=amp))
        return MultiplaneHologram(children)

    def test_each_child_resets_weights_once(self, subtests):
        mph = self._multiplane()
        counts = {id(h): 0 for h in mph.holograms}
        for h in mph.holograms:
            orig = h.reset_weights
            def counted(_h=h, _orig=orig):
                counts[id(_h)] += 1
                return _orig()
            h.reset_weights = counted

        mph.reset()

        with subtests.test("exactly once per child"):
            assert set(counts.values()) == {1}, counts

        with subtests.test("reset_weights=False skips it entirely"):
            for k in counts:
                counts[k] = 0
            mph.reset(reset_weights=False)
            assert set(counts.values()) == {0}, counts

    def test_reset_weights_still_correct_and_sync_free(self, subtests):
        """
        `cp.nan_to_num` was replaced because its scalar fill values route through
        `_check_nan_inf`, which takes the truth value of a 0-d device array — a host
        synchronization on every call. The replacement must still zero the NaN region.
        """
        shape = (32, 32)
        target = np.ones(shape, dtype=np.float32)
        target[4:8, 4:8] = np.nan
        h = Hologram(target=np.nan_to_num(target, nan=0.0))
        h.target = cp.asarray(target) if cp is not None else target

        h.reset_weights()
        weights = _np(h.weights)

        with subtests.test("NaN region is zeroed"):
            assert not np.any(np.isnan(weights))
            assert np.all(weights[4:8, 4:8] == 0)

        with subtests.test("rest of the target is untouched"):
            expected = np.nan_to_num(target, nan=0.0)
            assert np.allclose(weights, expected)


class TestPropagationKernelStack:

    @staticmethod
    def _multiplane(shape, slm_shape, n=3, amp=None):
        if amp is None:
            amp = np.ones(slm_shape, dtype=np.float32) / np.sqrt(np.prod(slm_shape))
        children = []
        for i in range(n):
            target = np.zeros(shape, dtype=np.float32)
            target[8 * (i + 1), 8 * (i + 1)] = 1.0
            children.append(Hologram(target=target, amp=amp, slm_shape=slm_shape))
        return MultiplaneHologram(children)

    @staticmethod
    def _gaussian_amp(slm_shape):
        """A non-uniform source. A *uniform* amp cannot detect the nearfield build
        dropping it — GS renormalizes, so a constant factor leaves the phase identical."""
        (yy, xx) = np.meshgrid(
            np.linspace(-1, 1, slm_shape[0]), np.linspace(-1, 1, slm_shape[1]), indexing="ij"
        )
        amp = np.exp(-(xx**2 + yy**2) / 0.5).astype(np.float32)
        return amp / np.sqrt(np.sum(amp**2))

    @pytest.mark.parametrize(
        "shape, slm_shape",
        [((64, 64), (64, 64)), ((64, 64), (32, 32))],
        ids=["unpadded", "padded"],
    )
    def test_matches_per_child_assignment(self, shape, slm_shape, subtests):
        """`set_propagation_kernels(stack)` must be indistinguishable from assigning
        each child individually — that is the whole basis for using it."""
        rng = np.random.default_rng(3)
        n = 3
        stack = (rng.random((n,) + slm_shape) * 6.28).astype(np.float32)

        legacy = self._multiplane(shape, slm_shape, n)
        for i, h in enumerate(legacy.holograms):
            h.propagation_kernel = cp.asarray(stack[i]) if cp is not None else stack[i]
        legacy._refresh_batched_kernels()

        batched = self._multiplane(shape, slm_shape, n)
        batched.set_propagation_kernels(stack)

        with subtests.test("phasor matches"):
            assert np.allclose(
                _np(batched._batched_kernel_phasor),
                _np(legacy._batched_kernel_phasor),
                atol=1e-6,
            )

        with subtests.test("children see their own kernel"):
            for i, h in enumerate(batched.holograms):
                assert np.allclose(_np(h.propagation_kernel), stack[i], atol=1e-6)

        with subtests.test("no restack is triggered afterwards"):
            phasor = batched._batched_kernel_phasor
            batched._refresh_batched_kernels()
            assert batched._batched_kernel_phasor is phasor

        with subtests.test("GS agrees with per-child assignment"):
            for mph in (legacy, batched):
                mph.reset_phase(random_phase=0)
                mph.optimize(method="GS", maxiter=5, verbose=False)
            assert np.allclose(_np(legacy.phase), _np(batched.phase), atol=1e-4)

    def test_broadcasts_a_single_kernel(self):
        """A single (slm_h, slm_w) kernel applies to every plane."""
        mph = self._multiplane((64, 64), (64, 64), n=3)
        rng = np.random.default_rng(4)
        one = (rng.random((64, 64)) * 6.28).astype(np.float32)
        mph.set_propagation_kernels(one)
        for h in mph.holograms:
            assert np.allclose(_np(h.propagation_kernel), one, atol=1e-6)

    def test_refresh_never_leaves_the_phasor_none(self, subtests):
        """
        `_refresh_batched_kernels` must guarantee a usable phasor on return -- the next
        transform multiplies by it unconditionally.

        Today the id-based restack happens to cover the all-`None`-kernels case, because
        `id(None)` never equals the `None` the id list is seeded with. That is an accident
        of the sentinel, not an invariant: upstream's content-fingerprint variant returned
        `None` for a `None` kernel and *did* leave the phasor unbuilt. The explicit
        `phasor is None` arm is what makes this hold regardless, so it is pinned directly
        rather than through the scenario the sentinel already handles.
        """
        mph = self._multiplane((64, 64), (64, 64), n=3)

        with subtests.test("all-None kernels still build a unit phasor"):
            assert all(h.propagation_kernel is None for h in mph.holograms)
            mph._refresh_batched_kernels()
            assert mph._batched_kernel_phasor is not None
            assert np.allclose(_np(mph._batched_kernel_phasor), 1.0, atol=1e-6)

        with subtests.test("rebuilds even when the restack reports no change"):
            # ids already match, so _restack_children returns False; only the explicit
            # guard can notice the phasor is missing.
            mph._batched_kernel_phasor = None
            mph._refresh_batched_kernels()
            assert mph._batched_kernel_phasor is not None

        with subtests.test("transform runs and stays finite"):
            mph.optimize(method="GS", maxiter=2, verbose=False)
            assert np.all(np.isfinite(_np(mph.phase)))

    def test_direct_rebinding_is_picked_up(self):
        """
        The cache promises to notice *rebinding*, even when it bypasses
        `set_propagation_kernels`. (In-place mutation is explicitly outside the contract
        -- see `_restack_children`.)
        """
        mph = self._multiplane((64, 64), (64, 64), n=3)
        mph._refresh_batched_kernels()
        before = _np(mph._batched_kernel_phasor).copy()

        rng = np.random.default_rng(9)
        kernel = (rng.random((64, 64)) * 6.28).astype(np.float32)
        mph.holograms[1].propagation_kernel = (
            cp.asarray(kernel) if cp is not None else kernel
        )
        mph._refresh_batched_kernels()

        after = _np(mph._batched_kernel_phasor)
        assert not np.allclose(after, before), "rebinding a child kernel must invalidate"
        assert np.allclose(after[1], np.exp(1j * kernel), atol=1e-5)
        assert np.allclose(after[0], 1.0, atol=1e-6), "untouched planes stay put"

    # "odd-pad" has an odd shape-minus-slm_shape difference, so `unpad` returns a slice
    # starting at 0 that nonetheless does not cover the batched tensor. That is the case
    # where skipping the zero-fill leaves the trailing row/column holding the previous
    # iteration's nearfield.
    @pytest.mark.parametrize(
        "shape, slm_shape",
        [((64, 64), (64, 64)), ((64, 64), (32, 32)), ((65, 65), (64, 64))],
        ids=["unpadded", "padded", "odd-pad"],
    )
    def test_batched_nearfield_matches_unbatched(self, shape, slm_shape, monkeypatch):
        """
        The batched path builds the nearfield with a fused kernel; the unbatched path
        builds it through `Hologram._build_nearfield`. Comparing them with *non-trivial*
        propagation kernels is what actually pins the fused kernel — every other test
        here compares two runs that both go through it, so none of them would notice if
        it dropped a term.
        """
        rng = np.random.default_rng(5)
        n = 3
        stack = (rng.random((n,) + slm_shape) * 6.28).astype(np.float32)
        amp = self._gaussian_amp(slm_shape)

        def run(force_unbatched):
            if force_unbatched:
                monkeypatch.setattr(MultiplaneHologram, "_can_batch", lambda self: False)
            mph = self._multiplane(shape, slm_shape, n, amp=amp)
            mph.set_propagation_kernels(stack)
            mph.reset_phase(random_phase=0)
            mph.optimize(method="GS", maxiter=6, verbose=False)
            return mph

        batched = run(False)
        assert batched._batched, "expected the batched fast path for this configuration"
        unbatched = run(True)
        assert not unbatched._batched

        assert np.allclose(_np(batched.phase), _np(unbatched.phase), atol=1e-4)


class TestPopulateResults:

    def test_amp_ff_matches_farfield(self, subtests):
        """Guards the removal of the duplicate `cp.abs` — `_nearfield2farfield` already
        populated `amp_ff`, so `_populate_results` must leave it consistent."""
        shape = (64, 64)
        target = np.zeros(shape, dtype=np.float32)
        target[16, 16] = 1.0
        h = Hologram(target=target)
        h.reset_phase(random_phase=0)
        h.optimize(method="GS", maxiter=3, verbose=False)

        with subtests.test("amp_ff == abs(farfield)"):
            assert np.allclose(_np(h.amp_ff), np.abs(_np(h.farfield)), atol=1e-6)

        with subtests.test("phase_ff == angle(farfield)"):
            assert np.allclose(_np(h.phase_ff), np.angle(_np(h.farfield)), atol=1e-5)

        with subtests.test("multiplane meta amp_ff is still populated"):
            amp = np.ones(shape, dtype=np.float32) / np.sqrt(np.prod(shape))
            t2 = np.zeros(shape, dtype=np.float32)
            t2[32, 32] = 1.0
            mph = MultiplaneHologram(
                [Hologram(target=target, amp=amp), Hologram(target=t2, amp=amp)]
            )
            mph.optimize(method="GS", maxiter=2, verbose=False)
            assert mph.amp_ff is not None


class TestTransparentCaches:
    """
    The resampler cache memoizes per-geometry device arrays sized like a hologram, so it
    needs a bound and a way to release what it holds -- matching
    `clear_zernike_basis_cache()`, the established precedent.
    """

    def test_resampler_cache_is_bounded_and_clearable(self, subtests):
        from slmsuite.holography.algorithms._feedback import (
            _IJCAM_TO_KNMSLM_CACHE, clear_ijcam_to_knmslm_cache
        )
        clear_ijcam_to_knmslm_cache()

        cam_shape = (60, 80)
        canvas = np.zeros(cam_shape, dtype=np.float32)
        canvas[24:44, 34:54] = 1.0

        with subtests.test("repeated calls reuse one entry"):
            h = FeedbackHologram(
                (32, 32), target_ij=canvas, cameraslm=_StubCameraSLM(cam_shape)
            )
            n = len(_IJCAM_TO_KNMSLM_CACHE)
            for _ in range(3):
                h.update_target(canvas)
            assert len(_IJCAM_TO_KNMSLM_CACHE) == n

        with subtests.test("evicts past capacity"):
            for size in range(8, 8 + 2 * (_IJCAM_TO_KNMSLM_CACHE.maxsize + 4), 2):
                FeedbackHologram(
                    (size, size), target_ij=canvas, cameraslm=_StubCameraSLM(cam_shape)
                )
            assert len(_IJCAM_TO_KNMSLM_CACHE) <= _IJCAM_TO_KNMSLM_CACHE.maxsize

        with subtests.test("clear releases everything"):
            clear_ijcam_to_knmslm_cache()
            assert len(_IJCAM_TO_KNMSLM_CACHE) == 0


class TestResetBuffers:

    def test_reset_reuses_matching_buffers(self, subtests):
        """
        reset() used to reallocate nearfield/farfield unconditionally, churning
        megabytes for callers that reset between optimizations.
        """
        h = Hologram(target=np.zeros((64, 64), dtype=np.float32))
        nearfield, farfield = h.nearfield, h.farfield

        with subtests.test("buffers are reused"):
            h.reset()
            assert h.nearfield is nearfield
            assert h.farfield is farfield

        with subtests.test("buffers are zeroed"):
            h.nearfield.fill(1)
            h.reset()
            assert not np.any(_np(h.nearfield))

        with subtests.test("a shape change still reallocates"):
            h.target = cp.zeros((32, 32), dtype=h.dtype) if cp is not None \
                else np.zeros((32, 32), dtype=h.dtype)
            h.reset()
            assert h.farfield is not farfield
            assert h.farfield.shape == (32, 32)


# ---------------------------------------------------------------------------------------
# Camera-feedback ROI


class _StubSLM:
    """Minimal duck-typed SLM: just the geometry `ijcam_to_knmslm` reads."""

    def __init__(self, shape=(32, 32), pitch=(1e-5, 1e-5)):
        self.shape = shape
        self.pitch = np.array(pitch, dtype=float)

    def set_phase(self, phase=None, **kwargs):
        """Never called; `toolbox.convert_vector` duck-types an SLM by this method."""

    def _get_source_amplitude(self):
        amp = np.ones(self.shape, dtype=np.float32)
        return amp / np.sqrt(np.sum(amp**2))


class _StubCameraSLM:
    """
    Minimal duck-typed FourierSLM carrying a synthetic Fourier calibration. Enough for
    `FeedbackHologram` to build targets; nothing here talks to hardware.
    """

    def __init__(self, cam_shape=(60, 80), slm=None, calibrated=True):
        self.slm = slm if slm is not None else _StubSLM()
        self.cam = SimpleNamespace(shape=cam_shape)
        self._calibrated = calibrated
        # kxy -> ij. With this SLM pitch and hologram shape one knm pixel is 3125 kxy,
        # so these scales put the knm grid at 0.6 and 0.5 camera pixels per knm pixel --
        # anisotropic, and compact enough that the whole grid lands inside the frame with
        # room for an ROI around it. The off-centre origin keeps the composite affine
        # from being accidentally trivial.
        self.calibrations = {"fourier": {}} if calibrated else {}
        self.fourier_affine = Affine(
            M=[[1.92e-4, 0.0], [0.0, 1.60e-4]],
            b=[[cam_shape[1] / 2 + 3], [cam_shape[0] / 2 - 4]],
        )

    def ijcam_to_kxyslm(self, ij):
        M_inv = np.linalg.inv(self.fourier_affine.M)
        return np.matmul(M_inv, np.array(ij, dtype=float) - self.fourier_affine.b)


class TestTargetROI:
    """
    A sub-image target must be indistinguishable from the same sub-image pasted into a
    full camera-sized NaN canvas -- that equivalence is the entire justification for the
    ROI path, which exists only to avoid uploading the canvas.

    The ROI here is sized with margin around the knm grid's camera footprint, matching
    the documented usage. Bounds are tested against the sub-image, so an ROI cropped
    tight to the footprint clips its outer half-pixel rim, where a canvas would have
    rounded inward; `test_tight_roi_clips_only_the_border` pins that boundary.
    """

    SHAPE = (32, 32)
    CAM_SHAPE = (60, 80)
    # The knm grid maps to roughly 19 x 16 camera pixels about (26, 43); this ROI covers
    # that footprint with several pixels to spare on every side.
    ROI = (12, 29)
    PATCH_SHAPE = (28, 28)

    def _canvas_and_patch(self, roi=None, patch_shape=None, seed=7):
        roi = self.ROI if roi is None else roi
        patch_shape = self.PATCH_SHAPE if patch_shape is None else patch_shape
        rng = np.random.default_rng(seed)
        patch = rng.random(patch_shape).astype(np.float32)
        canvas = np.full(self.CAM_SHAPE, np.nan, dtype=np.float32)
        (y0, x0) = roi
        canvas[y0:y0 + patch_shape[0], x0:x0 + patch_shape[1]] = patch
        return canvas, patch

    def test_roi_target_matches_full_canvas(self, subtests):
        canvas, patch = self._canvas_and_patch()

        full = FeedbackHologram(
            self.SHAPE, target_ij=canvas, cameraslm=_StubCameraSLM(self.CAM_SHAPE)
        )
        sub = FeedbackHologram(
            self.SHAPE,
            target_ij=patch,
            target_ij_roi=self.ROI,
            cameraslm=_StubCameraSLM(self.CAM_SHAPE),
        )

        with subtests.test("the target is actually populated"):
            assert np.count_nonzero(np.nan_to_num(_np(full.target))) > 100

        with subtests.test("knm target agrees"):
            assert np.allclose(_np(full.target), _np(sub.target), equal_nan=True)

        with subtests.test("roi is recorded"):
            assert sub.target_ij_roi == self.ROI
            assert full.target_ij_roi is None

    def test_tight_roi_clips_only_the_border(self):
        """
        An ROI cropped tight to the signal loses at most the outer rim relative to the
        canvas, and never disagrees on an interior sample. This is the documented
        caveat, pinned so it stays a rim effect rather than growing into the interior.
        """
        roi = (24, 34)
        canvas, patch = self._canvas_and_patch(roi=roi, patch_shape=(12, 12))

        full = FeedbackHologram(
            self.SHAPE, target_ij=canvas, cameraslm=_StubCameraSLM(self.CAM_SHAPE)
        )
        sub = FeedbackHologram(
            self.SHAPE, target_ij=patch, target_ij_roi=roi,
            cameraslm=_StubCameraSLM(self.CAM_SHAPE),
        )

        (a, b) = (_np(full.target), _np(sub.target))
        defined_in_both = (a != 0) & (b != 0)
        assert np.count_nonzero(defined_in_both) > 50, "nothing left to compare"

        # `ijcam_to_knmslm` normalizes, and the two have different support, so they agree
        # up to one global scale. A *consistent* ratio is the real claim: the interior
        # samples carry identical relative weight.
        ratio = b[defined_in_both] / a[defined_in_both]
        assert np.allclose(ratio, np.median(ratio), rtol=1e-5)

        # The ROI may only ever drop samples, never invent them.
        assert np.all((b != 0) <= (a != 0))

    def test_update_target_agrees_with_construction(self):
        """`update_target(roi=...)` is the hot path; it must land where the constructor does."""
        canvas, patch = self._canvas_and_patch()

        full = FeedbackHologram(
            self.SHAPE, target_ij=canvas, cameraslm=_StubCameraSLM(self.CAM_SHAPE)
        )
        sub = FeedbackHologram(
            self.SHAPE, target_ij=patch, target_ij_roi=self.ROI,
            cameraslm=_StubCameraSLM(self.CAM_SHAPE),
        )

        (canvas2, patch2) = self._canvas_and_patch(seed=11)
        full.update_target(canvas2, reset_weights=True)
        sub.update_target(patch2, reset_weights=True, roi=self.ROI)

        assert np.allclose(_np(full.target), _np(sub.target), equal_nan=True)
        assert np.allclose(_np(full.weights), _np(sub.weights), equal_nan=True)

    @pytest.mark.parametrize("roi", [(-1, 0), (0, -1), (52, 34), (24, 72)])
    def test_out_of_frame_roi_is_rejected(self, roi):
        """
        A sub-image hanging off the camera has samples no frame could supply, and would
        silently mis-slice img_ij when computing experimental stats.
        """
        _, patch = self._canvas_and_patch((0, 0))
        with pytest.raises(ValueError):
            FeedbackHologram(
                self.SHAPE, target_ij=patch, target_ij_roi=roi,
                cameraslm=_StubCameraSLM(self.CAM_SHAPE),
            )

    @pytest.mark.parametrize("roi", [(-1, 0), (0, -1), (52, 34), (24, 72)])
    def test_out_of_frame_roi_is_rejected_on_every_entry_point(self, roi):
        """
        `affine_transform` tests bounds against the sub-image alone, with no camera-frame
        intersection, so an ROI hanging off the frame silently resamples samples no camera
        could supply. The check has to live at the `_ijcam_to_knmslm_resampler` choke
        point, not only in `update_target`: `ijcam_to_knmslm` reaches it directly.
        """
        canvas, patch = self._canvas_and_patch()
        h = FeedbackHologram(
            self.SHAPE, target_ij=canvas, cameraslm=_StubCameraSLM(self.CAM_SHAPE)
        )

        with pytest.raises(ValueError, match="camera frame"):
            h.ijcam_to_knmslm(patch, order=0, roi=roi)

        with pytest.raises(ValueError, match="camera frame"):
            h.update_target(patch, roi=roi)

    def test_roi_without_a_calibration_is_rejected(self):
        """
        Without a Fourier calibration the constructor's update_target() never runs, so a
        roi would be dropped while target_ij stays a sub-image -- leaving the two
        silently inconsistent.
        """
        _, patch = self._canvas_and_patch()
        with pytest.raises(ValueError, match="Fourier calibration"):
            FeedbackHologram(
                self.SHAPE, target_ij=patch, target_ij_roi=self.ROI,
                cameraslm=_StubCameraSLM(self.CAM_SHAPE, calibrated=False),
            )

    def test_measure_knm_windows_by_the_roi(self, subtests):
        """
        `measure("knm")` must land on the same knm grid whether the hologram was given a
        full-frame target or a sub-image, otherwise `img_knm` and `target` disagree
        pixel-for-pixel and every feedback statistic is computed against a shifted image.
        The ROI path exists only to skip uploading the rest of the frame.

        Agreement is *exact* at order=0, the order `update_target` uses. At the cubic
        default that `measure` uses it is approximate: spline prefiltering is an IIR
        recursion over the whole input array, so cropping the input perturbs the
        coefficients everywhere, not just at the edge, and `measure`'s square root then
        amplifies that in the dimmest pixels. The bound below pins it as a rounding-level
        effect rather than a misalignment -- a misalignment would be O(1), not O(1e-3) --
        and would catch it growing.
        """
        canvas, patch = self._canvas_and_patch()
        frame = np.nan_to_num(canvas, nan=0.0).astype(np.float32)

        full = FeedbackHologram(
            self.SHAPE, target_ij=canvas, cameraslm=_StubCameraSLM(self.CAM_SHAPE)
        )
        sub = FeedbackHologram(
            self.SHAPE, target_ij=patch, target_ij_roi=self.ROI,
            cameraslm=_StubCameraSLM(self.CAM_SHAPE),
        )

        # Computational feedback leaves `should_update` False, so `_update_slm` is a
        # no-op and no camera is needed -- `measure` takes its "img_ij already present"
        # branch. `optimize()` would normally populate these flags.
        for h in (full, sub):
            h.flags.update({"feedback": "computational", "stat_groups": []})
            h.img_ij, h.img_knm = frame, None
            h.measure("knm")

        with subtests.test("img_knm agrees to prefilter tolerance"):
            # 2e-3 against a peak amplitude of ~0.29, i.e. under 1% of peak.
            assert np.allclose(
                _np(full.img_knm), _np(sub.img_knm), atol=2e-3, rtol=0, equal_nan=True
            )

        with subtests.test("img_knm agrees exactly at order=0"):
            windows = [
                h.ijcam_to_knmslm(
                    np.square(h._roi_window(frame)), roi=h.target_ij_roi, order=0
                )
                for h in (full, sub)
            ]
            assert np.array_equal(_np(windows[0]), _np(windows[1]), equal_nan=True)

        with subtests.test("the measurement is actually populated"):
            assert np.count_nonzero(np.nan_to_num(_np(sub.img_knm))) > 100

        with subtests.test("only the window is transformed"):
            assert _np(sub._roi_window(frame)).shape == self.PATCH_SHAPE
            assert _np(full._roi_window(frame)).shape == self.CAM_SHAPE

    @pytest.mark.parametrize("order", [0, 3])
    def test_augmented_matrix_matches_separate_offset(self, order):
        """
        `ijcam_to_knmslm` hands `affine_transform` an augmented (2, 3) matrix instead of a
        separate `offset=`, purely to dodge the per-element `float()` conversion cupy
        applies to a device offset. Pin the two forms as equivalent -- including the NaN
        fill that MRAF's "undefined region" depends on -- so a cupy change that quietly
        reinterprets the augmented form cannot pass unnoticed.
        """
        canvas, _ = self._canvas_and_patch((24, 34), patch_shape=(20, 20))
        canvas = np.nan_to_num(canvas, nan=0.0)     # affine_transform cannot blur NaNs.
        h = FeedbackHologram(
            self.SHAPE, target_ij=canvas, cameraslm=_StubCameraSLM(self.CAM_SHAPE)
        )

        augmented = h.ijcam_to_knmslm(canvas, order=order)

        (matrix, _fresh) = h._ijcam_to_knmslm_resampler(np.shape(canvas), None)
        assert matrix.shape == (2, 3)
        reference = cp_affine_transform(
            input=cp.array(canvas, dtype=h.dtype),
            matrix=matrix[:, :-1].copy(),
            offset=matrix[:, -1].copy(),
            output_shape=h.shape,
            order=order,
            mode="constant",
            cval=np.nan,
        )
        # Mirror the abs + normalize that `ijcam_to_knmslm` applies after the transform.
        reference = cp.abs(reference)
        reference /= Hologram._norm(reference)

        assert np.allclose(_np(augmented), _np(reference), equal_nan=True)
