"""
Unit tests for slmsuite.holography.algorithms module.
"""
import pytest
import numpy as np
import logging

from slmsuite.holography.algorithms import Hologram, SpotHologram, MultiplaneHologram

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
