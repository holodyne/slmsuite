"""
Unit tests for the SLM base class and its subclasses.
"""
import os
import tempfile
import warnings

import pytest
import numpy as np
import matplotlib.pyplot as plt

from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.hardware.slms.segmented import SegmentedSLM
from slmsuite.hardware.slms.screenmirrored import ScreenMirrored


def _quadratic_gamma(bitresolution):
    """A monotonic but distinctly non-uniform phase response."""
    levels = np.arange(bitresolution)
    return np.square(levels / (bitresolution - 1)) * (bitresolution - 1) / bitresolution


class TestSLM:
    """Tests for the SLM base class (via SimulatedSLM)."""

    @staticmethod
    def _slm(bitdepth=8, gpu=False):
        return SimulatedSLM(resolution=(32, 32), bitdepth=bitdepth, gpu=gpu)

    def test_selftest(self, slm):
        """SLM.test() exercises most methods; it must return True."""
        assert slm.test() is True

    def test_init(self, slm, subtests):
        """Validate constructor-derived attributes and conventions."""
        with subtests.test("default fixture attributes"):
            assert slm.shape == (1080, 1920)
            assert slm.bitdepth == 8
            assert slm.bitresolution == 256
            assert slm.wav_um == 0.78
            assert np.allclose(slm.pitch_um, [8.0, 8.0])

        with subtests.test("resolution (w,h) -> shape (h,w)"):
            h, w = 600, 800
            s = SimulatedSLM(resolution=(w, h))
            assert s.shape == (h, w)
            s.close()

        with subtests.test("grid shape matches SLM shape"):
            assert len(slm.grid) == 2
            for g in slm.grid:
                assert g.shape == slm.shape

        with subtests.test("custom wav_design_um"):
            s = SimulatedSLM(resolution=(128, 128), wav_um=0.78, wav_design_um=1.064)
            assert s.wav_design_um == 1.064
            assert s.phase_scaling == pytest.approx(0.78 / 1.064)
            s.close()

        with subtests.test("scalar pitch_um broadcasts"):
            s = SimulatedSLM(resolution=(128, 128), pitch_um=10)
            assert np.allclose(s.pitch_um, [10.0, 10.0])
            s.close()

        with subtests.test("invalid pitch_um raises"):
            with pytest.raises(ValueError):
                SimulatedSLM(resolution=(128, 128), pitch_um=(0, 8))

        with subtests.test("16-bit dtype for large bitdepth"):
            s = SimulatedSLM(resolution=(128, 128), bitdepth=10)
            assert s.dtype == np.dtype(np.uint16)
            assert s.bitresolution == 1024
            s.close()

    def test_backend(self, monkeypatch, subtests):
        """gpu selects the array backend; without cupy, numpy stands in for it."""
        s = self._slm()

        with subtests.test("gpu=False is numpy"):
            assert s.xp is np
            assert isinstance(s.phase, np.ndarray)
            assert isinstance(s.display, np.ndarray)

        with subtests.test("lut defaults to None"):
            assert s.lut is None

        s.close()

        import slmsuite.hardware.slms.slm as module
        monkeypatch.setattr(module, "cp", np)
        s = self._slm()
        phase = np.linspace(0, 2 * np.pi, s.shape[0] * s.shape[1]).reshape(s.shape)

        with subtests.test("without cupy the numpy path still runs"):
            assert np.all(s.set_phase(phase, phase_correct=False) < s.bitresolution)
            s.plot()
            plt.close("all")

        with subtests.test("without cupy, gpu=True is refused"):
            with pytest.raises(ImportError):
                self._slm(gpu=True)

        s.close()

    def test_phase2gray(self, slm, subtests, benchmark):
        """Edge cases for _phase2gray not covered by .test()."""
        with subtests.test("benchmark"):
            phase = np.random.uniform(0, 2 * np.pi, slm.shape).astype(np.float32)
            benchmark(slm._phase2gray, phase)

        with subtests.test("negative phase wraps to valid gray"):
            phase = -np.ones(slm.shape) * np.pi
            gray = slm._phase2gray(phase)
            assert np.all(gray >= 0) and np.all(gray < slm.bitresolution)

        with subtests.test("large phase gives the same gray as its wrap"):
            for p in (-1e4, 10 * np.pi, 1e4):
                np.testing.assert_array_equal(
                    slm._phase2gray(np.full(slm.shape, p)),
                    slm._phase2gray(np.full(slm.shape, np.mod(p, 2 * np.pi))),
                )

        with subtests.test("zero phase -> display max (sign convention)"):
            gray = slm._phase2gray(np.zeros(slm.shape))
            assert np.all(gray == slm.bitresolution - 1)

        with subtests.test("non-standard bitdepth uses bitwise_and mask"):
            s = SimulatedSLM(resolution=(64, 64), bitdepth=5)
            phase = np.linspace(0, 4 * np.pi, 64 * 64).reshape(s.shape)
            gray = s._phase2gray(phase)
            assert np.all(gray >= 0) and np.all(gray < s.bitresolution)
            s.close()

        s = self._slm()
        phase = np.linspace(-4 * np.pi, 4 * np.pi, s.shape[0] * s.shape[1]).reshape(s.shape)
        mirror = s._phase2gray(-phase.copy())
        s._gamma_sign = +1

        with subtests.test("the opposite sign convention mirrors the mapping"):
            np.testing.assert_array_equal(s._phase2gray(phase.copy()), mirror)

        with subtests.test("that sign is only implemented with a lookup table"):
            s.wav_design_um = 2 * s.wav_um
            with pytest.raises(NotImplementedError):
                s._phase2gray(phase.copy())

        s.close()

    def test_set_phase(self, slm, subtests, benchmark):
        """set_phase edge cases beyond what .test() exercises."""
        with subtests.test("benchmark"):
            phase = np.random.uniform(0, 2 * np.pi, slm.shape).astype(np.float32)
            benchmark(slm.set_phase, phase, phase_correct=False)

        with subtests.test("None zeros phase and display"):
            slm.set_phase(None, phase_correct=False)
            assert np.all(slm.phase == 0)

        with subtests.test("wrong integer type raises TypeError"):
            wrong_dtype = np.uint16 if slm.dtype == np.uint8 else np.uint8
            bad = np.zeros(slm.shape, dtype=wrong_dtype)
            with pytest.raises(TypeError):
                slm.set_phase(bad)

        with subtests.test("oversize integer is unpadded"):
            big = np.zeros((slm.shape[0] + 20, slm.shape[1] + 20), dtype=slm.dtype)
            big[:] = slm.bitresolution // 2
            slm.set_phase(big)
            assert slm.display.shape == slm.shape
            assert np.all(slm.display == slm.bitresolution // 2)

        with subtests.test("phase_correct adds source phase"):
            slm.source["phase"] = np.ones(slm.shape) * 0.1
            slm.set_phase(np.zeros(slm.shape), phase_correct=True)
            np.testing.assert_allclose(slm.phase, 0.1, atol=0.01)
            del slm.source["phase"]

        with subtests.test("write() deprecation alias"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                slm.write(np.zeros(slm.shape), phase_correct=False)
                assert any("depreciated" in str(x.message).lower() for x in w)

        with subtests.test("test integer passthrough"):
            int_data = np.full(slm.shape, slm.bitresolution // 2, dtype=slm.dtype)
            slm.set_phase(int_data, phase_correct=False)
            np.testing.assert_array_equal(slm.display, int_data)

        with subtests.test("test integer overflow"):
            int_data = np.full(slm.shape, 2 * slm.bitresolution, dtype=np.int64)
            with pytest.raises(TypeError, match="Unexpected integer type"):
                slm.set_phase(int_data, phase_correct=False)

        with subtests.test("display in valid range after random phase"):
            phase = np.random.uniform(-4 * np.pi, 4 * np.pi, slm.shape).astype(np.float32)
            slm.set_phase(phase, phase_correct=False)
            assert np.all(slm.display < slm.bitresolution)

        with subtests.test("set_phase returns display"):
            result = slm.set_phase(np.zeros(slm.shape), phase_correct=False)
            assert result is slm.display

    def test_set_gamma(self, subtests):
        """The table inverts a measured response; clearing it restores the linear path."""
        s = self._slm(bitdepth=6)
        B = s.bitresolution
        levels = np.arange(B)
        ideal = levels / B
        N = s.shape[0] * s.shape[1]

        # Sample a quarter of a level off each level, spanning many cycles of both signs.
        phase = ((np.arange(N) - N // 2 + 0.25) * (2 * np.pi / B)).reshape(s.shape)
        linear = s.set_phase(phase.copy(), phase_correct=False).copy()

        with subtests.test("an ideal gamma matches the linear path"):
            s.set_gamma(ideal)
            assert s.lut.shape == (1 << 16,)
            assert s.lut.dtype == s.dtype
            lut = s.set_phase(phase.copy(), phase_correct=False).copy()
            # The linear path shifts by one level so that zero phase is the max level.
            assert np.array_equal((lut.astype(int) - linear.astype(int)) % B, np.ones(s.shape))

        with subtests.test("every level is recovered from the phase it realizes"):
            swapped = ideal.copy()
            swapped[[3, 4]] = swapped[[4, 3]]
            for gamma in (_quadratic_gamma(B), swapped):
                s.set_gamma(gamma)
                realized = np.mod(s._gamma_sign * gamma * 2 * np.pi, 2 * np.pi)
                assert np.array_equal(s.lut[s._phase2lut(realized)], levels.astype(s.dtype))

        with subtests.test("a response spanning more or less than a cycle uses every level"):
            for gamma in (2.5 * ideal, 0.5 * ideal):
                s.set_gamma(gamma)
                assert len(np.unique(s.lut)) == B

        with subtests.test("a response short of a cycle piles up at the nearer endpoint"):
            s.set_gamma(0.5 * ideal)
            counts = np.bincount(np.asarray(s.lut).ravel(), minlength=B)
            assert np.argmax(counts) in (0, B - 1)

        with subtests.test("phase far outside one cycle wraps"):
            s.set_gamma(ideal)
            for p in (2.1e5, 1e6, 1e9):
                np.testing.assert_array_equal(
                    s._phase2gray(np.full(s.shape, p)),
                    s._phase2gray(np.full(s.shape, np.mod(p, 2 * np.pi))),
                )

        with subtests.test("clearing restores the linear path"):
            s.set_gamma(ideal)
            assert s.set_gamma(None) is None
            assert s.gamma is None and s.lut is None
            assert np.array_equal(s.set_phase(phase.copy(), phase_correct=False), linear)

        with subtests.test("derived data is not pickled"):
            # The pixel calibration is what persists; the table is rebuilt from it.
            s.set_gamma(ideal)
            data = s.pickle(attributes=True, metadata=False)
            assert "gamma" not in data and "lut" not in data

        with subtests.test("lut_size is honored"):
            s.set_gamma(ideal, lut_size=1 << 10)
            assert s.lut.size == 1 << 10
            assert s._phase_to_lut == pytest.approx((1 << 10) / (2 * np.pi))

        with subtests.test("a bitdepth past eight tabulates 16-bit levels"):
            wide = self._slm(bitdepth=10)
            wide_levels = np.arange(wide.bitresolution)
            wide.set_gamma(wide_levels / wide.bitresolution)
            assert wide.lut.dtype == np.dtype(np.uint16)
            realized = np.mod(
                wide._gamma_sign * (wide_levels / wide.bitresolution) * 2 * np.pi, 2 * np.pi
            )
            assert np.array_equal(
                wide.lut[wide._phase2lut(realized)], wide_levels.astype(wide.dtype)
            )
            wide.close()

        with subtests.test("invalid input raises"):
            with pytest.raises(ValueError):
                s.set_gamma(ideal, lut_size=1000)
            for bad in (ideal[:-1], [0.5]):
                with pytest.raises(ValueError, match="span all"):
                    s.set_gamma(bad)
            degenerate = ideal.copy()
            degenerate[3] = np.nan
            with pytest.raises(ValueError):
                s.set_gamma(degenerate)

        s.close()

    def test_interpolate_gamma(self, subtests):
        """Spreading a sampled response across every level."""
        s = self._slm()
        B = s.bitresolution
        ideal = np.arange(B) / B

        levels = np.arange(32) * (B / 32)

        with subtests.test("a subsampled linear response is recovered at every level"):
            np.testing.assert_allclose(s.interpolate_gamma(levels / B, levels), ideal)

            # Sampling that starts above zero must close at the bottom as well as the top.
            edged = np.arange(16, B, 16)
            np.testing.assert_allclose(s.interpolate_gamma(edged / B, edged), ideal)

        with subtests.test("sample order does not matter"):
            # A linear response interpolates alike between any pair, so it cannot see the sort.
            curved = _quadratic_gamma(B)[levels.astype(int)]
            order = np.random.permutation(len(levels))
            np.testing.assert_allclose(
                s.interpolate_gamma(curved[order], levels[order]),
                s.interpolate_gamma(curved, levels),
            )

        with subtests.test("non-uniform sampling follows the levels given"):
            sparse = np.array([0, 3, 9, 40, 150, 251])
            np.testing.assert_allclose(
                s.interpolate_gamma(sparse / B, sparse)[sparse], sparse / B
            )

        with subtests.test("degenerate input raises"):
            with pytest.raises(ValueError, match="pair with gamma"):
                s.interpolate_gamma([0, 0.5], [0, 1, 2])
            with pytest.raises(ValueError, match="two distinct levels"):
                s.interpolate_gamma([0, 0.5], [3, 3])
            # Levels spanning the whole range would close onto themselves.
            with pytest.raises(ValueError, match="span less than"):
                s.interpolate_gamma([0, 0.3, 1.0], [0, 100, 400])

        s.close()

    def test_gamma_integer_write(self, subtests):
        """An integer write records the phase that level actually realizes."""
        s = self._slm(bitdepth=6)
        B = s.bitresolution
        gamma = _quadratic_gamma(B)
        s.set_gamma(gamma)

        levels = (np.arange(s.shape[0] * s.shape[1]) % B).astype(s.dtype).reshape(s.shape)
        s.set_phase(levels, phase_correct=False)

        with subtests.test("phase follows gamma, not the linear inverse"):
            expected = np.mod(s._gamma_sign * 2 * np.pi * gamma[levels], 2 * np.pi)
            np.testing.assert_allclose(s.phase, expected)

        with subtests.test("phase re-quantizes to the levels written"):
            np.testing.assert_array_equal(s._phase2gray(s.phase.copy()), levels)

        s.close()

    @pytest.mark.gpu
    def test_gamma_backend_parity(self, subtests):
        """A GPU-backed SLM must display exactly what a CPU-backed one does."""
        from slmsuite.holography.toolbox.phase import blaze

        (cpu, gpu) = (self._slm(), self._slm(gpu=True))
        B = cpu.bitresolution

        with subtests.test("linear path"):
            phase = blaze(cpu, vector=(.05, .03))
            got = gpu.set_phase(phase.copy(), phase_correct=False)
            want = cpu.set_phase(phase.copy(), phase_correct=False)
            np.testing.assert_array_equal(np.asarray(got.get()), want)

        with subtests.test("lut path"):
            gamma = _quadratic_gamma(B)
            cpu.set_gamma(gamma)
            gpu.set_gamma(gamma)
            phase = blaze(cpu, vector=(.05, .03))
            got = gpu.set_phase(phase.copy(), phase_correct=False)
            want = cpu.set_phase(phase.copy(), phase_correct=False)
            np.testing.assert_array_equal(np.asarray(got.get()), want)

        (cpu.close(), gpu.close())

    def test_gamma_matches_plm_quantize_lut(self, subtests):
        """set_gamma generalizes PLM._init_quantize_lut, which it must reproduce exactly."""
        from slmsuite.hardware.slms.texasinstruments import PLM

        for model in PLM.get_model_list():
            config = PLM.load_model_config(model)

            # PLM is not instantiable without a display, so borrow its methods.
            s = self._slm(bitdepth=4)
            s._gamma_sign = +1
            s.model_config = config
            PLM._init_quantize_lut(s)

            B = s.bitresolution
            reference = s._quantize_lut.copy()
            s.set_gamma(np.array(config["displacement_ratios"]) * (B - 1) / B)

            with subtests.test(model):
                assert np.array_equal(reference, s.lut)

            s.close()

        with subtests.test("a PLM has no linear response to clear back to"):
            s = self._slm(bitdepth=4)
            with pytest.raises(ValueError, match="requires a lookup table"):
                PLM.set_gamma(s, None)
            s.close()

    def test_save_load_phase(self, slm, subtests):
        """Round-trip save/load of phase data."""
        with subtests.test("save then load restores display"):
            slm.set_phase(np.random.rand(*slm.shape) * 2 * np.pi, phase_correct=False)
            saved_display = slm.display.copy()
            with tempfile.TemporaryDirectory() as d:
                path = slm.save_phase(path=d, name="test")
                assert os.path.exists(path)
                slm.set_phase(None, phase_correct=False)
                slm.load_phase(path)
                np.testing.assert_array_equal(slm.display, saved_display)

        with subtests.test("load_phase with no file raises FileNotFoundError"):
            orig_cwd = os.getcwd()
            with tempfile.TemporaryDirectory() as d:
                try:
                    os.chdir(d)
                    with pytest.raises(FileNotFoundError):
                        slm.load_phase(None)
                finally:
                    os.chdir(orig_cwd)

    def test_set_source_analytic(self, slm, subtests):
        """set_source_analytic with various unit systems."""
        for units in ["norm", "frac", "um", "mm"]:
            with subtests.test(f"units={units}"):
                src = slm.set_source_analytic(units=units)
                assert "amplitude" in src and src["amplitude"].shape == slm.shape

        with subtests.test("bad units raises RuntimeError"):
            with pytest.raises(RuntimeError, match="Did not recognize"):
                slm.set_source_analytic(units="bad_unit")

        with subtests.test("sim=True stores sim keys"):
            src = slm.set_source_analytic(sim=True)
            assert "amplitude_sim" in src and "phase_sim" in src

        with subtests.test("custom fit_function lambda"):
            src = slm.set_source_analytic(
                fit_function=lambda xy, a=1: a * np.ones_like(xy[0]),
            )
            np.testing.assert_allclose(src["amplitude"], 1.0)

    def test_fit_aperture(self, slm, subtests):
        """fit_aperture with and without measured amplitude, and idempotency."""
        from slmsuite.holography.toolbox import Aperture

        with subtests.test("no amplitude -> guesses from grid"):
            slm.source.pop("amplitude", None)
            slm.set_aperture("cropped")
            ap = slm.fit_aperture()
            assert isinstance(ap, Aperture)
            assert slm.source_radius > 0

        with subtests.test("with amplitude -> moments method"):
            slm.set_source_analytic()
            slm.fit_aperture(method="moments")
            assert slm.source_radius > 0
            assert slm.aperture.center is not None

        with subtests.test("idempotent (no cumulative grid drift)"):
            slm.fit_aperture()
            g1 = [g.copy() for g in slm.grid]
            slm.fit_aperture()
            g2 = slm.grid
            assert all(np.allclose(a, b) for a, b in zip(g1, g2))

        with subtests.test("bad method raises"):
            with pytest.raises(ValueError, match="method"):
                slm.fit_aperture(method="bogus")

    def test_aperture(self, slm, subtests):
        """set_aperture, masking, and the zernike_sum unification invariant."""
        from slmsuite.holography.toolbox.phase import zernike_sum

        with subtests.test("default cropped masks nothing"):
            slm.set_aperture("cropped")
            assert np.all(slm.aperture_mask)

        with subtests.test("circular radius produces a sub-aperture"):
            slm.set_aperture(radius=0.3, units="frac")
            m = slm.aperture_mask
            assert 0 < m.mean() < 1

        with subtests.test("unification: aperture_mask == Aperture.mask"):
            from slmsuite.holography.toolbox import Aperture
            assert np.array_equal(
                slm.aperture_mask, Aperture.resolve(slm).mask
            )

        with subtests.test("aperture masks the source amplitude"):
            slm.source.pop("amplitude", None)
            assert np.array_equal(slm._get_source_amplitude() > 0, slm.aperture_mask)

        with subtests.test("spec and radius mutually exclusive"):
            with pytest.raises(ValueError):
                slm.set_aperture("circular", radius=0.3)

        with subtests.test("source_radius rejects an anisotropic aperture"):
            # A single radius cannot describe an elliptical aperture; fail loudly
            # rather than silently averaging the two axis scales.
            slm.set_aperture((0.01, 0.02))
            with pytest.raises(ValueError, match="isotropic"):
                _ = slm.source_radius
            slm.set_aperture(radius=0.3, units="frac")
            assert slm.source_radius > 0

        with subtests.test("unification holds for a centered aperture"):
            # slm.grid is already shifted, so a resolved aperture must not re-subtract.
            from slmsuite.holography.toolbox import Aperture
            cx, cy = slm.shape[1] / 2 + 120, slm.shape[0] / 2 - 80
            slm.set_aperture(radius=0.3, units="frac", center=(cx, cy))
            assert slm.aperture.center is not None
            assert Aperture.resolve(slm) is slm.aperture
            assert np.array_equal(
                np.asarray(slm.aperture_mask),
                np.asarray(Aperture.resolve(slm).mask),
            )

        with subtests.test("zernike_sum masks a centered aperture consistently"):
            # The actual pipeline (not just resolve): with use_mask=np.nan the region
            # outside the aperture is NaN; it must match the SLM's aperture mask.
            cx, cy = slm.shape[1] / 2 + 120, slm.shape[0] / 2 - 80
            slm.set_aperture(radius=0.3, units="frac", center=(cx, cy))
            result = zernike_sum(slm, [4], [1.0], use_mask=np.nan)
            outside = np.isnan(np.asarray(result))
            assert np.array_equal(outside, ~np.asarray(slm.aperture_mask))

    def test_source_helpers(self, slm, subtests):
        """_get_source_amplitude/phase fallbacks when source is empty."""
        with subtests.test("no amplitude -> ones"):
            slm.source.pop("amplitude", None)
            assert np.all(slm._get_source_amplitude() == 1)

        with subtests.test("no phase -> zeros"):
            slm.source.pop("phase", None)
            assert np.all(slm._get_source_phase() == 0)

        with subtests.test("with amplitude -> returns it"):
            amp = np.random.rand(*slm.shape)
            slm.source["amplitude"] = amp
            np.testing.assert_array_equal(slm._get_source_amplitude(), amp)

        with subtests.test("non-cropping aperture skips the mask multiply but stays safe"):
            # The default "cropped" aperture masks nothing, so the source is returned
            # unchanged -- but as an independent array, so a caller mutating the result
            # (e.g. Hologram's in-place amplitude normalization) cannot corrupt source.
            slm.set_aperture("cropped")
            amp = np.random.rand(*slm.shape)
            slm.source["amplitude"] = amp.copy()
            got = slm._get_source_amplitude()
            assert np.array_equal(got, amp)
            assert got is not slm.source["amplitude"]
            got *= 2.0
            assert np.array_equal(slm.source["amplitude"], amp)   # source untouched
            # A real aperture still masks.
            slm.set_aperture(radius=0.3, units="frac")
            masked = slm._get_source_amplitude()
            assert np.array_equal(masked > 0, slm.aperture_mask)

    def test_info(self, slm):
        """info() for SimulatedSLM returns empty list."""
        assert slm.info(verbose=False) == []

    def test_plot(self, slm, subtests):
        """plot() runs without error for common argument combos."""
        with subtests.test("default"):
            ax = slm.plot()
            assert ax is not None

        with subtests.test("scalar limits"):
            ax = slm.plot(limits=0.5)
            assert ax is not None

        with subtests.test("2x2 limits"):
            ax = slm.plot(limits=[[0, 100], [0, 100]])
            assert ax is not None

        with subtests.test("bad limits raises"):
            with pytest.raises(ValueError, match="not recognized"):
                slm.plot(limits=[1, 2, 3])

    def test_plot_source(self, slm, subtests):
        """plot_source for measured and simulated distributions."""

        slm.set_source_analytic()
        slm.set_source_analytic(sim=True)

        with subtests.test("measured amplitude & phase"):
            slm.plot_source(sim=False)
            plt.show()

        with subtests.test("simulated"):
            slm.plot_source(sim=True)
            plt.show()

        with subtests.test("power mode"):
            slm.plot_source(power=True)
            plt.show()

        with subtests.test("missing sim keys raises"):
            src_backup = slm.source.copy()
            slm.source.pop("amplitude_sim", None)
            with pytest.raises(RuntimeError, match="Simulated"):
                slm.plot_source(sim=True)
            slm.source.update(src_backup)

        with subtests.test("missing measured keys raises"):
            src_backup = slm.source.copy()
            slm.source.pop("amplitude", None)
            with pytest.raises(RuntimeError, match="amplitude"):
                slm.plot_source(sim=False)
            slm.source.update(src_backup)

    def test_psf_and_spot_radius(self, slm, subtests):
        """get_point_spread_function_knm and get_spot_radius_kxy."""
        slm.set_source_analytic()
        slm.fit_aperture()

        with subtests.test("PSF shape matches SLM"):
            psf = slm.get_point_spread_function_knm()
            assert psf.shape == slm.shape

        with subtests.test("PSF with padded_shape"):
            psf = slm.get_point_spread_function_knm(padded_shape=(2048, 2048))
            assert psf.shape == (2048, 2048)

        with subtests.test("spot radius positive"):
            r = slm.get_spot_radius_kxy()
            assert r > 0


class TestSegmented:
    """Tests for SegmentedSLM and SLM.segment()."""

    def test_segment_grid(self, subtests):
        """segment() produces the right count, shapes, and refresh assignment."""
        parent = SimulatedSLM(resolution=(120, 128))  # shape (h=128, w=120)
        children = parent.segment((2, 3))             # 2 rows, 3 cols → 6 children

        with subtests.test("count"):
            assert len(children) == 6

        with subtests.test("shape"):
            # segment_shape = (128//2, 120//3) = (64, 40) → each child shape (h, w)
            for child in children:
                assert child.shape == (64, 40)

        with subtests.test("only last child has refresh"):
            assert all(not c.refresh for c in children[:-1])
            assert children[-1].refresh is True

        with subtests.test("segments are SegmentedSLM instances"):
            assert all(isinstance(c, SegmentedSLM) for c in children)

        parent.close()

    def test_segment_write_through(self, subtests):
        """Writing integer data to a segment is reflected in the correct parent region."""
        parent = SimulatedSLM(resolution=(128, 128))  # shape (128, 128)
        children = parent.segment((2, 2))             # 4 quadrants, each (64, 64)

        parent.display[:] = 0  # start with blank slate
        parent_writes = [0]

        def _count_hw(display):
            parent_writes[0] += 1
        parent._set_phase_hw = _count_hw

        # Write a distinct value per segment.
        for i, child in enumerate(children):
            val = i + 1
            child.set_phase(
                np.full(child.shape, val, dtype=parent.dtype),
                phase_correct=False,
            )
            with subtests.test(f"child {i} display updated"):
                assert np.all(child.display == val)
            if i < len(children) - 1:  # only the last child should update the parent immediately
                with subtests.test(f"parent unchanged after child {i} write"):
                    assert parent_writes[0] == 0
            else:
                with subtests.test("parent updated after last child write"):
                    assert parent_writes[0] == 1

        # Each quadrant of the parent display must equal the child's display.
        for i, child in enumerate(children):
            with subtests.test(f"quadrant {i}"):
                np.testing.assert_array_equal(
                    parent.display[child.extent_slice],
                    child.display,
                )

        # Quadrants must be disjoint: no two children wrote to the same pixel.
        covered = np.zeros(parent.shape, dtype=int)
        for child in children:
            covered[child.extent_slice] += 1
        with subtests.test("disjoint coverage"):
            assert np.all(covered[covered > 0] == 1)

        parent.close()

    def test_segment_boolean_window(self):
        """Non-rectangular boolean mask: only masked pixels are written to parent."""
        parent = SimulatedSLM(resolution=(64, 64))

        # Checkerboard-like mask in the upper-left 32×32 region.
        mask = np.zeros(parent.shape, dtype=bool)
        mask[:32, :32] = True
        mask[::2, ::2] = False  # punch holes so it is genuinely non-rectangular

        child = SegmentedSLM(parent, window=mask, name="sparse")
        assert child.subwindow is not None

        parent.display[:] = 0
        val = parent.bitresolution // 2
        child.set_phase(
            np.full(child.shape, val, dtype=parent.dtype),
            phase_correct=False,
        )

        # Pixels inside the subwindow should equal val; others must stay 0.
        sub = parent.display[child.extent_slice]
        assert np.all(sub[child.subwindow] == val)
        assert np.all(sub[~child.subwindow] == 0)

        parent.close()

    def test_segment_index_list_window(self):
        """Non-rectangular index-list window: only indexed pixels are written to parent."""
        parent = SimulatedSLM(resolution=(64, 64))

        # Diagonal stripe: y == x, within a 32x32 sub-region.
        coords = np.arange(32)
        y_ind = coords + 8   # rows 8..39
        x_ind = coords + 16  # cols 16..47

        child = SegmentedSLM(parent, window=(y_ind, x_ind), name="diag")
        assert child.subwindow is not None

        parent.display[:] = 0
        val = parent.bitresolution // 2
        child.set_phase(
            np.full(child.shape, val, dtype=parent.dtype),
            phase_correct=False,
        )

        # Each indexed pixel in the parent should equal val.
        assert np.all(parent.display[y_ind, x_ind] == val)
        # All other pixels must remain 0.
        mask = np.zeros(parent.shape, dtype=bool)
        mask[y_ind, x_ind] = True
        assert np.all(parent.display[~mask] == 0)

        parent.close()

    def test_segment_out_of_bounds(self):
        """A window that extends beyond the parent raises ValueError."""
        parent = SimulatedSLM(resolution=(64, 64))
        with pytest.raises(ValueError, match="out of bounds"):
            SegmentedSLM(parent, window=(50, 30, 0, 64), name="oob")
        parent.close()

    def test_segment_source_inherits(self):
        """Child source arrays are views into the parent source at the right region."""
        parent = SimulatedSLM(resolution=(64, 64))
        parent.set_source_analytic()   # populates source["amplitude"] and source["phase"]

        children = parent.segment((2, 2))
        for child in children:
            np.testing.assert_array_equal(
                child.source["amplitude"],
                parent.source["amplitude"][child.extent_slice],
            )

        parent.close()

    def test_segment_lut_inherits(self, subtests):
        """A segment tracks the parent's lut until given one of its own."""
        parent = SimulatedSLM(resolution=(64, 64))
        B = parent.bitresolution
        (first, second) = parent.segment(2)

        with subtests.test("no lut anywhere"):
            assert first.lut is None

        parent.set_gamma(np.arange(B) / B)

        with subtests.test("segments track the parent"):
            assert np.array_equal(first.lut, parent.lut)
            assert first._phase_to_lut == parent._phase_to_lut

        with subtests.test("a segment may carry its own"):
            second.set_gamma(_quadratic_gamma(B), lut_size=1 << 10)
            assert not np.array_equal(second.lut, parent.lut)
            assert second._phase_to_lut == pytest.approx((1 << 10) / (2 * np.pi))
            assert np.array_equal(first.lut, parent.lut)

        with subtests.test("clearing reverts to the parent"):
            second.set_gamma(None)
            assert np.array_equal(second.lut, parent.lut)
            assert second._phase_to_lut == parent._phase_to_lut

        with subtests.test("gamma tracks the parent alongside the lut"):
            np.testing.assert_array_equal(first.gamma, parent.gamma)
            level = np.full(first.shape, 100, dtype=first.dtype)
            first.set_phase(level, phase_correct=False)
            expected = np.mod(
                first._gamma_sign * 2 * np.pi * np.asarray(parent.gamma)[100], 2 * np.pi
            )
            np.testing.assert_allclose(first.phase, expected)

        with subtests.test("segments write the parent's mapping into its display"):
            phase = np.linspace(0, 2 * np.pi, first.shape[0] * first.shape[1])
            phase = phase.reshape(first.shape)
            first.set_phase(phase.copy(), phase_correct=False)
            np.testing.assert_array_equal(
                parent.display[first.extent_slice],
                parent._phase2gray(phase.copy(), out=np.zeros(first.shape, dtype=parent.dtype)),
            )

        parent.close()


class TestScreenMirrored:
    """Tests for ScreenMirrored's RGBA packing, which needs no display."""

    shape = (32, 48)

    @staticmethod
    def _pack(display, frame):
        """Run ScreenMirrored._pack against a bare instance."""
        slm = ScreenMirrored.__new__(ScreenMirrored)
        slm._display_rgba = None
        slm._pack(display, frame)
        return frame

    @classmethod
    def _blank(cls, xp=np):
        """An opaque black RGBA frame on the given backend."""
        frame = xp.zeros(cls.shape + (4,), dtype=xp.uint8)
        frame[:, :, 3] = 255
        return frame

    def test_pack(self, subtests):
        """Grayscale and three-plane data must land in R, G, B with alpha untouched."""
        gray = np.random.randint(0, 256, self.shape, dtype=np.uint8)
        planes = np.random.randint(0, 256, (3,) + self.shape, dtype=np.uint8)

        with subtests.test("grayscale fills all three channels"):
            frame = self._pack(gray, self._blank())
            for c in range(3):
                assert np.array_equal(frame[:, :, c], gray)

        with subtests.test("three planes keep index 0 in red"):
            frame = self._pack(planes, self._blank())
            for c in range(3):
                assert np.array_equal(frame[:, :, c], planes[c])

        with subtests.test("alpha is preserved"):
            assert np.all(self._pack(gray, self._blank())[:, :, 3] == 255)

        with subtests.test("mismatched shape raises"):
            with pytest.raises(ValueError):
                self._pack(np.zeros((3, 3), dtype=np.uint8), self._blank())

    @pytest.mark.gpu
    def test_pack_gpu(self, subtests):
        """Every combination of host/device display and frame must agree with the host path."""
        import cupy as cp

        gray = np.random.randint(0, 256, self.shape, dtype=np.uint8)
        planes = np.random.randint(0, 256, (3,) + self.shape, dtype=np.uint8)

        for name, display in (("grayscale", gray), ("three planes", planes)):
            reference = self._pack(display, self._blank())

            # A cupy frame stands in for the interop mode, where OpenGL memory is mapped.
            for label, d, xp in (
                ("cupy display, host frame", cp.asarray(display), np),
                ("cupy display, mapped frame", cp.asarray(display), cp),
                ("numpy display, mapped frame", display, cp),
            ):
                with subtests.test("{}: {}".format(name, label)):
                    frame = self._pack(d, self._blank(xp))
                    assert np.array_equal(cp.asnumpy(frame), reference)
