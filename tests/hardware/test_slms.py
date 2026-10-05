"""
Unit tests for the SLM base class and its subclasses.
"""

import os
from typing import ClassVar
import warnings

from conftest import driver_classes
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from slmsuite.hardware.slms.screenmirrored import ScreenMirrored
from slmsuite.hardware.slms.segmented import SegmentedSLM
from slmsuite.hardware.slms.simulated import SimulatedSLM
import slmsuite.hardware.slms.slm as slm_module
from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.slms.texasinstruments import PLM
from slmsuite.holography.toolbox import Aperture
from slmsuite.holography.toolbox.phase import zernike_sum
from slmsuite.misc.xp import as_numpy


def _quadratic_gamma(bitresolution):
    """A monotonic but distinctly non-uniform phase response."""
    levels = np.arange(bitresolution)
    return np.square(levels / (bitresolution - 1)) * (bitresolution - 1) / bitresolution


class TestSLM:
    """Tests for the SLM base class (via SimulatedSLM), and for its drivers."""

    @staticmethod
    def _slm(bitdepth=8, gpu=False, phase_scaling=1):
        return SimulatedSLM(
            resolution=(32, 32), bitdepth=bitdepth, gpu=gpu, wav_design_um=1 / phase_scaling
        )

    @staticmethod
    def _wrapped_phase(slm):
        """A random phase spanning several wraps."""
        return np.random.default_rng(0).uniform(-3 * np.pi, 3 * np.pi, slm.shape)

    def test_selftest(self, slm, subtests):
        """test() drives set_phase, the triggers and info, and leaves the phase as it found it."""
        slm.set_phase(np.full(slm.shape, 1.0), phase_correct=False)
        before = slm.phase.copy()

        with subtests.test("the self-test passes"):
            assert slm.test() is True

        with subtests.test("the displayed phase is restored"):
            np.testing.assert_array_equal(as_numpy(slm.phase), as_numpy(before))

        with subtests.test("a failing test() still restores the phase"):
            slm.info = lambda verbose=True: 1 / 0
            try:
                with pytest.raises(AssertionError, match="info"):
                    slm.test()
            finally:
                del slm.info
            np.testing.assert_array_equal(as_numpy(slm.phase), as_numpy(before))

    def test_init(self, slm, monkeypatch, subtests):
        """Validate constructor-derived attributes and conventions."""
        with subtests.test("resolution (w, h) transposes into shape (h, w)"):
            s = SimulatedSLM(resolution=(800, 600), wav_um=0.78, wav_design_um=1.064, pitch_um=10)
            assert s.shape == (600, 800)
            assert s.phase_scaling == pytest.approx(0.78 / 1.064)
            assert np.allclose(s.pitch_um, [10.0, 10.0])
            s.close()

        with subtests.test("bitresolution is two to the bitdepth, in the narrowest uint"):
            for bitdepth, dtype in ((4, np.uint8), (8, np.uint8), (10, np.uint16)):
                s = SimulatedSLM(resolution=(64, 64), bitdepth=bitdepth)
                assert s.bitresolution == 2**bitdepth
                assert s.dtype == np.dtype(dtype)
                s.close()

        with subtests.test("grid is an (x, y) float32 pair shaped like the SLM"):
            assert len(slm.grid) == 2
            assert all(g.shape == slm.shape and g.dtype == np.float32 for g in slm.grid)
            assert slm.phase.dtype == np.float32

        with subtests.test("the grid is measured from the aperture center"):
            (cx, cy) = (slm.shape[1] // 3, slm.shape[0] // 3)
            slm.set_aperture(radius=0.3, units="frac", center=(cx, cy))
            assert all(float(g[cy, cx]) == pytest.approx(0) for g in slm.grid)
            assert all(g.dtype == np.float32 for g in slm.grid)
            slm.set_aperture("cropped")

        with subtests.test("invalid pitch_um raises"):
            with pytest.raises(ValueError):
                SimulatedSLM(resolution=(128, 128), pitch_um=(0, 8))

        with subtests.test("gpu=False keeps the data in numpy"):
            s = self._slm()
            assert s.xp is np
            assert isinstance(s.phase, np.ndarray) and isinstance(s.display, np.ndarray)
            s.close()

        with subtests.test("gpu=True without cupy is refused"):
            monkeypatch.setattr(slm_module, "cp", np)
            with pytest.raises(ImportError):
                self._slm(gpu=True)

    def test_phase2gray(self, slm, subtests, benchmark):
        """_phase2gray wraps phase and quantizes it onto grayscale levels."""
        # _phase2gray is private and works in-place on the SLM's own backend; the
        # public set_phase() is what coerces a host array. So feed it slm.xp arrays.
        xp = slm.xp

        with subtests.test("benchmark"):
            phase = xp.asarray(np.random.uniform(0, 2 * np.pi, slm.shape).astype(np.float32))
            benchmark(slm._phase2gray, phase)

        with subtests.test("zero phase is the maximum level (sign convention)"):
            assert np.all(as_numpy(slm._phase2gray(xp.zeros(slm.shape))) == slm.bitresolution - 1)

        with subtests.test("a ramp of one level per level descends the staircase, wrapping"):
            for bitdepth in (5, 8):
                s = SimulatedSLM(resolution=(32, 32), bitdepth=bitdepth)
                B = s.bitresolution
                k = np.arange(-2 * B, 2 * B)
                phase = np.zeros(s.shape, dtype=np.float32)
                phase.ravel()[: k.size] = k * (2 * np.pi / B)
                gray = as_numpy(s._phase2gray(s.xp.asarray(phase)))
                np.testing.assert_array_equal(gray.ravel()[: k.size], (B - 1 - k) % B)
                s.close()

        with subtests.test("large phase gives the same gray as its wrap"):
            for p in (-1e4, 10 * np.pi, 1e4):
                np.testing.assert_array_equal(
                    as_numpy(slm._phase2gray(xp.full(slm.shape, p))),
                    as_numpy(slm._phase2gray(xp.full(slm.shape, np.mod(p, 2 * np.pi)))),
                )

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
        """set_phase checks and wraps phase, or displays integer data directly."""
        with subtests.test("benchmark"):
            phase = np.random.uniform(0, 2 * np.pi, slm.shape).astype(np.float32)
            benchmark(slm.set_phase, phase, phase_correct=False)

        with subtests.test("None zeros the phase"):
            slm.set_phase(None, phase_correct=False)
            assert np.all(slm.phase == 0)

        with subtests.test("the display is returned, not a copy of it"):
            display = slm.set_phase(np.zeros(slm.shape), phase_correct=False)
            assert display is slm.display
            assert display.shape == slm.shape
            assert display.dtype == slm.dtype

        with subtests.test("integer data of another type is refused"):
            for dtype in (np.uint16 if slm.dtype == np.uint8 else np.uint8, np.int64):
                with pytest.raises(TypeError, match="Unexpected integer type"):
                    slm.set_phase(np.zeros(slm.shape, dtype=dtype))

        with subtests.test("integer data is displayed verbatim, unpadded to shape"):
            level = slm.bitresolution // 2
            oversize = np.full((slm.shape[0] + 20, slm.shape[1] + 20), level, dtype=slm.dtype)
            slm.set_phase(oversize)
            assert slm.display.shape == slm.shape
            assert np.all(slm.display == level)
            assert slm.phase.dtype == np.float32

        with subtests.test("an integer write records the phase that gamma realizes"):
            s = self._slm(bitdepth=6)
            gamma = _quadratic_gamma(s.bitresolution)
            s.set_gamma(gamma)
            levels = np.arange(s.shape[0] * s.shape[1]) % s.bitresolution
            levels = levels.astype(s.dtype).reshape(s.shape)
            s.set_phase(levels, phase_correct=False)
            np.testing.assert_allclose(
                s.phase, np.mod(s._gamma_sign * 2 * np.pi * gamma[levels], 2 * np.pi)
            )
            np.testing.assert_array_equal(s._phase2gray(s.phase.copy()), levels)
            s.close()

        with subtests.test("phase_correct adds the source phase"):
            # One level: the stored phase is quantized, so an aligned offset is exact.
            offset = 2 * np.pi / slm.bitresolution
            slm.source["phase"] = np.full(slm.shape, offset)
            slm.set_phase(np.zeros(slm.shape), phase_correct=True)
            np.testing.assert_allclose(as_numpy(slm.phase), offset, rtol=1e-6)
            del slm.source["phase"]

        with subtests.test("the stored phase re-displays as displayed at any phase_scaling"):
            for phase_scaling in (1, 0.75, 1.3):
                s = self._slm(phase_scaling=phase_scaling)
                s.source["phase"] = np.random.default_rng(1).uniform(0, 1, s.shape)
                display = s.set_phase(self._wrapped_phase(s), phase_correct=True).copy()
                s.set_phase(s.phase.copy(), phase_correct=False)
                np.testing.assert_array_equal(s.display, display, err_msg=f"{phase_scaling=}")
                s.close()

        with subtests.test("write() is a deprecated alias"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                display = slm.write(np.zeros(slm.shape), phase_correct=False)
            assert any("deprecated" in str(w.message).lower() for w in caught)
            assert display is slm.display

        with subtests.test("execute and block reach the hardware which advertises them"):

            class Recording(SimulatedSLM):
                """An SLM advertising execute and block, recording what it is given."""

                calls: ClassVar[list] = []

                def _set_phase_hw(self, display, execute=True, block=True):
                    self.calls.append((execute, block))

            s = Recording(resolution=(32, 32))
            for kwargs, expected in (
                ({}, [(True, True)]),
                ({"execute": False}, [(False, True)]),
                ({"execute": False, "block": False}, [(False, False)]),
                ({"execute": True, "block": False}, [(True, False)]),
            ):
                s.calls.clear()
                s.set_phase(np.zeros(s.shape), phase_correct=False, **kwargs)
                assert s.calls == expected
            s.close()

        with subtests.test("hardware without execute or block refuses them"):
            for kwarg in ("execute", "block"):
                with pytest.raises(ValueError, match=kwarg):
                    slm.set_phase(np.zeros(slm.shape), **{kwarg: False})

    @pytest.mark.gpu
    def test_set_phase_gpu(self):
        """A device-backed SLM re-displays its stored phase as displayed."""
        for phase_scaling in (0.75, 1.3):
            s = self._slm(gpu=True, phase_scaling=phase_scaling)
            display = s.set_phase(self._wrapped_phase(s), phase_correct=False).copy()
            s.set_phase(s.phase.copy(), phase_correct=False)
            np.testing.assert_array_equal(
                s.display.get(), display.get(), err_msg=f"{phase_scaling=}"
            )
            s.close()

    def test_set_phase_settle(self, temp_dir, monkeypatch, subtests):
        """`settle` waits the SLM's settle time when True, or that many seconds when a number."""
        monkeypatch.chdir(temp_dir)
        s = self._slm()
        s.settle_time_s = 100
        sleeps = []
        monkeypatch.setattr(slm_module.time, "sleep", sleeps.append)

        cases = [(True, 100), (np.True_, 100), (30, 30), (2.5, 2.5)]
        for settle, expected in cases:
            with subtests.test(f"settle={settle!r} waits {expected} s less the render time"):
                sleeps.clear()
                s.set_phase(None, settle=settle)
                assert len(sleeps) == 1 and expected - 1 < sleeps[0] <= expected

        for settle in (False, 0, 0.0):
            with subtests.test(f"settle={settle!r} does not wait"):
                sleeps.clear()
                s.set_phase(None, settle=settle)
                assert sleeps == []

        with subtests.test("None follows the SLM's settle default"):
            (s.settle, sleeps[:]) = (True, [])
            s.set_phase(None)
            assert len(sleeps) == 1 and 99 < sleeps[0] <= 100

        s.save_phase()
        for settle, expected in [(True, [100]), (2.5, [2.5]), (False, [])]:
            with subtests.test(f"load_phase(settle={settle!r}) waits {expected}"):
                sleeps.clear()
                s.load_phase(None, settle=settle)
                assert sleeps == expected

        s.close()

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
            # Short of a cycle, the unrealizable phases pile up at the nearer endpoint.
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

        with subtests.test("a sparse response with its levels fills every level"):
            sampled = levels[::8]
            s.set_gamma(_quadratic_gamma(B)[sampled], levels=sampled)
            np.testing.assert_allclose(
                as_numpy(s.gamma), s._interpolate_gamma(_quadratic_gamma(B)[sampled], sampled)
            )
            assert len(as_numpy(s.gamma)) == B

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
            np.testing.assert_allclose(s._interpolate_gamma(levels / B, levels), ideal)

            # Sampling that starts above zero must close at the bottom as well as the top.
            edged = np.arange(16, B, 16)
            np.testing.assert_allclose(s._interpolate_gamma(edged / B, edged), ideal)

        with subtests.test("sample order does not matter"):
            # A linear response interpolates alike between any pair, so it cannot see the sort.
            curved = _quadratic_gamma(B)[levels.astype(int)]
            order = np.random.permutation(len(levels))
            np.testing.assert_allclose(
                s._interpolate_gamma(curved[order], levels[order]),
                s._interpolate_gamma(curved, levels),
            )

        with subtests.test("non-uniform sampling follows the levels given"):
            sparse = np.array([0, 3, 9, 40, 150, 251])
            np.testing.assert_allclose(s._interpolate_gamma(sparse / B, sparse)[sparse], sparse / B)

        with subtests.test("degenerate input raises"):
            with pytest.raises(ValueError, match="pair with gamma"):
                s._interpolate_gamma([0, 0.5], [0, 1, 2])
            with pytest.raises(ValueError, match="two distinct levels"):
                s._interpolate_gamma([0, 0.5], [3, 3])
            # Levels spanning the whole range would close onto themselves.
            with pytest.raises(ValueError, match="span less than"):
                s._interpolate_gamma([0, 0.3, 1.0], [0, 100, 400])

        s.close()

    @pytest.mark.gpu
    def test_gamma_backend_parity(self, subtests):
        """A GPU-backed SLM must display exactly what a CPU-backed one does."""
        import cupy as cp

        from slmsuite.holography.toolbox.phase import blaze

        (cpu, gpu) = (self._slm(), self._slm(gpu=True))
        B = cpu.bitresolution

        with subtests.test("linear path"):
            phase = blaze(cpu, vector=(0.05, 0.03))
            got = gpu.set_phase(phase.copy(), phase_correct=False)
            want = cpu.set_phase(phase.copy(), phase_correct=False)
            np.testing.assert_array_equal(np.asarray(got.get()), want)

        with subtests.test("lut path"):
            gamma = _quadratic_gamma(B)
            cpu.set_gamma(gamma)
            gpu.set_gamma(gamma)
            phase = blaze(cpu, vector=(0.05, 0.03))
            got = gpu.set_phase(phase.copy(), phase_correct=False)
            want = cpu.set_phase(phase.copy(), phase_correct=False)
            np.testing.assert_array_equal(np.asarray(got.get()), want)

        with subtests.test("interpolate_gamma accepts a device response"):
            levels = np.arange(0, B, 16)
            np.testing.assert_array_equal(
                gpu._interpolate_gamma(cp.asarray(levels / B), cp.asarray(levels)),
                cpu._interpolate_gamma(levels / B, levels),
            )

        with subtests.test("gamma_sim accepts a device response"):
            sim = SimulatedSLM((16, 16), gpu=True)
            sim.gamma_sim = cp.asarray(_quadratic_gamma(sim.bitresolution))
            assert np.array_equal(sim.gamma_sim, _quadratic_gamma(sim.bitresolution))
            sim.close()

        (cpu.close(), gpu.close())

    def test_save_load_phase(self, slm, temp_dir, monkeypatch, subtests):
        """Round-trip save/load of phase data."""
        monkeypatch.chdir(temp_dir)

        with subtests.test("loading from a directory with no phase file raises"):
            with pytest.raises(FileNotFoundError):
                slm.load_phase(None)

        with subtests.test("save then load restores the display"):
            slm.set_phase(np.random.rand(*slm.shape) * 2 * np.pi, phase_correct=False)
            saved_display = slm.display.copy()
            path = slm.save_phase()
            assert os.path.exists(path)
            slm.set_phase(None, phase_correct=False)
            slm.load_phase(None)
            np.testing.assert_array_equal(as_numpy(slm.display), as_numpy(saved_display))

    def test_load_vendor_phase_correction(self, temp_dir, subtests):
        """A vendor image of gray levels becomes the source phase that displays those levels."""
        slm = SimulatedSLM(resolution=(48, 32), wav_um=1.0, wav_design_um=1.0, gpu=False)
        levels = np.random.default_rng(0).integers(0, slm.bitresolution, slm.shape).astype(np.uint8)
        path = os.path.join(temp_dir, "vendor.png")
        Image.fromarray(levels).save(path)
        full = np.copy(slm.load_vendor_phase_correction(path))

        with subtests.test(
            "at unit phase_scaling the correction alone displays the vendor's levels"
        ):
            slm.set_phase(None, phase_correct=True)
            np.testing.assert_array_equal(as_numpy(slm.display), levels)

        with subtests.test("a smaller file is padded to the slm about its center"):
            Image.fromarray(levels[4:-4, 4:-4]).save(path)
            padded = slm.load_vendor_phase_correction(path)
            assert padded.shape == slm.shape
            np.testing.assert_array_equal(padded[4:-4, 4:-4], full[4:-4, 4:-4])

        with subtests.test("a file larger on one axis and smaller on the other raises"):
            Image.fromarray(np.zeros((slm.shape[0] + 4, slm.shape[1] - 4), np.uint8)).save(path)
            with pytest.raises(ValueError, match="pad"):
                slm.load_vendor_phase_correction(path)

        with subtests.test("a color image raises"):
            Image.fromarray(np.zeros((*slm.shape, 3), np.uint8)).save(path)
            with pytest.raises(ValueError, match="2D"):
                slm.load_vendor_phase_correction(path)

    def test_set_source_analytic(self, slm, subtests):
        """set_source_analytic fills source from an analytic profile."""
        with subtests.test("units scale the grid the profile is fit to"):
            norm = as_numpy(slm.set_source_analytic(units="norm")["amplitude"]).copy()
            for units in ("um", "mm"):
                np.testing.assert_allclose(
                    as_numpy(slm.set_source_analytic(units=units)["amplitude"]), norm, atol=1e-6
                )
            # "frac" scales each axis by its own extent, so it alone is anisotropic.
            assert not np.allclose(
                as_numpy(slm.set_source_analytic(units="frac")["amplitude"]), norm
            )

        with subtests.test("a real profile carries phase_offset as its phase"):
            np.testing.assert_allclose(
                as_numpy(slm.set_source_analytic(phase_offset=0.3)["phase"]), 0.3
            )

        with subtests.test("sim stores a separate distribution"):
            src = slm.set_source_analytic(sim=True)
            assert "amplitude_sim" in src and "phase_sim" in src

        with subtests.test("a fit_function may be passed directly"):
            src = slm.set_source_analytic(fit_function=lambda xy, a=1: a * np.ones_like(xy[0]))
            np.testing.assert_allclose(as_numpy(src["amplitude"]), 1.0)

        with subtests.test("unrecognized units raise"):
            with pytest.raises(RuntimeError, match="Did not recognize"):
                slm.set_source_analytic(units="bad_unit")

    def test_fit_aperture(self, slm, subtests):
        """fit_aperture sets the aperture from the measured source amplitude."""
        with subtests.test("the fit recovers the 1/e radius and center of a known source"):
            # A fifth of the smaller extent leaves the Gaussian tails well inside the SLM.
            w = np.amin([float(g.max()) for g in slm.grid]) / 5
            slm.set_source_analytic(x0=0, y0=0, a=1, c=0, wx=w, wy=w)
            assert isinstance(slm.fit_aperture(method="moments"), Aperture)
            assert slm.source_radius == pytest.approx(np.sqrt(2) * w)
            assert np.allclose(as_numpy(slm.aperture.center), 0)
            # A Gaussian has no rolloff edge, so the pupil sits where its power falls to edge_level.
            pupil = 1 / slm.aperture._isotropic_scale()
            assert pupil == pytest.approx(np.sqrt(2) * w * np.sqrt(np.log(1 / 0.1) / 2), rel=2e-2)

        with subtests.test("a flat-top source gets a pupil at its edge, not twice its radius"):
            # A uniform disk of radius a has a moment (1/e) radius of a too.
            slm.set_aperture("cropped")
            a = np.amin([float(g.max()) for g in slm.grid]) / 2
            slm.source["amplitude"] = (np.hypot(*map(as_numpy, slm.grid)) <= a).astype(float)
            # edge_level=0.5 places the pupil at the middle of the rolloff, the geometric edge.
            slm.fit_aperture(method="moments", edge_level=0.5)
            assert slm.source_radius == pytest.approx(a, rel=2e-2)
            assert 1 / slm.aperture._isotropic_scale() == pytest.approx(a, rel=2e-2)

        with subtests.test("a clipped, unevenly lit source is fit at its edge, not its centroid"):
            slm.set_aperture("cropped")
            (x, y) = map(as_numpy, slm.grid)
            # An iris larger than the SLM's height, off center, so the display clips its top.
            (cx, cy, a) = (0.15 * float(x.max()), -0.1 * float(y.max()), float(y.max()))
            r = np.hypot(x - cx, y - cy)
            iris = np.clip((a - r) / (0.03 * a) + 0.5, 0, 1)
            # Brighter to one side, with a hot spot, both of which pull the centroid.
            lit = (0.5 + 0.5 * (x - x.min()) / np.ptp(x)) * (
                1 + np.exp(-((x - cx - 0.5 * a) ** 2 + (y - cy + 0.3 * a) ** 2) / (0.2 * a) ** 2)
            )
            power = iris * lit
            slm.source["amplitude"] = np.sqrt(power)

            centroid = np.array([np.sum(x * power), np.sum(y * power)]) / np.sum(power)
            assert np.hypot(*(centroid - (cx, cy))) > 0.05 * a

            slm.fit_aperture()
            np.testing.assert_allclose(as_numpy(slm.aperture.center), (cx, cy), atol=0.01 * a)
            assert 1 / slm.aperture._isotropic_scale() == pytest.approx(a, rel=2e-2)

        with subtests.test("a lower edge_level widens the pupil about the same center"):
            slm.fit_aperture(edge_level=0.5)
            (center, pupil) = (
                as_numpy(slm.aperture.center).copy(),
                1 / slm.aperture._isotropic_scale(),
            )
            slm.fit_aperture(edge_level=0.01)
            np.testing.assert_allclose(as_numpy(slm.aperture.center), center)
            assert 1 / slm.aperture._isotropic_scale() > pupil

        with subtests.test("edge_level outside (0, 1) raises"):
            with pytest.raises(ValueError, match="edge_level"):
                slm.fit_aperture(edge_level=0)

        with subtests.test("an unmeasured source guesses a quarter of the smallest extent"):
            slm.source.pop("amplitude", None)
            slm.fit_aperture()
            assert slm.source_radius == pytest.approx(
                0.25 * np.min((slm.shape[1] * slm.pitch[0], slm.shape[0] * slm.pitch[1]))
            )

        with subtests.test("refitting does not drift the grid or the radius"):
            slm.set_source_analytic()
            slm.fit_aperture()
            (grid, radius) = ([g.copy() for g in slm.grid], slm.source_radius)
            slm.fit_aperture()
            assert all(np.array_equal(a, b) for (a, b) in zip(grid, slm.grid))
            assert slm.source_radius == radius

        with subtests.test("an unknown method raises"):
            with pytest.raises(ValueError, match="method"):
                slm.fit_aperture(method="bogus")

    def test_set_aperture(self, slm, subtests):
        """set_aperture and the mask, scaling, and source radius derived from it."""
        (cx, cy) = (slm.shape[1] / 2 + 120, slm.shape[0] / 2 - 80)

        with subtests.test("the default cropped aperture masks nothing"):
            slm.set_aperture("cropped")
            assert np.all(slm.aperture_mask)

        with subtests.test("a radius is the source radius, and crops the mask"):
            slm.set_aperture(radius=0.4, units="norm")
            assert slm.source_radius == pytest.approx(0.4)
            slm.set_aperture(radius=0.3, units="frac")
            assert 0 < np.mean(slm.aperture_mask) < 1

        with subtests.test("a new spec drops a stored source radius; recentering keeps it"):
            slm.set_aperture(radius=0.4, units="norm")
            slm.set_aperture(center=(cx, cy))
            assert slm.source_radius == pytest.approx(0.4)
            slm.set_aperture(0.5)
            assert slm.source_radius == pytest.approx(1.0)

        with subtests.test("aperture_mask is the resolved Aperture's mask"):
            # slm.grid is already shifted, so a resolved aperture must not re-subtract.
            for spec in (
                {"spec": "cropped"},
                {"radius": 0.3, "units": "frac"},
                {"radius": 0.3, "units": "frac", "center": (cx, cy)},
            ):
                slm.set_aperture(**spec)
                assert Aperture.resolve(slm) is slm.aperture
                assert np.array_equal(
                    as_numpy(slm.aperture_mask), as_numpy(Aperture.resolve(slm).mask)
                )

        with subtests.test("spec and radius are mutually exclusive"):
            with pytest.raises(ValueError):
                slm.set_aperture("circular", radius=0.3)

        with subtests.test("source_radius rejects an anisotropic aperture"):
            slm.set_aperture((0.01, 0.02))
            with pytest.raises(ValueError, match="isotropic"):
                _ = slm.source_radius

        with subtests.test("zernike_sum masks a centered aperture consistently"):
            slm.set_aperture(radius=0.3, units="frac", center=(cx, cy))
            outside = np.isnan(as_numpy(zernike_sum(slm, [4], [1.0], use_mask=np.nan)))
            assert np.array_equal(outside, ~as_numpy(slm.aperture_mask))

    def test_get_source_amplitude(self, slm, subtests):
        """_get_source_amplitude falls back to unity and applies the aperture."""
        with subtests.test("an unmeasured source is unity"):
            slm.source.pop("amplitude", None)
            slm.set_aperture("cropped")
            assert np.all(slm._get_source_amplitude() == 1)

        with subtests.test("a measured source is returned as an independent copy"):
            # A caller normalizing the result in place must not corrupt slm.source.
            amp = np.random.rand(*slm.shape)
            slm.source["amplitude"] = amp.copy()
            got = slm._get_source_amplitude()
            np.testing.assert_array_equal(as_numpy(got), amp)
            got *= 2.0
            np.testing.assert_array_equal(as_numpy(slm.source["amplitude"]), amp)

        with subtests.test("an aperture masks the source to its own support"):
            slm.source.pop("amplitude", None)
            slm.set_aperture(radius=0.3, units="frac")
            assert np.array_equal(slm._get_source_amplitude() > 0, slm.aperture_mask)

    def test_info(self, slm):
        """info() lists the displays this SLM class can find, empty where unsupported."""
        assert isinstance(slm.info(verbose=False), list)

    def test_get_source_phase(self, slm):
        """_get_source_phase falls back to zero where the source is unmeasured."""
        slm.source.pop("phase", None)
        assert np.all(slm._get_source_phase() == 0)

    def test_plot(self, slm, subtests):
        """plot() renders wrapped phase in units of pi."""
        phase = np.linspace(-4 * np.pi, 6 * np.pi, slm.shape[0] * slm.shape[1])
        phase = phase.reshape(slm.shape).astype(np.float32)

        with subtests.test("the last written phase is plotted by default"):
            slm.set_phase(phase, phase_correct=False)
            default = np.asarray(slm.plot().get_images()[0].get_array())
            plt.close("all")
            explicit = np.asarray(slm.plot(phase=slm.phase).get_images()[0].get_array())
            plt.close("all")
            np.testing.assert_array_equal(default, explicit)

        with subtests.test("phase is rendered wrapped, in units of pi"):
            ax = slm.plot(phase=phase)
            im = ax.get_images()[0]
            np.testing.assert_allclose(np.asarray(im.get_array()), np.mod(phase, 2 * np.pi) / np.pi)
            assert im.get_cmap().name == "twilight"
            assert im.get_interpolation() == "none"
            np.testing.assert_allclose(im.get_clim(), (0, 2))
            plt.close("all")

        with subtests.test("colorbar drawn only when requested, labeled in pi"):
            ax = slm.plot(phase=phase, cbar=True)
            axes = ax.get_figure().axes
            assert len(axes) == 2
            assert [t.get_text() for t in axes[1].get_yticklabels()] == [
                "$0\\pi$",
                "$1\\pi$",
                "$2\\pi$",
            ]
            plt.close("all")
            ax = slm.plot(phase=phase, cbar=False)
            assert len(ax.get_figure().axes) == 1
            plt.close("all")

        with subtests.test("labels and title applied only when the phase fills the SLM"):
            ax = slm.plot(phase=phase, title="MyTitle")
            assert ax.get_title() == "MyTitle"
            assert ax.get_xlabel() == "SLM $n$ [pix]"
            assert ax.get_ylabel() == "SLM $m$ [pix]"
            plt.close("all")
            ax = slm.plot(phase=phase[::2, ::2])
            assert ax.get_xlabel() == "" and ax.get_ylabel() == ""
            plt.close("all")

        with subtests.test("aperture overlaid only when it crops and is requested"):
            ax = slm.plot(phase=phase, aperture=True)
            assert len(ax.collections) == 0
            plt.close("all")

            slm.set_aperture(radius=0.3, units="frac", center=(100, 100))
            ax = slm.plot(phase=phase, aperture=True)
            assert len(ax.collections) == 1
            plt.close("all")
            ax = slm.plot(phase=phase, aperture=False)
            assert len(ax.collections) == 0
            plt.close("all")
            ax = slm.plot(phase=phase[::2, ::2], aperture=True)
            assert len(ax.collections) == 0
            plt.close("all")
            slm.set_aperture("cropped")

    def test_plot_source(self, slm, subtests):
        """plot_source renders the measured or simulated source in two panels."""
        slm.set_source_analytic()
        slm.set_source_analytic(sim=True)

        with subtests.test("each mode plots the distribution its title names"):
            for kwargs, title, expected in (
                ({"sim": False}, "Source Amplitude", slm.source["amplitude"]),
                ({"power": True}, "Source Power", np.square(slm.source["amplitude"])),
                ({"sim": True}, "Simulated Source Amplitude", slm.source["amplitude_sim"]),
            ):
                axs = slm.plot_source(**kwargs)
                assert axs[1].get_title() == title
                np.testing.assert_allclose(
                    np.asarray(axs[1].get_images()[0].get_array()), as_numpy(expected)
                )
                plt.close("all")

        with subtests.test("a missing distribution raises"):
            for key, kwargs, match in (
                ("amplitude_sim", {"sim": True}, "Simulated"),
                ("amplitude", {"sim": False}, "amplitude"),
            ):
                stored = slm.source.pop(key)
                with pytest.raises(RuntimeError, match=match):
                    slm.plot_source(**kwargs)
                slm.source[key] = stored

    def test_get_point_spread_function_knm(self, slm, subtests):
        """The PSF transforms the source amplitude, conserving its power."""
        slm.set_source_analytic()
        power = float(np.sum(np.square(as_numpy(slm._get_source_amplitude()))))

        # Power is conserved exactly in exact arithmetic, but these are float32 FFTs, so
        # the sum only agrees to float32 precision -- and the summation order (and so the
        # last bits) differs between the numpy and cupy backends.
        with subtests.test("an unpadded PSF has the SLM's shape"):
            psf = slm.get_point_spread_function_knm()
            assert psf.shape == slm.shape
            assert float(np.sum(np.square(as_numpy(psf)))) == pytest.approx(power, rel=1e-5)

        with subtests.test("padding changes the resolution, not the power"):
            psf = slm.get_point_spread_function_knm(padded_shape=(2048, 2048))
            assert psf.shape == (2048, 2048)
            assert float(np.sum(np.square(as_numpy(psf)))) == pytest.approx(power, rel=1e-5)

    def test_get_spot_radius_kxy(self, slm_small, subtests):
        """The farfield 1/e amplitude radius of a gaussian source of 1/e amplitude radius w."""
        pitch = float(np.mean(slm_small.pitch))
        w = 20 * pitch  # The pupil, at 2 w, stays inside the grid.
        slm_small.set_aperture(radius=w)
        radius = float(slm_small.get_spot_radius_kxy())

        with subtests.test("exp(-r^2/w^2) transforms to exp(-pi^2 w^2 k^2), so it is 1/(pi w)"):
            assert radius == pytest.approx(1 / (np.pi * w), rel=1e-6)

        with subtests.test("it is the 1/e radius of the point spread function"):
            (x, y) = [as_numpy(g) for g in slm_small.grid]
            slm_small.source["amplitude"] = np.exp(-(x**2 + y**2) / w**2)
            pad = 2048
            psf = as_numpy(slm_small.get_point_spread_function_knm(padded_shape=(pad, pad)))
            row = psf[pad // 2, pad // 2 :] / psf[pad // 2, pad // 2]
            i = np.argmax(row < np.exp(-1))
            knm = i - 1 + (row[i - 1] - np.exp(-1)) / (row[i - 1] - row[i])
            # The pupil truncates the gaussian at 2 w, which widens the spot slightly.
            assert radius == pytest.approx(knm / (pad * pitch), rel=3e-2)

    @pytest.mark.parametrize(
        "driver", driver_classes(SLM), ids=lambda cls: cls.__module__.rsplit(".", 1)[-1]
    )
    def test_driver_is_concrete(self, driver):
        """Every shipped SLM driver implements the whole abstract interface."""
        assert not driver.__abstractmethods__, (
            f"{driver.__module__}.{driver.__name__} leaves "
            f"{sorted(driver.__abstractmethods__)} abstract, so it cannot be instantiated."
        )


class TestSimulatedSLM:
    """
    ``SimulatedSLM`` accepts a *measured* source in place of a simulated one, which is
    what lets a wavefront-calibrated experiment be carried into simulation.
    """

    RESOLUTION = (64, 48)

    def _slm(self, source):
        return SimulatedSLM(self.RESOLUTION, source=source)

    def test_init(self, subtests):
        """The measured phase is a correction, so the aberration simulated is its negative."""
        shape = np.flip(self.RESOLUTION)

        with subtests.test("no source is an ideal one"):
            slm = self._slm(None)
            assert np.allclose(slm.source["amplitude_sim"], 1)
            assert np.allclose(slm.source["phase_sim"], 0)

        with subtests.test("a measured source carries over, its phase negated"):
            (amplitude, phase) = (np.random.rand(*shape), np.random.rand(*shape))
            slm = self._slm({"amplitude": amplitude, "phase": phase})
            assert np.allclose(slm.source["amplitude_sim"], amplitude)
            assert np.allclose(slm.source["phase_sim"], -phase)

        with subtests.test("the simulated truth does not alias the measurement"):
            slm = self._slm({"amplitude": np.random.rand(*shape)})
            before = np.copy(as_numpy(slm.source["amplitude_sim"]))
            slm.source["amplitude"] *= 0
            assert np.array_equal(as_numpy(slm.source["amplitude_sim"]), before)

        # A vendor phase correction comes with no measured amplitude, and an amplitude
        # measurement can arrive before any phase.
        for measured in ("amplitude", "phase"):
            with subtests.test(f"an unmeasured half is ideal, given only {measured}"):
                value = np.random.rand(*shape)
                slm = self._slm({measured: value, "r2": np.ones(shape)})
                if measured == "amplitude":
                    assert np.allclose(slm.source["amplitude_sim"], value)
                    assert np.allclose(slm.source["phase_sim"], 0)
                else:
                    assert np.allclose(slm.source["amplitude_sim"], 1)
                    assert np.allclose(slm.source["phase_sim"], -value)

        with subtests.test("an explicit truth is used as given, not derived"):
            truth = np.random.rand(*shape)
            slm = self._slm(
                {
                    "amplitude": np.zeros(shape),
                    "phase": np.zeros(shape),
                    "amplitude_sim": truth,
                    "phase_sim": truth,
                }
            )
            assert np.allclose(slm.source["amplitude_sim"], truth)
            assert np.allclose(slm.source["phase_sim"], truth)


class TestSegmentedSLM:
    """Tests for SegmentedSLM and SLM.segment()."""

    def test_segment(self, subtests):
        """segment() tiles the parent into equal children, the last one refreshing."""
        parent = SimulatedSLM(resolution=(120, 128))
        children = parent.segment((2, 3))

        with subtests.test("one child per (row, column) of the tiling"):
            assert len(children) == 6
            assert all(isinstance(c, SegmentedSLM) and c.parent is parent for c in children)

        with subtests.test("each child is the parent's shape divided by the tiling"):
            assert all(c.shape == (128 // 2, 120 // 3) for c in children)

        with subtests.test("only the last child refreshes the parent"):
            assert [c.refresh for c in children] == [False] * 5 + [True]

        parent.close()

    def test_init(self, subtests):
        """A segment covers the window it is given, in any of its formats."""
        parent = SimulatedSLM(resolution=(64, 64))
        parent.set_source_analytic()

        with subtests.test("a rectangular window inherits the parent's source"):
            child = SegmentedSLM(parent, window=(16, 32, 8, 24), name="rect")
            assert child.subwindow is None
            assert child.shape == (24, 32)
            np.testing.assert_array_equal(
                as_numpy(child.source["amplitude"]),
                as_numpy(parent.source["amplitude"][tuple(child.extent_slice)]),
            )

        with subtests.test("a non-rectangular window keeps a subwindow of its extent"):
            mask = np.zeros(parent.shape, dtype=bool)
            mask[:32, :32] = True
            child = SegmentedSLM(parent, window=mask, name="sparse")
            assert child.subwindow is not None
            assert child.shape == (32, 32)

        with subtests.test("a window past the parent's edge raises"):
            with pytest.raises(ValueError, match="out of bounds"):
                SegmentedSLM(parent, window=(50, 30, 0, 64), name="oob")

        parent.close()

    def test_set_phase_hw(self, subtests):
        """A segment writes into its own region of the parent and nowhere else."""
        parent = SimulatedSLM(resolution=(128, 128))
        children = parent.segment((2, 2))
        level = parent.bitresolution // 2

        writes = []
        parent._set_phase_hw = writes.append
        for i, child in enumerate(children):
            child.set_phase(np.full(child.shape, i + 1, dtype=parent.dtype), phase_correct=False)

        with subtests.test("the parent is written once, by the refreshing child"):
            assert len(writes) == 1

        with subtests.test("each child's display lands in its own quadrant"):
            covered = np.zeros(parent.shape, dtype=int)
            for child in children:
                np.testing.assert_array_equal(
                    as_numpy(parent.display[tuple(child.extent_slice)]),
                    as_numpy(child.display),
                )
                covered[tuple(child.extent_slice)] += 1
            assert np.all(covered == 1)

        with subtests.test("a non-rectangular window leaves the pixels it excludes alone"):
            mask = np.zeros(parent.shape, dtype=bool)
            mask[:32, :32] = True
            mask[::2, ::2] = False
            (rows, cols) = (np.arange(32) + 8, np.arange(32) + 16)
            diagonal = np.zeros(parent.shape, dtype=bool)
            diagonal[rows, cols] = True

            for window, written in ((mask, mask), ((rows, cols), diagonal)):
                parent.display[:] = 0
                child = SegmentedSLM(parent, window=window, name="sparse")
                child.set_phase(
                    np.full(child.shape, level, dtype=parent.dtype), phase_correct=False
                )
                assert np.all(parent.display[written] == level)
                assert np.all(parent.display[~written] == 0)

        parent.close()

    def test_lut(self, subtests):
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
            np.testing.assert_array_equal(as_numpy(first.gamma), as_numpy(parent.gamma))
            level = np.full(first.shape, 100, dtype=first.dtype)
            first.set_phase(level, phase_correct=False)
            expected = np.mod(
                first._gamma_sign * 2 * np.pi * as_numpy(parent.gamma)[100], 2 * np.pi
            )
            np.testing.assert_allclose(as_numpy(first.phase), expected)

        with subtests.test("segments write the parent's mapping into its display"):
            xp = parent.xp
            phase = np.linspace(0, 2 * np.pi, first.shape[0] * first.shape[1])
            phase = phase.reshape(first.shape)
            first.set_phase(phase.copy(), phase_correct=False)
            np.testing.assert_array_equal(
                as_numpy(parent.display[first.extent_slice]),
                as_numpy(
                    parent._phase2gray(
                        xp.asarray(phase), out=xp.zeros(first.shape, dtype=parent.dtype)
                    )
                ),
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
        frame = xp.zeros((*cls.shape, 4), dtype=xp.uint8)
        frame[:, :, 3] = 255
        return frame

    def test_pack(self, subtests):
        """Grayscale and three-plane data must land in R, G, B with alpha untouched."""
        gray = np.random.randint(0, 256, self.shape, dtype=np.uint8)
        planes = np.random.randint(0, 256, (3, *self.shape), dtype=np.uint8)

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
        planes = np.random.randint(0, 256, (3, *self.shape), dtype=np.uint8)

        for name, display in (("grayscale", gray), ("three planes", planes)):
            reference = self._pack(display, self._blank())

            # A cupy frame stands in for the interop mode, where OpenGL memory is mapped.
            for label, d, xp in (
                ("cupy display, host frame", cp.asarray(display), np),
                ("cupy display, mapped frame", cp.asarray(display), cp),
                ("numpy display, mapped frame", display, cp),
            ):
                with subtests.test(f"{name}: {label}"):
                    frame = self._pack(d, self._blank(xp))
                    assert np.array_equal(cp.asnumpy(frame), reference)


class TestPLM:
    """Tests for PLM's bit packing, which needs no display."""

    def test_bitpack(self, subtests):
        """Test PLM.bitpack."""
        shape = (4, 6)
        bitmaps = [np.zeros(shape, dtype=np.uint8) for _ in range(8)]
        bitmaps[7][:] = 1
        bitmaps[0][0, 0] = 1

        with subtests.test("bit k of the packed byte is the lsb of bitmap k"):
            packed = PLM.bitpack(bitmaps)
            assert packed.shape == (1, *shape)
            assert packed.dtype == np.uint8
            assert packed[0, 0, 0] == 129 and packed[0, 1, 1] == 128

        with subtests.test("24 bitmaps pack into three channels"):
            assert PLM.bitpack(bitmaps * 3).shape == (3, *shape)

    @pytest.mark.gpu
    def test_bitpack_gpu(self):
        """Device bitmaps pack to the same host array."""
        import cupy as cp

        bitmaps = list(np.random.default_rng(0).integers(0, 2, (8, 4, 6), dtype=np.uint8))
        packed = PLM.bitpack([cp.asarray(b) for b in bitmaps])
        assert isinstance(packed, np.ndarray)
        assert np.array_equal(packed, PLM.bitpack(bitmaps))
