"""
Unit tests for SLM base class.
"""
import os
import tempfile
import warnings

import pytest
import numpy as np
import matplotlib.pyplot as plt

from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.hardware.slms.segmented import SegmentedSLM


class TestSLM:
    """Tests for the SLM base class (via SimulatedSLM)."""

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

    def test_phase2gray(self, slm, subtests, benchmark):
        """Edge cases for _phase2gray not covered by .test()."""
        with subtests.test("benchmark"):
            phase = np.random.uniform(0, 2 * np.pi, slm.shape).astype(np.float32)
            benchmark(slm._phase2gray, phase)

        with subtests.test("negative phase wraps to valid gray"):
            phase = -np.ones(slm.shape) * np.pi
            gray = slm._phase2gray(phase)
            assert np.all(gray >= 0) and np.all(gray < slm.bitresolution)

        with subtests.test("large phase wraps to valid gray"):
            phase = np.ones(slm.shape) * 10 * np.pi
            gray = slm._phase2gray(phase)
            assert np.all(gray >= 0) and np.all(gray < slm.bitresolution)

        with subtests.test("zero phase -> display max (sign convention)"):
            gray = slm._phase2gray(np.zeros(slm.shape))
            assert np.all(gray == slm.bitresolution - 1)

        with subtests.test("non-standard bitdepth uses bitwise_and mask"):
            s = SimulatedSLM(resolution=(64, 64), bitdepth=5)
            phase = np.linspace(0, 4 * np.pi, 64 * 64).reshape(s.shape)
            gray = s._phase2gray(phase)
            assert np.all(gray >= 0) and np.all(gray < s.bitresolution)
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
            # Regression for the resolve double-centering bug: with a non-default
            # center, slm.grid is already shifted, so a resolved aperture must NOT
            # re-subtract the center. slm.aperture_mask and Aperture.resolve(slm).mask
            # must agree (and resolve(slm) returns the SLM's own aperture).
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
        import matplotlib.pyplot as plt

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
            axs = slm.plot_source(sim=False)
            plt.show()

        with subtests.test("simulated"):
            axs = slm.plot_source(sim=True)
            plt.show()

        with subtests.test("power mode"):
            axs = slm.plot_source(power=True)
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