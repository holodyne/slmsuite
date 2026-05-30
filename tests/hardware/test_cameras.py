"""
Unit tests for Camera base class using SimulatedCamera.
"""

import numpy as np
import pytest

from slmsuite.hardware.cameras.camera import Camera
from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography.toolbox.phase import zernike


class TestCamera:
    """Tests for the Camera base class via SimulatedCamera."""

    def test_selftest(self, camera, subtests):
        """camera.test() covers core properties, dtype, exposure, capture,
        averaging, HDR, WOI, and info."""
        assert camera.test() is True

    def test_init(self, slm, subtests):
        """Verify constructor sets shape, pitch, bitdepth, and resolution convention."""
        cam = SimulatedCamera(slm=slm, resolution=(512, 512), pitch_um=(5.5, 5.5), bitdepth=8)

        with subtests.test("shape"):
            assert cam.shape == (512, 512)

        with subtests.test("pitch_um"):
            np.testing.assert_allclose(cam.pitch_um, [5.5, 5.5])

        with subtests.test("bitdepth"):
            assert cam.bitdepth == 8

        with subtests.test("bitresolution"):
            assert cam.bitresolution == 256

        with subtests.test("height-width convention"):
            height, width = 480, 640
            cam2 = SimulatedCamera(slm=slm, resolution=(width, height))
            assert cam2.shape == (height, width)

        with subtests.test("defaults"):
            cam3 = SimulatedCamera(slm=slm, resolution=(256, 256))
            assert cam3.exposure_s is not None
            assert cam3.averaging is None or isinstance(cam3.averaging, int)

        with subtests.test("rotation swaps axes"):
            cam_rot = SimulatedCamera(slm=slm, resolution=(200, 100), rot="90")
            assert cam_rot.shape == (200, 100)

        cam.close()

    def test_basic_properties(self, camera, subtests):
        """Basic properties and attributes are correctly set."""
        from slmsuite.misc.math import INTEGER_TYPES

        with subtests.test("has shape"):
            assert hasattr(camera, "shape")
            assert len(camera.shape) == 2
            assert all(isinstance(dim, INTEGER_TYPES) and dim > 0 for dim in camera.shape)

        with subtests.test("has bitresolution"):
            assert hasattr(camera, "bitresolution")
            assert camera.bitresolution == 2**camera.bitdepth

    def test_get_dtype(self, camera, subtests):
        """_get_dtype infers dtype from various get_image callables."""
        orig_dtype = camera.dtype
        orig_bitdepth = camera.bitdepth

        # (expected_dtype, fake_dtype_flag, bitdepth)
        #   fake_dtype_flag=False  -> use real capture
        #   fake_dtype_flag=None   -> raise to trigger fallback
        #   fake_dtype_flag=<type> -> return zeros of that type
        cases = [
            (orig_dtype, False, orig_bitdepth),
            (np.dtype(np.uint8), None, 8),
            (np.dtype(np.uint16), None, 12),
            (np.dtype(np.uint8), np.uint8, 8),
            (np.dtype(np.uint16), np.uint16, 12),
        ]

        try:
            for solution_dtype, fake_dtype, bitdepth in cases:
                with subtests.test(f"dtype={solution_dtype}, fake={fake_dtype}, bits={bitdepth}"):

                    def fake_get_image(_fd=fake_dtype, _sd=solution_dtype):
                        if _fd is False:
                            return camera._get_image_hw_tolerant(timeout_s=1)
                        elif _fd is None:
                            raise RuntimeError("Fake error")
                        else:
                            return np.zeros((5, 5), dtype=_sd)

                    camera.bitdepth = bitdepth
                    dtype = camera._get_dtype(fake_get_image)
                    assert dtype is solution_dtype
                    assert dtype is camera.dtype
        finally:
            camera.dtype = orig_dtype
            camera.bitdepth = orig_bitdepth

    def test_parse_averaging(self, camera, subtests):
        """_parse_averaging returns correct values and raises on bad input."""
        orig_averaging = camera.averaging

        try:
            camera.averaging = 1

            with subtests.test("preserve_none"):
                assert camera._parse_averaging(None, preserve_none=True) is None

            with subtests.test("None falls back to self.averaging"):
                assert camera._parse_averaging(None) == camera.averaging

            with subtests.test("False returns 1"):
                assert camera._parse_averaging(False) == 1

            with subtests.test("explicit int"):
                assert camera._parse_averaging(5) == 5

            with subtests.test("negative raises"):
                with pytest.raises(ValueError, match="Cannot have negative averaging"):
                    camera._parse_averaging(-1)
        finally:
            camera.averaging = orig_averaging

    def test_get_averaging_dtype(self, camera, subtests):
        """_get_averaging_dtype returns correct dtype for various averaging levels."""
        orig_averaging = camera.averaging

        try:
            with subtests.test("averaging=1 keeps dtype"):
                camera.averaging = 1
                assert camera._get_averaging_dtype(1) == camera.dtype

            with subtests.test("high averaging may promote to float"):
                camera.averaging = 1000
                dtype_high = camera._get_averaging_dtype(1000)
                assert dtype_high == camera.dtype or dtype_high is float

            with subtests.test("None with no averaging raises"):
                camera.averaging = None
                with pytest.raises(ValueError, match="Averaging is not enabled"):
                    camera._get_averaging_dtype()

            with subtests.test("negative raises"):
                with pytest.raises(ValueError, match="Cannot have negative averaging"):
                    camera._get_averaging_dtype(-1)
        finally:
            camera.averaging = orig_averaging

    def test_parse_hdr(self, camera, subtests):
        """_parse_hdr returns correct tuples for various inputs."""
        with subtests.test("preserve_none"):
            assert camera._parse_hdr(None, preserve_none=True) is None

        with subtests.test("False disables"):
            assert camera._parse_hdr(False) == (1, 0)

        with subtests.test("scalar uses base 2"):
            assert camera._parse_hdr(3) == (3, 2)

        with subtests.test("tuple passthrough"):
            assert camera._parse_hdr((4, 3)) == (4, 3)

    def test_get_image_hdr_analysis(self, subtests):
        """get_image_hdr_analysis produces correct output and validates input."""
        rng = np.random.default_rng(42)
        test_img = rng.random((10, 10)) * 200
        test_imgs = np.array([np.minimum(test_img * (2**i), 255) for i in range(3)], dtype=np.uint8)

        with subtests.test("basic analysis"):
            result = Camera.get_image_hdr_analysis(test_imgs, overexposure_threshold=200)
            assert isinstance(result, np.ndarray)
            assert result.shape == (10, 10)
            assert result.dtype in (np.float64, np.float32)

            assert np.all(np.abs(result - test_img) < 1)

        with subtests.test("custom exposure_power list"):
            result = Camera.get_image_hdr_analysis(
                test_imgs, overexposure_threshold=200, exposure_power=[1.0, 2.0, 4.0]
            )
            assert isinstance(result, np.ndarray)
            assert result.shape == (10, 10)

            assert np.all(np.abs(result - test_img) < 1)

        with subtests.test("all-zero exposure_power raises"):
            with pytest.raises(ValueError):
                Camera.get_image_hdr_analysis(test_imgs, exposure_power=[0, 0, 0])

    def test_autoexposure(self, camera, subtests):
        """Autoexposure converges to same result from different starting points."""
        with subtests.test("convergence"):
            camera.set_exposure(0.01)
            result1 = camera.autoexposure(verbose=False)
            camera.set_exposure(1)
            result2 = camera.autoexposure(verbose=False)
            assert pytest.approx(result1, rel=0.15) == result2

        with subtests.test("custom set_fraction"):
            camera.set_exposure(0.01)
            result3 = camera.autoexposure(set_fraction=0.3, verbose=False)
            assert result3 > 0

    def test_autofocus(self, camera, slm, subtests):
        """Autofocus recovers known defocus applied via Zernike."""
        slm = slm
        slm.set_source_analytic()

        fs = FourierSLM(camera, slm)
        fs.fourier_calibrate(array_pitch=10, verbose=False)

        defocus_zernike = 1
        slm.source["phase_sim"] = zernike(slm, 4, -defocus_zernike, use_mask=False)

        with subtests.test("recovers defocus"):
            defocus_opt = camera.autofocus(set_z=slm, verbose=False)
            assert pytest.approx(defocus_opt, rel=0.25) == defocus_zernike

        with subtests.test("set_z validation"):
            with pytest.raises(ValueError, match="set_z must be"):
                camera.autofocus(set_z="not_callable")

    def test_woi(self, camera, subtests):
        """
        WOI (window of interest) test: various sizes and offsets.

        For each candidate WOI the test verifies:
        - ``camera.woi`` is updated after ``set_woi``
        - ``camera.shape`` is consistent with the WOI dimensions
        - ``get_image()`` returns an array whose shape matches ``camera.shape``
        - The WOI stays within sensor bounds
        - The snapped WOI offset + size does not exceed the sensor boundary

        Cameras that do not implement ``set_woi`` are skipped.
        """
        # Skip if set_woi is not implemented.
        try:
            camera.set_woi()
        except NotImplementedError:
            pytest.skip("set_woi not implemented for this camera")

        orig_woi = camera.woi
        orig_shape = camera.shape

        _x0, w_max, _y0, h_max = orig_woi  # full-sensor WOI after reset

        # Determine normal vs rotated orientation once.
        # default_shape is (height, width) for normal, (width, height) for 90/270 rot.
        normal_orientation = camera.default_shape[0] == h_max

        def expected_shape(w, h):
            """Return numpy (rows, cols) shape for a WOI of pixel dims (w, h)."""
            return (h, w) if normal_orientation else (w, h)

        def check_woi(label, woi_request):
            """Set WOI, capture an image, and assert consistency."""
            with subtests.test(label):
                camera.set_woi(woi_request)
                x, w, y, h = camera.woi

                # WOI must stay inside sensor.
                assert x >= 0, f"OffsetX {x} < 0"
                assert y >= 0, f"OffsetY {y} < 0"
                assert x + w <= w_max, f"x+w={x + w} exceeds sensor width {w_max}"
                assert y + h <= h_max, f"y+h={y + h} exceeds sensor height {h_max}"
                assert w > 0 and h > 0, "WOI dimensions must be positive"

                # camera.shape must be consistent with WOI.
                exp_shape = expected_shape(w, h)
                assert camera.shape == exp_shape, (
                    f"camera.shape {camera.shape} != expected {exp_shape} for woi=({x},{w},{y},{h})"
                )

                # Captured image must match camera.shape.
                img = camera.get_image()
                assert img.shape == camera.shape, (
                    f"get_image() shape {img.shape} != camera.shape {camera.shape}"
                )

        try:
            # Full sensor (explicit)
            check_woi("full sensor", (0, w_max, 0, h_max))

            # Halves
            check_woi("left half", (0, w_max // 2, 0, h_max))
            check_woi("right half", (w_max // 2, w_max // 2, 0, h_max))
            check_woi("top half", (0, w_max, 0, h_max // 2))
            check_woi("bottom half", (0, w_max, h_max // 2, h_max // 2))

            # Quadrant corners
            check_woi("top-left quarter", (0, w_max // 2, 0, h_max // 2))
            check_woi("top-right quarter", (w_max // 2, w_max // 2, 0, h_max // 2))
            check_woi("bottom-left quarter", (0, w_max // 2, h_max // 2, h_max // 2))
            check_woi("bottom-right quarter", (w_max // 2, w_max // 2, h_max // 2, h_max // 2))

            # Centred half-size patch
            check_woi("centred half", (w_max // 4, w_max // 2, h_max // 4, h_max // 2))

            # Thin strips
            check_woi("wide strip (centre rows)", (0, w_max, h_max * 3 // 8, h_max // 4))
            check_woi("tall strip (centre cols)", (w_max * 3 // 8, w_max // 4, 0, h_max))

            # Small patch (~1/8 sensor), offset to several positions
            sw, sh = w_max // 8, h_max // 8
            check_woi("small patch ; near origin", (0, sw, 0, sh))
            check_woi("small patch ; top-right corner", (w_max - sw, sw, 0, sh))
            check_woi("small patch ; bottom-left corner", (0, sw, h_max - sh, sh))
            check_woi("small patch ; bottom-right corner", (w_max - sw, sw, h_max - sh, sh))
            check_woi("small patch ; centre", (w_max // 2 - sw // 2, sw, h_max // 2 - sh // 2, sh))

            # Asymmetric: very wide but short, and very tall but narrow
            check_woi("wide thin strip", (0, w_max, h_max * 2 // 5, h_max // 5))
            check_woi("narrow tall strip", (w_max * 2 // 5, w_max // 5, 0, h_max))

            # Non-power-of-two offsets (stress-test snapping arithmetic)
            check_woi(
                "odd offset ; 10% inset", (w_max // 10, w_max * 4 // 5, h_max // 10, h_max * 4 // 5)
            )
            check_woi(
                "odd offset ; 30% inset",
                (w_max * 3 // 10, w_max * 2 // 5, h_max * 3 // 10, h_max * 2 // 5),
            )

        finally:
            # Always restore original WOI so subsequent tests see the full sensor.
            camera.set_woi(orig_woi)
            assert camera.shape == orig_shape, (
                f"Failed to restore original shape {orig_shape}; got {camera.shape}"
            )

    def test_plot(self, camera, subtests):
        """Camera plot method produces an axes."""
        import matplotlib.pyplot as plt

        with subtests.test("plot with captured image"):
            ax = camera.plot()
            assert ax is not None
            plt.close("all")

        with subtests.test("plot with last_image"):
            camera.get_image()
            ax = camera.plot(image=False)
            assert ax is not None
            plt.close("all")

        with subtests.test("plot with explicit image"):
            img = np.zeros(camera.shape, dtype=camera.dtype)
            ax = camera.plot(image=img, title="Test", limits=0.5)
            assert ax is not None
            plt.close("all")
