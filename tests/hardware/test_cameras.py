"""
Unit tests for Camera base class using SimulatedCamera.
"""
import pytest
import numpy as np

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
        cam = SimulatedCamera(
            slm=slm, resolution=(512, 512), pitch_um=(5.5, 5.5), bitdepth=8
        )

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

        with subtests.test("rotation sets default_shape to (width, height)"):
            cam_rot = SimulatedCamera(
                slm=slm, resolution=(200, 100), rot="90"
            )
            # _Common always stores shape as (height, width); rotation does not
            # change self.shape but DOES change self.default_shape.
            assert cam_rot.shape == (100, 200)
            assert cam_rot.default_shape == (200, 100)
            assert cam_rot.default_shape != cam_rot.shape

        cam.close()

    def test_basic_properties(self, camera, subtests):
        """bitresolution equals 2^bitdepth and scales correctly with averaging."""

        with subtests.test("bitresolution equals 2^bitdepth"):
            assert camera.bitresolution == 2 ** camera.bitdepth

        with subtests.test("bitresolution scales with averaging"):
            orig_avg = camera.averaging
            camera.averaging = 4
            assert camera.bitresolution == (2 ** camera.bitdepth) * 4
            camera.averaging = orig_avg

    def test_get_image(self, camera, subtests):
        """get_image() returns correct shape, dtype, last_image pointer, and pixel range."""

        with subtests.test("shape matches camera.shape"):
            img = camera.get_image()
            assert img.shape == camera.shape

        with subtests.test("dtype matches camera.dtype"):
            img = camera.get_image()
            assert img.dtype == camera.dtype

        with subtests.test("last_image is the returned image"):
            img = camera.get_image()
            assert camera.last_image is img

        with subtests.test("pixel values are non-negative"):
            img = camera.get_image()
            assert np.all(img >= 0)

        with subtests.test("pixel values do not exceed bitresolution - 1"):
            img = camera.get_image()
            assert np.all(img <= camera.bitresolution - 1)

        with subtests.test("image contains nonzero signal"):
            img = camera.get_image()
            assert np.any(img > 0)

    def test_averaging_sum(self, camera, subtests):
        """get_image(averaging=N) sums N frames; pixel values scale as N, not 1."""
        saved_exposure = camera.exposure_s
        # Low exposure keeps every pixel well below saturation so each of N
        # identical (noise-free) frames contributes the same value.
        camera.set_exposure(saved_exposure * 0.05)

        try:
            with subtests.test("averaging=2 doubles single-frame pixel values"):
                img1 = camera.get_image(averaging=1)
                img2 = camera.get_image(averaging=2)
                img1_f = img1.astype(float)
                # Only evaluate on pixels that are not saturated in the single frame.
                unsaturated = img1_f < (camera.bitresolution - 1) * 0.9
                assert np.any(unsaturated), "expected unsaturated pixels at reduced exposure"
                np.testing.assert_allclose(
                    img2[unsaturated],
                    2.0 * img1_f[unsaturated],
                    rtol=1e-6,
                )

            with subtests.test("averaging=False is equivalent to averaging=1"):
                img_one = camera.get_image(averaging=1)
                img_false = camera.get_image(averaging=False)
                np.testing.assert_array_equal(img_one, img_false)

        finally:
            camera.set_exposure(saved_exposure)

    def test_get_images(self, camera, subtests):
        """get_images() returns a frame stack with correct shape, dtype, and last_image update."""
        count = 3

        with subtests.test("shape is (count, H, W)"):
            imgs = camera.get_images(count)
            assert imgs.shape == (count, camera.shape[0], camera.shape[1])

        with subtests.test("dtype matches camera.dtype"):
            imgs = camera.get_images(2)
            assert imgs.dtype == camera.dtype

        with subtests.test("last_image is the final captured frame"):
            imgs = camera.get_images(count)
            np.testing.assert_array_equal(camera.last_image, imgs[-1])

        with subtests.test("preallocated out buffer is filled with correct shape and dtype"):
            out = np.empty(
                (count, camera.default_shape[0], camera.default_shape[1]),
                dtype=camera.dtype,
            )
            imgs = camera.get_images(count, out=out, transform=False)
            assert imgs.shape == out.shape
            assert imgs.dtype == camera.dtype

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
                assert dtype_high == camera.dtype or dtype_high == float

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
        test_img = np.random.rand(10, 10) * 200
        test_imgs = np.array(
            [
                np.minimum(test_img * (2 ** i), 255)
                for i in range(3)
            ],
            dtype=np.uint8
        )

        with subtests.test("basic analysis"):
            result = Camera.get_image_hdr_analysis(
                test_imgs,
                overexposure_threshold=200
            )
            assert isinstance(result, np.ndarray)
            assert result.shape == (10, 10)
            assert result.dtype in (np.float64, np.float32)

            assert np.all(np.abs(result - test_img) < 1)

        with subtests.test("custom exposure_power list"):
            result = Camera.get_image_hdr_analysis(
                test_imgs,
                overexposure_threshold=200,
                exposure_power=[1.0, 2.0, 4.0]
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
            result1 = camera.autoexposure()
            camera.set_exposure(1)
            result2 = camera.autoexposure()
            assert pytest.approx(result1, rel=0.15) == result2

        with subtests.test("custom set_fraction"):
            camera.set_exposure(0.01)
            result3 = camera.autoexposure(set_fraction=0.3)
            assert result3 > 0

    def test_autofocus(self, camera, slm, subtests):
        """Autofocus recovers known defocus applied via Zernike."""
        slm = slm
        slm.set_source_analytic()

        fs = FourierSLM(camera, slm)
        fs.fourier_calibrate(array_pitch=10)

        defocus_zernike = 1
        slm.source["phase_sim"] = zernike(slm, 4, -defocus_zernike, use_mask=False)

        with subtests.test("recovers defocus"):
            defocus_opt = camera.autofocus(set_z=slm)
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

        x0, w_max, y0, h_max = orig_woi  # full-sensor WOI after reset

        # Determine normal vs rotated orientation once.
        # default_shape is (height, width) for normal, (width, height) for 90/270 rot.
        normal_orientation = (camera.default_shape[0] == h_max)

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
                assert x + w <= w_max, f"x+w={x+w} exceeds sensor width {w_max}"
                assert y + h <= h_max, f"y+h={y+h} exceeds sensor height {h_max}"
                assert w > 0 and h > 0, "WOI dimensions must be positive"

                # camera.shape must be consistent with WOI.
                exp_shape = expected_shape(w, h)
                assert camera.shape == exp_shape, (
                    f"camera.shape {camera.shape} != expected {exp_shape} "
                    f"for woi=({x},{w},{y},{h})"
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
            check_woi("left half",   (0, w_max // 2, 0, h_max))
            check_woi("right half",  (w_max // 2, w_max // 2, 0, h_max))
            check_woi("top half",    (0, w_max, 0, h_max // 2))
            check_woi("bottom half", (0, w_max, h_max // 2, h_max // 2))

            # Quadrant corners
            check_woi("top-left quarter",     (0,          w_max // 2, 0,          h_max // 2))
            check_woi("top-right quarter",    (w_max // 2, w_max // 2, 0,          h_max // 2))
            check_woi("bottom-left quarter",  (0,          w_max // 2, h_max // 2, h_max // 2))
            check_woi("bottom-right quarter", (w_max // 2, w_max // 2, h_max // 2, h_max // 2))

            # Centred half-size patch
            check_woi("centred half", (w_max // 4, w_max // 2, h_max // 4, h_max // 2))

            # Thin strips
            check_woi("wide strip (centre rows)",  (0, w_max, h_max * 3 // 8, h_max // 4))
            check_woi("tall strip (centre cols)",  (w_max * 3 // 8, w_max // 4, 0, h_max))

            # Small patch (~1/8 sensor), offset to several positions
            sw, sh = w_max // 8, h_max // 8
            check_woi("small patch ; near origin",        (0,               sw, 0,               sh))
            check_woi("small patch ; top-right corner",   (w_max - sw,      sw, 0,               sh))
            check_woi("small patch ; bottom-left corner", (0,               sw, h_max - sh,      sh))
            check_woi("small patch ; bottom-right corner",(w_max - sw,      sw, h_max - sh,      sh))
            check_woi("small patch ; centre",             (w_max // 2 - sw // 2, sw,
                                                           h_max // 2 - sh // 2, sh))

            # Asymmetric: very wide but short, and very tall but narrow
            check_woi("wide thin strip",  (0, w_max, h_max * 2 // 5, h_max // 5))
            check_woi("narrow tall strip",(w_max * 2 // 5, w_max // 5, 0, h_max))

            # Non-power-of-two offsets (stress-test snapping arithmetic)
            check_woi("odd offset ; 10% inset",
                      (w_max // 10, w_max * 4 // 5, h_max // 10, h_max * 4 // 5))
            check_woi("odd offset ; 30% inset",
                      (w_max * 3 // 10, w_max * 2 // 5, h_max * 3 // 10, h_max * 2 // 5))

        finally:
            # Always restore original WOI so subsequent tests see the full sensor.
            camera.set_woi(orig_woi)
            assert camera.shape == orig_shape, (
                f"Failed to restore original shape {orig_shape}; got {camera.shape}"
            )

    def test_plot(self, camera, subtests):
        """Camera.plot() renders the correct array shape and applies metadata."""
        import matplotlib.pyplot as plt

        with subtests.test("plotted array shape matches camera shape"):
            img = np.zeros(camera.shape, dtype=camera.dtype)
            ax = camera.plot(image=img, title="Shape Test")
            assert ax.get_images()[0].get_array().shape == camera.shape
            plt.close("all")

        with subtests.test("title is applied"):
            ax = camera.plot(image=np.zeros(camera.shape), title="MyTitle")
            assert ax.get_title() == "MyTitle"
            plt.close("all")

        with subtests.test("last_image rendered when image=False"):
            camera.get_image()
            stored_shape = camera.last_image.shape
            ax = camera.plot(image=False)
            assert ax.get_images()[0].get_array().shape == stored_shape
            plt.close("all")
