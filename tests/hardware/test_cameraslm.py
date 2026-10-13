"""
Unit tests for FourierSLM and the calibrations mixed into it.
"""

import logging
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import (
    SIMULATED_SYSTEM_CASES,
    SIMULATED_SYSTEM_DEFAULTS,
    array_kxy,
    f_eff_from_ratio,
    farfield_corners_ij,
    farfield_support_mask,
    ground_truth_affine,
    ground_truth_kxy_to_ij,
    in_view_kxy,
    install_ground_truth_calibration,
    plot_calibration_diagnostic,
    plot_image_dim,
    seed_for,
    spot_size_ij,
    view_kxy_grid,
)
from scipy import ndimage
from scipy.optimize import OptimizeWarning

from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.cameraslms import FourierSLM, _wavefront_superpixel
from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.holography import analysis
from slmsuite.holography.algorithms import SpotHologram
from slmsuite.holography.analysis.files import load_h5, save_h5
from slmsuite.holography.toolbox import convert_vector
from slmsuite.holography.toolbox.phase import blaze, zernike, zernike_sum
from slmsuite.misc.xp import as_numpy

# Enough geometry to exercise the simulator: a contained farfield with aperture edges in
# view, a cropped one, rotation, and a 0th order steered off the sensor.
GEOMETRY_CASES = ("matched", "fov_much_larger", "rotated", "zeroth_outside")

BATTERY_ARRAY_SHAPE = 10
BATTERY_ARRAY_PITCH = 10

# Cases where fourier_calibrate with a fixed array recovers the correct affine. The
# remaining cases silently produce a *wrong* calibration (or raise).
DEFAULT_CALIBRATION_OK = {
    "identity",
    "matched",
    "fov_much_larger",
    "fov_larger",
    "mirrored",
    "pitch_anisotropic",
    "defocus",
}


# Superpixels outside the source are dark, so their center fits have no covariance.
DARK_SUPERPIXEL_FITS = pytest.mark.filterwarnings(
    "ignore:Covariance of the parameters:scipy.optimize.OptimizeWarning"
)


def _binary_grating(period, a, b, duty_cycle=0.5):
    """One period of a binary grating, as per-pixel phase in radians."""
    return np.where(np.arange(period) < round(period * duty_cycle), a, b)


def _zernike_calibrated(factory):
    """
    A wavefront Zernike calibration small enough for the fast suite: a 3x3 grid of points
    over tilt and focus, swept at five perturbations. Returns the system and the points.
    """
    fs = factory("matched")
    install_ground_truth_calibration(fs)
    points_ij = fs.kxyslm_to_ijcam(view_kxy_grid(fs, frac=0.4))

    fs.wavefront_calibrate_zernike(
        calibration_points=convert_vector(points_ij, "ij", "zernike", hardware=fs),
        zernike_indices=[2, 1, 4],
        perturbation=[-0.5, -0.25, 0.0, 0.25, 0.5],
        optimize_position=True,
        optimize_weights=False,
        plot=-1,
    )
    return (fs, points_ij)


def _ground_truth_error(fs):
    """
    Largest disagreement between the installed calibration and the ground-truth affine,
    together with the tolerance it is held to, both in camera pixels. Measured over the
    area the camera views, not along one ray through its center: a ray is blind to any
    error orthogonal to itself.
    """
    grid = view_kxy_grid(fs)
    error = float(
        np.max(np.linalg.norm(fs.kxyslm_to_ijcam(grid) - ground_truth_kxy_to_ij(fs, grid), axis=0))
    )
    return (error, max(2.0, np.max(spot_size_ij(fs))))


@pytest.fixture(scope="module")
def calibration_plot_level(request):
    """
    Plot level for the calibration battery: 2 emits the diagnostics that explain *why*
    a geometry fails, which cost more to render than the calibration costs to run.
    """
    return 2 if request.config.getoption("--save-plots") else 0


@pytest.fixture(scope="module")
def calibration_results(test_output_dir, request):
    """Accumulator for the battery, written at teardown as one montage of every case."""
    results = {}
    yield results
    _write_summary(results, test_output_dir, request)


def _write_summary(results, test_output_dir, request):
    """Montage of every case's calibration against the ground truth."""
    if not results or test_output_dir is None:
        return
    if not request.config.getoption("--save-plots"):
        return

    columns = 4
    rows = int(np.ceil(len(results) / columns))
    (fig, axs) = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axs = np.atleast_1d(axs).ravel()

    for ax, (name, result) in zip(axs, results.items()):
        plot_image_dim(ax, result["img"])
        gt = result["truth"]
        ax.scatter(gt[0], gt[1], fc="none", ec="lime", s=25, lw=0.5)
        if result["calibrated"] is not None:
            cal = result["calibrated"]
            ax.scatter(cal[0], cal[1], c="r", marker="x", s=12, lw=0.5)

        color = "green" if result["ok"] else "red"
        error = result["error"]
        ax.set_title(
            f"{name}\n" + ("FAILED" if not np.isfinite(error) else f"{error:.1f} px error"),
            color=color,
            fontsize="medium",
        )
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axs[len(results) :]:
        ax.axis("off")

    fig.suptitle(
        "fourier_calibrate vs ground truth\n"
        "circles = true spot positions, crosses = calibrated prediction"
    )
    fig.tight_layout()
    path = test_output_dir / "fourier_calibrate_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved summary: {path}")


def _run_calibration(fs, case, source, results, plot):
    """
    Calibrates, compares against ground truth, and records the diagnostics.

    Failures are returned rather than raised: the geometries that defeat a calibration
    are the ones whose plots are worth having. Returns the error in camera pixels
    (infinite if nothing was produced), the tolerance, a description, and any exception.
    """
    seed_for(case)  # The hologram starts from a random phase.
    name = f"{case}-{source or 'uniform'}"
    # Fewer spots share this dim beam, so enough of them register to place the array.
    array_shape = 8 if name == "defocus-gaussian2d" else BATTERY_ARRAY_SHAPE

    failure = None
    try:
        fs.fourier_calibrate(
            array_shape=array_shape,
            array_pitch=BATTERY_ARRAY_PITCH,
            plot=plot,
            verbose=False,
        )
    except Exception as e:
        failure = e

    (error, tolerance) = (np.inf, max(2.0, np.max(spot_size_ij(fs))))
    if failure is None:
        (error, tolerance) = _ground_truth_error(fs)

    # The array the calibration used, for the diagnostic overlays.
    array = fs.calibrations.get("fourier", {}).get("array", {})
    spots_kxy = array_kxy(
        fs,
        array.get("array_shape", array_shape),
        array.get("array_pitch", BATTERY_ARRAY_PITCH),
        array.get("array_center"),
    )
    note = (
        f"{type(failure).__name__} raised"
        if failure is not None
        else f"max error {error:.2f} px (tolerance {tolerance:.2f} px)"
    )
    img = fs.cam.last_image if fs.cam.last_image is not None else fs.cam.get_image()
    plot_calibration_diagnostic(
        fs,
        img=img,
        spots_kxy=spots_kxy,
        name=name,
        note=note,
    )

    results[name] = {
        "img": img,
        "truth": ground_truth_kxy_to_ij(fs, spots_kxy),
        "calibrated": (fs.kxyslm_to_ijcam(spots_kxy) if "fourier" in fs.calibrations else None),
        "error": error,
        "ok": error < tolerance,
    }

    return (error, tolerance, note, failure)


class TestFourierSLM:
    """Tests for public methods on FourierSLM."""

    def test_init(self, camera, slm, subtests):
        """Test FourierSLM.__init__."""

        with subtests.test("pairs the two devices, uncalibrated"):
            fs = FourierSLM(camera, slm)
            assert (fs.cam is camera) and (fs.slm is slm)
            assert fs.mag == 1.0
            assert fs.name == f"{camera.name}-{slm.name}"
            assert fs.calibrations == {}

        with subtests.test("magnification is stored"):
            assert FourierSLM(camera, slm, mag=5.0).mag == 5.0

        with subtests.test("rejects hardware of the wrong kind"):
            with pytest.raises(ValueError, match="Expected Camera"):
                FourierSLM(slm, slm)
            with pytest.raises(ValueError, match="Expected SLM"):
                FourierSLM(camera, camera)

        with subtests.test("a (width, height) shape builds a simulated pair of that resolution"):
            fs = FourierSLM((192, 128))
            assert isinstance(fs.slm, SimulatedSLM) and isinstance(fs.cam, SimulatedCamera)
            assert fs.slm.shape == fs.cam.shape == (128, 192)

        with subtests.test("a shape alongside an SLM raises"):
            with pytest.raises(ValueError, match="slm must be None"):
                FourierSLM((192, 128), slm)

    def test_fourier_calibrate(self, fourierslm, subtests):
        """Test FourierSLM.fourier_calibrate, the primary calibration of the package."""

        with subtests.test("recovers the affine the hardware was built with"):
            seed_for("fourier_calibrate")
            fourierslm.fourier_calibrate(array_pitch=30, array_shape=10, plot=False)
            (error, tolerance) = _ground_truth_error(fourierslm)
            assert error < tolerance, f"calibration is {error:.2f} px off ground truth."

        with subtests.test("a different array, given per axis, recovers the same affine"):
            seed_for("fourier_calibrate")
            fourierslm.fourier_calibrate(array_pitch=[35, 35], array_shape=[5, 5], plot=False)
            (error, tolerance) = _ground_truth_error(fourierslm)
            assert error < tolerance

        with subtests.test("carries the hardware metadata alongside the affine"):
            cal = fourierslm.calibrations["fourier"]
            assert cal["M"].shape == (2, 2) and cal["b"].shape == (2, 1)
            assert "__meta__" in cal

        with subtests.test("non-positive pitch raises"):
            with pytest.raises(ValueError, match="array_pitch must be positive"):
                fourierslm.fourier_calibrate(array_pitch=-1, array_shape=5, plot=False)

        with subtests.test("an off-center array recovers the same affine"):
            seed_for("fourier_calibrate")
            fourierslm.fourier_calibrate(
                array_pitch=30, array_shape=6, array_center=(40, -25), plot=False
            )
            (error, tolerance) = _ground_truth_error(fourierslm)
            assert error < tolerance, f"calibration is {error:.2f} px off ground truth."

        with subtests.test("an array a pitch from the camera edge recovers the same affine"):
            # This pair images 30 knm as 28 px, so the outermost spots sit 31 px from the edge.
            slm = SimulatedSLM(
                resolution=(1920, 1080), pitch_um=(8.0, 8.0), bitdepth=8, wav_um=0.78
            )
            cam = SimulatedCamera(slm, resolution=(512, 512), pitch_um=(5.5, 5.5), bitdepth=8)
            cam.set_exposure(0.1)
            edge = FourierSLM(cam, slm)
            for x in (-165, 165):
                seed_for("fourier_calibrate")
                edge.calibrations.pop("fourier", None)
                edge.fourier_calibrate(
                    array_pitch=30, array_shape=6, array_center=(x, 0), plot=False
                )
                (error, tolerance) = _ground_truth_error(edge)
                assert error < tolerance, f"calibration at x = {x} is {error:.2f} px off."

        with subtests.test("the retired autoexposure keyword warns and forwards"):
            seed_for("fourier_calibrate")
            with pytest.warns(UserWarning, match="renamed"):
                fourierslm.fourier_calibrate(
                    array_pitch=30, array_shape=6, autoexposure={"set_fraction": 0.4}, plot=False
                )
            assert fourierslm.calibrations["fourier"]["array"]["autoexpose"] == {
                "set_fraction": 0.4
            }

    @pytest.mark.slow
    def test_fourier_calibrate_noise(self, slm, subtests):
        """Camera noise and a rotated sensor still calibrate to the ground truth."""
        noise = {
            "dark": lambda img: np.random.normal(0.005 * img, 0.002 * img),
            "read": lambda img: np.random.poisson(0.03 * img),
        }
        for theta, nz in [(0.0, None), (0.0, noise), (0.2, noise), (-0.3, noise)]:
            with subtests.test(theta=theta, noisy=nz is not None):
                seed_for("fourier_calibrate_noise")
                cam = SimulatedCamera(slm, resolution=(512, 512), pitch_um=(5.5, 5.5), noise=nz)
                cam.set_affine(f_eff=170000.0, units="norm", theta=theta)
                fs = FourierSLM(cam, slm, mag=1.0)
                fs.cam.set_exposure(0.1)
                fs.fourier_calibrate(array_pitch=30, array_shape=10, plot=False)
                (error, tolerance) = _ground_truth_error(fs)
                assert error < tolerance, f"calibration is {error:.2f} px off ground truth."

    def test_fourier_calibrate_analytic(self, fourierslm, subtests):
        """Test FourierSLM.fourier_calibrate_analytic."""
        (M, b) = ground_truth_affine(fourierslm)

        with subtests.test("the supplied affine is the one the system then uses"):
            fourierslm.fourier_calibrate_analytic(M, b)
            assert _ground_truth_error(fourierslm)[0] == pytest.approx(0, abs=1e-9)

        with subtests.test("a camera with no affine of its own adopts it"):
            assert np.allclose(fourierslm.cam.M, M)
            assert np.allclose(np.squeeze(fourierslm.cam.b), np.squeeze(b))

        with subtests.test("wrong-shape M raises"):
            with pytest.raises(ValueError, match="Expected a 2x2 matrix"):
                fourierslm.fourier_calibrate_analytic(np.eye(3), b)

    def test_fourier_calibration_build(self, fourierslm, subtests):
        """Test FourierSLM.fourier_calibration_build."""
        with subtests.test("the offset defaults to the center of the camera image"):
            (_, b) = fourierslm.fourier_calibration_build(1000)
            assert np.allclose(np.ravel(b), np.flip(fourierslm.cam.shape) / 2)

    def test_fourier_grid_project(self, fourierslm_calibrated, caplog, subtests):
        """Test FourierSLM.fourier_grid_project."""
        hologram = fourierslm_calibrated.fourier_grid_project(array_shape=3, array_pitch=35)
        img = fourierslm_calibrated.cam.get_image()

        with subtests.test("two spots are omitted as an orientation check"):
            assert hologram.spot_kxy_rounded.shape == (2, 3 * 3 - 2)

        with subtests.test("every projected spot is among the brightest pixels"):
            ij = np.rint(hologram.spot_ij).astype(int)
            assert np.min(img[ij[1], ij[0]]) > 0.5 * np.max(img)

        with subtests.test("a non-integer pitch and an unknown optimize argument are flagged"):
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                caplog.clear()
                fourierslm_calibrated.fourier_grid_project(
                    array_shape=3, array_pitch=35.5, maxiter=2, mexiter=5
                )
            assert "array_pitch is non-integer" in caplog.text
            assert "Unexpected argument 'mexiter'" in caplog.text

    def test_kxyslm_to_ijcam(self, simulated_system_factory, subtests):
        """Test FourierSLM.kxyslm_to_ijcam."""
        fs = simulated_system_factory("sheared")
        install_ground_truth_calibration(fs)
        grid = view_kxy_grid(fs)

        with subtests.test("maps kxy where the hardware places it"):
            assert np.allclose(fs.kxyslm_to_ijcam(grid), ground_truth_kxy_to_ij(fs, grid))

        with subtests.test("a single vector returns a (2, 1) column"):
            assert fs.kxyslm_to_ijcam([0, 0]).shape == (2, 1)

        with subtests.test("a window of interest shifts the prediction by its offset"):
            (x0, y0) = (12, 7)
            before = fs.kxyslm_to_ijcam(grid)
            try:
                fs.cam.set_woi((x0, 64, y0, 48))
                assert np.allclose(fs.kxyslm_to_ijcam(grid), before - [[x0], [y0]])
            finally:
                fs.cam.set_woi(None)

        with subtests.test("raises without a calibration"):
            with pytest.raises((KeyError, RuntimeError)):
                FourierSLM(fs.cam, fs.slm).kxyslm_to_ijcam([10.0, 20.0])

        with subtests.test("a third row carries depth linearly, leaving x and y alone"):
            kxyz = np.array([[0.01, -0.02, 0.004], [0.005, 0.01, -0.01], [1e-6, -3e-6, 2e-6]])
            ijz = fs.kxyslm_to_ijcam(kxyz)
            assert ijz.shape == (3, 3)
            np.testing.assert_allclose(ijz[:2], fs.kxyslm_to_ijcam(kxyz[:2]))
            np.testing.assert_allclose(ijz[2] / kxyz[2], ijz[2, 0] / kxyz[2, 0])

        with subtests.test("depth is the focal shift F^2 / f of a lens of focal power wav / f"):
            (power, pitch_um) = (1e-6, np.mean(fs.cam.pitch_um))
            f_um = np.mean(fs.get_effective_focal_length(units="um"))
            # Newton's lens equation: a lens of focal length f shifts the focus of F by F^2 / f.
            shift_um = f_um**2 / (fs.slm.wav_um / power)
            depth_ij = fs.kxyslm_to_ijcam([[0.0], [0.0], [power]])[2, 0]
            assert np.abs(depth_ij) == pytest.approx(shift_um / pitch_um)

        with subtests.test("depth needs a camera pitch, and is nan without one"):
            fs.cam.pitch_um = None
            assert np.all(np.isnan(fs.kxyslm_to_ijcam([[0.01], [0.0], [1e-6]])[2]))

    def test_ijcam_to_kxyslm(self, simulated_system_factory, subtests):
        """Test FourierSLM.ijcam_to_kxyslm."""
        fs = simulated_system_factory("sheared")
        install_ground_truth_calibration(fs)
        (h, w) = fs.cam.shape
        ij = np.array([[0, w / 2, w - 1], [0, h / 2, h - 1]], dtype=float)

        with subtests.test("inverts the ground-truth placement of the camera"):
            kxy = fs.ijcam_to_kxyslm(ij)
            assert np.allclose(ground_truth_kxy_to_ij(fs, kxy), ij)

        with subtests.test("is the exact inverse of kxyslm_to_ijcam"):
            assert np.allclose(fs.kxyslm_to_ijcam(fs.ijcam_to_kxyslm(ij)), ij, atol=1e-10)
            grid = view_kxy_grid(fs)
            assert np.allclose(fs.ijcam_to_kxyslm(fs.kxyslm_to_ijcam(grid)), grid, atol=1e-10)

        with subtests.test("inverts the depth row too"):
            kxyz = np.array([[0.01, -0.02], [0.005, 0.01], [1e-6, -3e-6]])
            np.testing.assert_allclose(fs.ijcam_to_kxyslm(fs.kxyslm_to_ijcam(kxyz)), kxyz)

    def test_check_fourier_calibration_stale(self, simulated_system_factory, subtests):
        """A wavefront calibration newer than the Fourier one warns on every conversion."""
        fs = simulated_system_factory("matched")

        with subtests.test("no Fourier calibration raises"):
            with pytest.raises(RuntimeError, match="must exist"):
                fs.ijcam_to_kxyslm([10.0, 20.0])

        fourier = install_ground_truth_calibration(fs)
        wavefront = dict(fs._get_calibration_metadata())

        with subtests.test("a newer wavefront calibration warns"):
            wavefront["__timestamp__"] = fourier["__timestamp__"] + 1
            fs.calibrations["wavefront_superpixel"] = wavefront
            for convert in (fs.ijcam_to_kxyslm, fs.kxyslm_to_ijcam):
                with pytest.warns(UserWarning, match="Fourier calibration may be stale"):
                    convert([10.0, 20.0])

        with subtests.test("an older wavefront calibration is silent"):
            wavefront["__timestamp__"] = fourier["__timestamp__"] - 1
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                fs.ijcam_to_kxyslm([10.0, 20.0])

        with subtests.test("a wavefront calibration without a timestamp is silent"):
            fs.calibrations["wavefront_superpixel"] = {}
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                fs.ijcam_to_kxyslm([10.0, 20.0])

    def test_zeroth_order_ij(self, simulated_system_factory, subtests):
        """Test the FourierSLM.zeroth_order_ij property."""
        fs = simulated_system_factory("offset")
        install_ground_truth_calibration(fs)
        b = np.copy(fs.calibrations["fourier"]["b"])

        with subtests.test("is kxyslm_to_ijcam at the origin"):
            np.testing.assert_allclose(fs.zeroth_order_ij, fs.kxyslm_to_ijcam([0, 0]))

        with subtests.test("follows a window of interest while the raw offset stays put"):
            (x0, y0) = (12, 7)
            fs.cam.set_woi((x0, 64, y0, 48))
            np.testing.assert_allclose(fs.zeroth_order_ij, b - [[x0], [y0]])
            np.testing.assert_array_equal(fs.calibrations["fourier"]["b"], b)

    def test_fourier_affine(self, fourierslm_calibrated, subtests):
        """Test the FourierSLM.fourier_affine property."""
        affine = fourierslm_calibrated.fourier_affine
        kxy = np.array([[10.0], [20.0]])
        ij = np.array([[150.0], [200.0]])

        with subtests.test("applying it is kxyslm_to_ijcam"):
            assert isinstance(affine, analysis.Affine)
            assert np.allclose(affine @ kxy, fourierslm_calibrated.kxyslm_to_ijcam(kxy))

        with subtests.test("applying its inverse is ijcam_to_kxyslm"):
            assert np.allclose(affine.inv @ ij, fourierslm_calibrated.ijcam_to_kxyslm(ij))

    def test_get_farfield_spot_size(self, simulated_system_factory, subtests):
        """Test FourierSLM.get_farfield_spot_size."""
        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)
        aperture = np.flip(np.squeeze(fs.slm.shape)) * np.squeeze(fs.slm.pitch)

        with subtests.test("the whole aperture gives the diffraction-limited spot"):
            size_kxy = fs.get_farfield_spot_size(aperture, basis="kxy")
            size_ij = fs.get_farfield_spot_size(aperture, basis="ij")
            assert np.allclose(size_ij, spot_size_ij(fs))
            # This camera is isotropic, so the bases differ only by its affine.
            assert np.allclose(size_ij, np.abs(ground_truth_affine(fs)[0]) @ size_kxy)

        with subtests.test("ij spot size is measured in the frame the user sees"):
            # The de-rotation reduces algebraically to
            #   size_ij == sqrt(|det(fourier_affine.M)|) * (1/Wx, 1/Wy)
            # for any WOI, binning, or orientation. An anisotropic aperture plus a
            # non-trivial orientation is what distinguishes this from the raw affine.
            (Wx, Wy) = (1.0, 2.0)
            try:
                for rot, binning, woi in [
                    ("0", 1, None),
                    ("90", 1, None),
                    ("180", 2, None),
                    ("0", 2, (20, 60, 30, 50)),
                ]:
                    fs.cam.transform = analysis.get_orientation_transformation(rot)
                    fs.cam.set_binning(binning)
                    fs.cam.set_woi(woi)

                    size = fs.get_farfield_spot_size((Wx, Wy), basis="ij")
                    expected = np.sqrt(np.abs(fs.fourier_affine.det())) * np.array([1 / Wx, 1 / Wy])
                    assert np.allclose(size, expected), (
                        f"rot={rot} binning={binning} woi={woi}: {size} != {expected}"
                    )
            finally:
                fs.cam.transform = analysis.get_orientation_transformation("0")
                fs.cam.set_binning(1)
                fs.cam.set_woi(None)

        with subtests.test("bad basis raises"):
            with pytest.raises(ValueError, match="Unrecognized basis"):
                fs.get_farfield_spot_size(slm_size=1.0, basis="badvalue")

    def test_get_effective_focal_length(self, simulated_system_factory, subtests):
        """Test FourierSLM.get_effective_focal_length."""
        defaults = SIMULATED_SYSTEM_DEFAULTS

        for case in ("matched", "fov_larger"):
            with subtests.test(f"{case}: recovers the focal length the case was built at"):
                fs = simulated_system_factory(case)
                install_ground_truth_calibration(fs)
                truth = f_eff_from_ratio(
                    SIMULATED_SYSTEM_CASES[case]["ratio"],
                    defaults["cam_resolution"][0],
                    defaults["cam_pitch_um"][0],
                    defaults["slm_pitch_um"][0],
                    defaults["wav_um"],
                )
                assert np.allclose(fs.get_effective_focal_length(units="norm"), truth)
                assert fs.get_effective_focal_length(units="ij") == pytest.approx(
                    truth * defaults["wav_um"] / defaults["cam_pitch_um"][0]
                )

        with subtests.test("raises without a calibration"):
            with pytest.raises(RuntimeError):
                simulated_system_factory("matched").get_effective_focal_length()

        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)
        f_norm = fs.get_effective_focal_length(units="norm")

        with subtests.test("metric units are the normalized length in wavelengths"):
            np.testing.assert_allclose(
                fs.get_effective_focal_length(units="um"), f_norm * fs.slm.wav_um
            )
            np.testing.assert_allclose(
                fs.get_effective_focal_length(units="mm"), f_norm * fs.slm.wav_um / 1000
            )

        with subtests.test("an unknown unit raises"):
            with pytest.raises(ValueError, match="not recognized"):
                fs.get_effective_focal_length(units="furlong")

        with subtests.test("without a camera pitch only pixel units are defined"):
            f_ij = fs.get_effective_focal_length(units="ij")
            fs.cam.pitch_um = None
            assert np.isnan(fs.get_effective_focal_length(units="norm"))
            assert fs.get_effective_focal_length(units="ij") == f_ij

    @pytest.mark.parametrize("name", GEOMETRY_CASES)
    def test_get_farfield_extent(self, simulated_system_factory, name, subtests):
        """Test FourierSLM.get_farfield_extent against the ground-truth farfield square."""
        fs = simulated_system_factory(name)
        install_ground_truth_calibration(fs)

        with subtests.test("corners close the polygon"):
            corners = fs.get_farfield_extent(return_mask=False)
            assert corners.shape == (2, 5)
            assert np.allclose(corners[:, 0], corners[:, 4])

        with subtests.test("the mask agrees with the ground-truth farfield polygon"):
            canvas = np.zeros(fs.cam.shape, np.uint8)
            cv2.fillConvexPoly(canvas, np.rint(farfield_corners_ij(fs).T).astype(np.int32), 255)
            agreement = (fs.get_farfield_extent(return_mask=True) == (canvas > 128)).mean()
            assert agreement > 0.97, (
                f"disagrees with the ground truth on {(1 - agreement) * 100:.1f}% of pixels."
            )

        with subtests.test("a margin without inscribe raises"):
            with pytest.raises(ValueError, match="margin requires inscribe"):
                fs.get_farfield_extent(margin=3)

    def test_get_camera_extent(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.get_camera_extent."""
        (h, w) = fourierslm_calibrated.cam.shape

        with subtests.test("kxy corners map back onto the camera corners"):
            corners = fourierslm_calibrated.get_camera_extent(units="kxy", return_mask=False)
            assert corners.shape == (2, 5)
            assert np.allclose(
                fourierslm_calibrated.kxyslm_to_ijcam(corners),
                [[0, w - 1, w - 1, 0, 0], [0, 0, h - 1, h - 1, 0]],
            )

        with subtests.test("the knm mask covers the camera's share of the canvas"):
            shape = SpotHologram.get_padded_shape(
                fourierslm_calibrated, padding_order=1, square_padding=True
            )
            mask = fourierslm_calibrated.get_camera_extent(units=shape, return_mask=True)
            assert mask.dtype == bool and mask.shape == shape
            assert mask.any()

        with subtests.test("a mask in a string basis raises"):
            with pytest.raises(ValueError, match="return_mask must be False"):
                fourierslm_calibrated.get_camera_extent(units="kxy", return_mask=True)

    def test_simulate(self, fourierslm_calibrated, simulated_system_factory, subtests):
        """Test FourierSLM.simulate."""

        with subtests.test("the clone displays what the hardware displays at any phase_scaling"):
            for phase_scaling in (1, 0.75, 1.3):
                fs = simulated_system_factory("matched")
                fs.slm.wav_design_um = fs.slm.wav_um / phase_scaling
                install_ground_truth_calibration(fs)
                phase = np.random.default_rng(0).uniform(0, 2 * np.pi, fs.slm.shape)
                fs.slm.set_phase(phase, phase_correct=False)
                np.testing.assert_array_equal(
                    as_numpy(fs.simulate().slm.display),
                    as_numpy(fs.slm.display),
                    err_msg=f"{phase_scaling=}",
                )

        with subtests.test("the clone displays what the hardware displays after an integer write"):
            # Below unit phase_scaling, an integer write can re-display as an alias level.
            for phase_scaling in (1, 1.3):
                fs = simulated_system_factory("matched")
                fs.slm.wav_design_um = fs.slm.wav_um / phase_scaling
                install_ground_truth_calibration(fs)
                levels = np.random.default_rng(0).integers(0, fs.slm.bitresolution, fs.slm.shape)
                fs.slm.set_phase(levels.astype(fs.slm.dtype), phase_correct=False)
                np.testing.assert_array_equal(
                    as_numpy(fs.simulate().slm.display), levels, err_msg=f"{phase_scaling=}"
                )

        with subtests.test("the simulated twin transforms coordinates identically"):
            fs_sim = fourierslm_calibrated.simulate()
            assert isinstance(fs_sim, FourierSLM)
            assert isinstance(fs_sim.slm, SimulatedSLM)
            assert isinstance(fs_sim.cam, SimulatedCamera)
            kxy = np.array([10.0, 15.0])
            assert np.allclose(
                fs_sim.kxyslm_to_ijcam(kxy), fourierslm_calibrated.kxyslm_to_ijcam(kxy)
            )

        with subtests.test("raises without a calibration"):
            with pytest.raises(ValueError, match="Cannot simulate"):
                FourierSLM(fourierslm_calibrated.cam, fourierslm_calibrated.slm).simulate()

    def test_name_calibration(self, fourierslm, subtests):
        """Test FourierSLM.name_calibration."""
        for calibration_type in ("fourier", "wavefront"):
            with subtests.test(f"names the {calibration_type} calibration after its type"):
                assert calibration_type in fourierslm.name_calibration(calibration_type).lower()

    def test_save_load_calibration(self, fourierslm_calibrated, temp_dir, caplog, subtests):
        """Test FourierSLM.save_calibration and FourierSLM.load_calibration."""
        path = fourierslm_calibrated.save_calibration("fourier", path=temp_dir, name="test_save")

        with subtests.test("the file round-trips the affine"):
            assert os.path.exists(path)
            fs_new = FourierSLM(fourierslm_calibrated.cam, fourierslm_calibrated.slm)
            fs_new.load_calibration("fourier", file_path=path)
            for key in ("M", "b", "a"):
                assert np.allclose(
                    fs_new.calibrations["fourier"][key],
                    fourierslm_calibrated.calibrations["fourier"][key],
                )

        with subtests.test("a default-named file loads by its path alone, under the type it names"):
            default = fourierslm_calibrated.save_calibration("fourier", path=temp_dir)
            fs_new = FourierSLM(fourierslm_calibrated.cam, fourierslm_calibrated.slm)
            assert fs_new.load_calibration(default) == default
            np.testing.assert_allclose(
                fs_new.calibrations["fourier"]["M"],
                fourierslm_calibrated.calibrations["fourier"]["M"],
            )

        with subtests.test("a path whose name carries no type raises"):
            fs_new = FourierSLM(fourierslm_calibrated.cam, fourierslm_calibrated.slm)
            with pytest.raises(ValueError, match="parse calibration type"):
                fs_new.load_calibration(path)

        with subtests.test("saving a calibration that was never taken raises"):
            with pytest.raises(ValueError, match="Could not find calibration"):
                fourierslm_calibrated.save_calibration("nonexistent", path=temp_dir)

        with subtests.test("without a path, the newest default-named file here loads"):
            cwd = os.getcwd()
            fs_new = FourierSLM(fourierslm_calibrated.cam, fourierslm_calibrated.slm)
            try:
                os.chdir(temp_dir)
                with pytest.raises(FileNotFoundError, match="Unable to find"):
                    fs_new.load_calibration("settle")
                newest = fourierslm_calibrated.save_calibration("fourier")
                assert os.path.samefile(fs_new.load_calibration("fourier"), newest)
            finally:
                os.chdir(cwd)

        with subtests.test("a file from another release is flagged"):
            data = load_h5(path)
            data["__version__"] = "0.0.0"
            old = os.path.join(temp_dir, "test_old_version.h5")
            save_h5(old, data)
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                caplog.clear()
                FourierSLM(fourierslm_calibrated.cam, fourierslm_calibrated.slm).load_calibration(
                    "fourier", file_path=old
                )
            assert "was created in 0.0.0" in caplog.text

        with subtests.test("the retired write_calibration and read_calibration warn and forward"):
            with pytest.warns(UserWarning, match="save_calibration"):
                fourierslm_calibrated.write_calibration("fourier", temp_dir, "test_alias")
            (alias,) = [f for f in os.listdir(temp_dir) if f.startswith("test_alias")]
            fs_new = FourierSLM(fourierslm_calibrated.cam, fourierslm_calibrated.slm)
            with pytest.warns(UserWarning, match="load_calibration"):
                loaded = fs_new.read_calibration("fourier", os.path.join(temp_dir, alias))
            assert loaded == os.path.join(temp_dir, alias)
            np.testing.assert_array_equal(
                fs_new.calibrations["fourier"]["M"],
                fourierslm_calibrated.calibrations["fourier"]["M"],
            )

    def test_load(
        self, fourierslm_calibrated, simulated_system_factory, temp_dir, caplog, subtests
    ):
        """Test the FourierSLM.load static constructor."""
        path = fourierslm_calibrated.save_calibration(
            "fourier", path=temp_dir, name="test_static_load"
        )
        fs = FourierSLM.load(path)

        with subtests.test("rebuilds simulated hardware from the stored metadata"):
            assert isinstance(fs, FourierSLM)
            assert isinstance(fs.slm, SimulatedSLM)
            assert isinstance(fs.cam, SimulatedCamera)

        with subtests.test("the rebuilt system, and its simulation, map like the original"):
            fs.load_calibration("fourier", file_path=path)
            kxy = np.array([10.0, 15.0])
            assert np.allclose(fs.kxyslm_to_ijcam(kxy), fourierslm_calibrated.kxyslm_to_ijcam(kxy))
            assert np.allclose(fs.simulate().kxyslm_to_ijcam(kxy), fs.kxyslm_to_ijcam(kxy))

        with subtests.test("a saved binning is reported, not reapplied"):
            binned_fs = simulated_system_factory("matched")
            binned_fs.cam.set_binning((2, 2))
            install_ground_truth_calibration(binned_fs)
            binned = binned_fs.save_calibration(
                "fourier", path=temp_dir, name="test_static_load_binned"
            )
            with caplog.at_level(logging.WARNING):
                FourierSLM.load(binned)
            assert "does not reapply binning" in caplog.text

        with subtests.test("a file without the hardware metadata raises"):
            bare = os.path.join(temp_dir, "test_static_load_bare.h5")
            save_h5(bare, {"M": np.eye(2)})
            with pytest.raises(ValueError, match="without field '__meta__'"):
                FourierSLM.load(bare)

            data = load_h5(path)
            del data["__meta__"]["cam"]
            camless = os.path.join(temp_dir, "test_static_load_camless.h5")
            save_h5(camless, data)
            with pytest.raises(ValueError, match="without metadata field 'cam'"):
                FourierSLM.load(camless)

    def test_check_calibration_frame(self, simulated_system_factory, caplog, subtests):
        """A calibration stated in camera pixels checks the frame it was taken in."""
        fs = simulated_system_factory("matched")
        fs.calibrations["wavefront_zernike"] = dict(fs._get_calibration_metadata())

        with subtests.test("the frame it was taken in passes"):
            assert fs._check_calibration_frame("wavefront_zernike", raise_error=True)

        with subtests.test("an unrecorded frame passes"):
            assert fs._check_calibration_frame("nonexistent", raise_error=True)

        with subtests.test("a changed window warns, or raises on request"):
            try:
                fs.cam.set_woi((10, 64, 20, 48))
                with caplog.at_level(logging.WARNING, logger="slmsuite"):
                    caplog.clear()
                    assert not fs._check_calibration_frame("wavefront_zernike")
                assert "does not follow the window" in caplog.text
                with pytest.raises(RuntimeError, match="does not follow the window"):
                    fs._check_calibration_frame("wavefront_zernike", raise_error=True)
            finally:
                fs.cam.set_woi(None)

    def test_plot(self, fourierslm, subtests):
        """Test FourierSLM.plot, which shows the nearfield beside the farfield."""
        with subtests.test("plots a phase and an image on a new pair of axes"):
            assert len(fourierslm.plot(phase=blaze(fourierslm.slm, vector=(1e-3, 2e-3)))) == 2
            assert len(fourierslm.plot(image=fourierslm.cam.get_image())) == 2
        plt.close("all")

        with subtests.test("draws on the axes it is given, which must be a pair"):
            (_, axs) = plt.subplots(1, 3)
            image = fourierslm.cam.get_image()
            drawn = fourierslm.plot(image=image, axs=axs[:2])
            assert (drawn[0] is axs[0]) and (drawn[1] is axs[1])
            assert axs[0].images and axs[1].images and not axs[2].images
            with pytest.raises(ValueError, match="tuple of two axes"):
                fourierslm.plot(image=image, axs=axs)
            plt.close("all")

    @pytest.mark.slow
    @DARK_SUPERPIXEL_FITS
    def test_wavefront_calibrate_superpixel(
        self, fourierslm_calibrated, simulated_system_factory, subtests
    ):
        """Test FourierSLM.wavefront_calibrate_superpixel against a planted aberration."""
        fs = fourierslm_calibrated
        cal_point = [150, 150]
        sp_size = fs.slm.shape[0] // 6

        # An aberrated source, so that there is something to measure.
        fs.slm.set_source_analytic(
            phase_offset=zernike_sum(
                fs.slm,
                indices=(3, 4, 5, 7, 8),
                weights=(1, -2, 3, 1, 1),
                aperture=None,
                use_mask=False,
            ),
            sim=True,
        )
        fs.cam.set_exposure(0.1)

        with subtests.test("interference happens where the blaze lands"):
            fs.slm.set_phase(blaze(fs.slm, vector=fs.ijcam_to_kxyslm(cal_point)))
            img = fs.cam.get_image()
            assert img[140:160, 140:160].mean() > img.mean()

        for phase_steps, name in [(None, "amplitude only"), (1, "one shot"), (5, "many shot")]:
            fs.slm.source["phase"] = None  # Clear any previous calibration.

            with subtests.test(f"{name}: measures the power at every superpixel"):
                result = fs.wavefront_calibrate_superpixel(
                    calibration_points=cal_point,
                    superpixel_size=sp_size,
                    phase_steps=phase_steps,
                    plot=-1,
                )
                assert "power" in result
                cal = fs.calibrations["wavefront_superpixel"]
                for key in ("superpixel_size", "scheduling", "slm_supershape"):
                    assert key in cal

            # Phase is hard to verify at this resolution; amplitude is not.
            with subtests.test(f"{name}: the measured amplitude matches the source"):
                fs.wavefront_calibration_superpixel_process(smooth=False)
                amplitude = fs.slm.source["amplitude"]
                simulated = fs.slm.source["amplitude_sim"]
                error = np.sum(np.abs(amplitude - simulated)) / np.sum(simulated)
                logging.getLogger("conftest").info(
                    "Normalized amplitude difference (%s): %.2f", name, error
                )
                assert error < 0.5, f"amplitude is {error:.2f} off the simulated source."

        # Circular standard deviation, blind to the piston a correction leaves free.
        def spread(phase):
            return np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * phase)))))

        for case in ("rotated", "anisotropic"):
            residual = {}
            for multiplier in (4, 6):
                fs_case = simulated_system_factory(
                    case,
                    slm_resolution=(256, 256),
                    cam_resolution=(512, 512),
                    aberration=((3, 4, 5), (1.5, -2.0, 1.0)),
                )
                install_ground_truth_calibration(fs_case)
                fs_case.cam.set_exposure(1)
                fs_case._wavefront_calibration_window_multiplier = multiplier
                aberration = as_numpy(fs_case.slm.source["phase_sim"])
                fs_case.wavefront_calibrate_superpixel(
                    calibration_points=[[368], [138]],
                    superpixel_size=32,
                    phase_steps=1,
                    plot=-1,
                )
                correction = as_numpy(
                    fs_case.wavefront_calibration_superpixel_process(
                        smooth=False,
                        remove_blaze=False,
                        apply=False,
                        r2_threshold=0.5,
                    )["phase"]
                )
                residual[multiplier] = spread(correction + aberration)

            with subtests.test(f"one shot on the {case} camera corrects the planted aberration"):
                assert max(residual.values()) < 0.1 * spread(aberration), residual

            with subtests.test(f"one shot on the {case} camera does not depend on the window"):
                assert residual[6] == pytest.approx(residual[4], abs=0.01)

        with subtests.test("only fresh_calibration=False keeps the source phase, as a host array"):
            fs_case = simulated_system_factory(
                "matched", slm_resolution=(128, 128), cam_resolution=(256, 256)
            )
            install_ground_truth_calibration(fs_case)
            phase = fs_case.slm.xp.full(fs_case.slm.shape, 0.5)

            stored = []
            for fresh_calibration in (True, False):
                fs_case.slm.source["phase"] = phase
                fs_case.wavefront_calibrate_superpixel(
                    calibration_points=[[190], [170]],
                    superpixel_size=32,
                    phase_steps=1,
                    fresh_calibration=fresh_calibration,
                    plot=-1,
                )
                stored.append(
                    fs_case.calibrations["wavefront_superpixel"]["previous_phase_correction"]
                )

            assert stored[0] is False
            assert isinstance(stored[1], np.ndarray)
            np.testing.assert_array_equal(stored[1], as_numpy(phase))

        with subtests.test("test_index measures one point and restores the raw source"):
            fs.slm.set_aperture(radius=0.3, units="frac")
            phase = fs.slm.xp.ones(fs.slm.shape)
            fs.slm.source["phase"] = phase
            fs.slm.source.pop("amplitude", None)
            keys = {k for (k, v) in fs.slm.source.items() if v is not None}
            result = fs.wavefront_calibrate_superpixel(
                calibration_points=cal_point,
                superpixel_size=sp_size,
                phase_steps=5,
                test_index=-2,
                plot=-1,
            )
            assert "power" in result
            assert {k for (k, v) in fs.slm.source.items() if v is not None} == keys
            assert np.array_equal(as_numpy(fs.slm.source["phase"]), as_numpy(phase))

        with subtests.test("requires a Fourier calibration"):
            with pytest.raises((RuntimeError, KeyError)):
                FourierSLM(fs.cam, fs.slm).wavefront_calibrate_superpixel(
                    calibration_points=cal_point,
                    superpixel_size=sp_size,
                    plot=-1,
                )

    # Incidental here; test_check_fourier_calibration_stale asserts this warning.
    @pytest.mark.filterwarnings("ignore:The wavefront calibration is newer:UserWarning")
    @DARK_SUPERPIXEL_FITS
    def test_wavefront_calibrate_superpixel_abort(
        self, fourierslm_calibrated, simulated_system_factory, monkeypatch, subtests
    ):
        """An interrupt inside a fit escapes the calibration and leaves the source intact."""
        fs = fourierslm_calibrated
        cal_point = [150, 150]
        sp_size = fs.slm.shape[0] // 6

        fs.slm.set_source_analytic(
            phase_offset=zernike_sum(
                fs.slm,
                indices=(3, 4, 5, 7, 8),
                weights=(1, -2, 3, 1, 1),
                aperture=None,
                use_mask=False,
            ),
            sim=True,
        )
        fs.cam.set_exposure(0.1)

        # A measured source for the aborted run to strip.
        fs.wavefront_calibrate_superpixel(
            calibration_points=cal_point,
            superpixel_size=sp_size,
            phase_steps=5,
            plot=-1,
        )
        fs.wavefront_calibration_superpixel_process(smooth=False)
        before_amplitude = as_numpy(fs.slm.source["amplitude"]).copy()
        before_phase = as_numpy(fs.slm.source["phase"]).copy()

        curve_fit = _wavefront_superpixel.optimize.curve_fit
        state = {"fits": 0}

        def interrupt_on_third_fit(*args, **kwargs):
            state["fits"] += 1
            if state["fits"] == 3:
                raise KeyboardInterrupt("simulated Ctrl-C inside a fit")
            return curve_fit(*args, **kwargs)

        monkeypatch.setattr(_wavefront_superpixel.optimize, "curve_fit", interrupt_on_third_fit)

        with subtests.test("the interrupt is not swallowed by the fit guard"):
            with pytest.raises(KeyboardInterrupt):
                fs.wavefront_calibrate_superpixel(
                    calibration_points=cal_point,
                    superpixel_size=sp_size,
                    phase_steps=5,
                    plot=-1,
                )

        with subtests.test("the stripped source is reinstated"):
            assert np.allclose(
                as_numpy(fs.slm.source["amplitude"]), before_amplitude, equal_nan=True
            )
            assert np.allclose(as_numpy(fs.slm.source["phase"]), before_phase, equal_nan=True)

        with subtests.test("the slm still projects after the abort"):
            fs.slm.set_phase(blaze(fs.slm, vector=fs.ijcam_to_kxyslm(cal_point)))
            assert fs.cam.get_image().mean() > 0

        with subtests.test("the raw source is reinstated, not its aperture-masked view"):
            fs = simulated_system_factory(
                "matched", slm_resolution=(128, 128), cam_resolution=(256, 256)
            )
            install_ground_truth_calibration(fs)
            fs.slm.set_aperture(radius=0.3, units="frac")
            phase = fs.slm.xp.ones(fs.slm.shape)
            fs.slm.source["phase"] = phase
            fs.slm.source.pop("amplitude", None)
            keys = {k for (k, v) in fs.slm.source.items() if v is not None}

            state["fits"] = 0
            with pytest.raises(KeyboardInterrupt):
                fs.wavefront_calibrate_superpixel(
                    calibration_points=[[190], [170]],
                    superpixel_size=16,
                    phase_steps=5,
                    plot=-1,
                )
            assert {k for (k, v) in fs.slm.source.items() if v is not None} == keys
            assert np.array_equal(as_numpy(fs.slm.source["phase"]), as_numpy(phase))

    @DARK_SUPERPIXEL_FITS
    def test_wavefront_calibrate_superpixel_scheduling(self, simulated_system_factory):
        """
        No calibration point measures at a superpixel that another point is using as its
        reference, in any measurement where that other point is also measuring.
        """
        fs = simulated_system_factory(
            "matched", slm_resolution=(128, 128), cam_resolution=(256, 256)
        )
        install_ground_truth_calibration(fs)
        fs.wavefront_calibrate_superpixel(
            calibration_points=np.array([[60, 190, 128], [60, 60, 190]]),
            superpixel_size=32,
            phase_steps=1,
            reference_superpixels=np.array([[3, 1, 3], [0, 1, 1]]),
            plot=-1,
        )
        cal = fs.calibrations["wavefront_superpixel"]
        (scheduling, references) = (cal["scheduling"], np.ravel(cal["reference_superpixels"]))

        conflicts = [
            (measurement, writer, reader, int(references[reader]))
            for measurement in range(scheduling.shape[1])
            for writer in range(len(references))
            for reader in range(len(references))
            if scheduling[writer, measurement] == references[reader]
            and scheduling[reader, measurement] != -1
        ]
        assert not conflicts, (
            f"(measurement, writer, reader, superpixel) hijacks: {conflicts}\n{scheduling}"
        )

    def test_wavefront_calibrate_superpixel_options(
        self, simulated_system_factory, caplog, subtests
    ):
        """The amplitude-only path, superpixel exclusion, and input checks, on a 4x4 supergrid."""
        # A 16-bit camera, exposed so a quarter-power superpixel is still far above one count.
        fs = simulated_system_factory(
            "matched", slm_resolution=(128, 128), cam_resolution=(256, 256), cam_bitdepth=16
        )
        fs.cam.set_exposure(30)
        point = [[190], [170]]
        install_ground_truth_calibration(fs)
        pixel_kxy = np.abs(np.diff(fs.ijcam_to_kxyslm([[0, 1], [0, 0]]), axis=1))[0, 0]

        def calibrate(**kwargs):
            # A fresh Fourier calibration, so the previous run does not read as newer.
            install_ground_truth_calibration(fs)
            kwargs = {"calibration_points": point, "superpixel_size": 32, "plot": -1, **kwargs}
            return fs.wavefront_calibrate_superpixel(**kwargs)

        amplitude = fs.slm.xp.ones(fs.slm.shape)
        amplitude[:, 64:] = 0.5
        fs.slm.source["amplitude_sim"] = amplitude
        cal = calibrate(phase_steps=None)
        power = cal["power"][0]

        with subtests.test("amplitude only measures power proportional to source intensity"):
            # Columns are x, so every row sees the same left/right step.
            np.testing.assert_allclose(power[[0, 1, 3]], power[[0]].repeat(3, axis=0), rtol=0.01)
            # The camera floors sub-count tails, which costs the dim half a few percent.
            assert np.nanmean(power[:, 2:]) / np.nanmean(power[:, :2]) == pytest.approx(
                0.25, rel=0.04
            )
            assert np.all(np.isnan(cal["phase"])) and np.all(np.isnan(cal["kx"]))

        with subtests.test("every measurement is normalized against the same reference"):
            normalization = cal["normalization"][0]
            assert np.unique(normalization[np.isfinite(normalization)]).size == 1
            reference = np.ravel(cal["reference_superpixels"])[0]
            assert np.isnan(power.ravel()[reference])

        with subtests.test("a margin skips the edge superpixels, and a denylist does too"):
            margin = calibrate(phase_steps=None, exclude_superpixels=(1, 1))
            edge = np.ones((4, 4), dtype=bool)
            edge[1:3, 1:3] = False
            assert np.all(np.isnan(margin["power"][0][edge]))
            assert np.sum(np.isfinite(margin["power"][0])) == 3  # The 2x2 core less its reference.

            denylist = calibrate(phase_steps=None, exclude_superpixels=edge)
            np.testing.assert_array_equal(denylist["scheduling"], margin["scheduling"])
            np.testing.assert_array_equal(denylist["power"], margin["power"])

        with subtests.test("a corrected amplitude measures the target blaze, here none needed"):
            fs.slm.source["amplitude_sim"] = fs.slm.xp.ones(fs.slm.shape)
            cal = calibrate(phase_steps=None, corrected_amplitude=True, measure_background=True)
            measured = np.isfinite(cal["power"][0])
            assert np.sum(measured) == 15  # All 16 less the reference.
            assert np.all(np.abs(cal["kx"][0][measured]) < 0.05 * pixel_kxy)
            assert np.all(np.abs(cal["ky"][0][measured]) < 0.05 * pixel_kxy)
            # A noiseless camera has no background.
            assert np.all(cal["background"][0][measured] == 0)

        with subtests.test("invalid inputs raise and leave the source untouched"):
            phase = fs.slm.xp.full(fs.slm.shape, 0.5)
            fs.slm.source["phase"] = phase
            for kwargs, message in (
                ({"exclude_superpixels": (1, 2, 3)}, "Did not recognize"),
                ({"exclude_superpixels": (1, 1), "reference_superpixels": (0, 0)}, "out of range"),
                ({"superpixel_size": 64, "calibration_points": np.full((2, 5), 128)}, "need as"),
                ({"phase_steps": 1.5}, "integer phase_steps"),
                ({"phase_steps": 0}, "positive phase_steps"),
                ({"phase_steps": 1, "test_index": 0, "plot": 3}, "phase_steps > 1"),
            ):
                with pytest.raises(ValueError, match=message):
                    calibrate(**kwargs)
                np.testing.assert_array_equal(
                    as_numpy(fs.slm.source["phase"]), as_numpy(phase), err_msg=str(kwargs)
                )

        with subtests.test("points closer than a window raise, and only warn for one index"):
            close = np.array([[180, 190], [170, 170]])
            with pytest.raises(ValueError, match="too close together"):
                calibrate(calibration_points=close, phase_steps=None)
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                caplog.clear()
                result = calibrate(calibration_points=close, phase_steps=None, test_index=0)
            assert "too close together" in caplog.text
            assert np.shape(result["power"]) == (2,)

    def test_wavefront_calibration_superpixel_process(self, simulated_system_factory, subtests):
        """
        Superpixels below the r2 threshold are filled in from their neighbors, so a pure
        diagonal ramp comes back out of the processor exactly. Only a lever arm carried
        along both axes at once separates a correct interpolation from a plausible one.
        """
        (NX, NY, superpixel_size) = (8, 8, 16)
        fs = simulated_system_factory(
            "matched", slm_resolution=(NX * superpixel_size, NY * superpixel_size)
        )
        holes = ((2, 3), (5, 2), (3, 5), (6, 6), (1, 6))

        def ramp_data(vector):
            """The processor's input for a pure ramp, with ``holes`` below threshold."""
            r2 = np.ones((NY, NX))
            for hx, hy in holes:
                r2[hy, hx] = 0
            return {
                "NX": NX,
                "NY": NY,
                "nxref": NX // 2,
                "nyref": NY // 2,
                "superpixel_size": superpixel_size,
                "r2_fit": r2,
                "power": np.ones((NY, NX)),
                "normalization": 2 * np.ones((NY, NX)),
                "background": np.zeros((NY, NX)),
                "kx": np.full((NY, NX), vector[0]),
                "ky": np.full((NY, NX), vector[1]),
                "phase": np.zeros((NY, NX)),
            }

        (x_grid, y_grid) = fs.slm.grid
        for vector in ((0.013, 0.021), (0.005, 0.009), (0.041, 0.033)):
            with subtests.test(f"kxy={vector}"):
                truth = 2 * np.pi * (vector[0] * as_numpy(x_grid) + vector[1] * as_numpy(y_grid))
                fs.calibrations["wavefront_superpixel"] = ramp_data(vector)
                phase = fs.wavefront_calibration_superpixel_process(
                    smooth=False,
                    r2_threshold=0.5,
                    remove_blaze=False,
                    remove_background=False,
                    apply=False,
                    plot=0,
                )["phase"]

                # A correction is defined only up to a global piston, so remove it.
                delta = np.exp(1j * (np.asarray(phase) - truth))
                piston = np.mean(delta)
                deviation = np.angle(delta * np.conj(piston) / np.abs(piston))
                assert np.max(np.abs(deviation)) < 1e-3

        with subtests.test("a failed fit reads as zero r2, not NaN"):
            data = ramp_data((0.013, 0.021))
            data["r2_fit"][holes[0][1], holes[0][0]] = np.nan
            fs.calibrations["wavefront_superpixel"] = data
            r2 = fs.wavefront_calibration_superpixel_process(
                smooth=False, r2_threshold=0.5, apply=False, plot=0
            )["r2"]
            assert not np.any(np.isnan(as_numpy(r2)))

        with subtests.test("an infinite power is clamped to the finite maximum"):
            data = ramp_data((0.013, 0.021))
            (hx, hy) = holes[0]
            data["power"][hy, hx] = np.inf
            fs.calibrations["wavefront_superpixel"] = data
            amplitude = fs.wavefront_calibration_superpixel_process(
                smooth=False, r2_threshold=0.5, remove_background=False, apply=False, plot=0
            )["amplitude"]
            center = (np.array([hy, hx]) + 0.5) * superpixel_size
            assert as_numpy(amplitude)[int(center[0]), int(center[1])] == pytest.approx(1)

        with subtests.test("a previous correction is added to the new one"):
            previous = as_numpy(zernike_sum(fs.slm, indices=(3, 4), weights=(1.0, -0.5)))
            data = ramp_data((0.013, 0.021))
            kwargs = {"smooth": False, "remove_background": False, "apply": False, "plot": 0}
            fs.calibrations["wavefront_superpixel"] = dict(data)
            fresh = fs.wavefront_calibration_superpixel_process(**kwargs)
            fs.calibrations["wavefront_superpixel"] = dict(data, previous_phase_correction=previous)
            stacked = fs.wavefront_calibration_superpixel_process(**kwargs)
            np.testing.assert_allclose(
                as_numpy(stacked["phase"]) - as_numpy(fresh["phase"]), previous, atol=1e-5
            )

        with subtests.test("smoothing leaves a gentle ramp unchanged away from the edges"):
            vector = (0.004, 0.003)
            truth = 2 * np.pi * (vector[0] * as_numpy(x_grid) + vector[1] * as_numpy(y_grid))
            fs.calibrations["wavefront_superpixel"] = ramp_data(vector)
            phase = fs.wavefront_calibration_superpixel_process(
                smooth=True,
                r2_threshold=0.5,
                remove_blaze=False,
                remove_background=False,
                apply=False,
                plot=0,
            )["phase"]

            # The blur reflects at the border, so only the interior keeps the ramp.
            interior = (slice(2 * superpixel_size, -2 * superpixel_size),) * 2
            delta = np.exp(1j * (np.asarray(phase) - truth))[interior]
            piston = np.mean(delta)
            deviation = np.angle(delta * np.conj(piston) / np.abs(piston))
            assert np.max(np.abs(deviation)) < 1e-4

        kwargs = {"smooth": False, "r2_threshold": 0.5, "apply": False, "plot": 0}

        with subtests.test("processing leaves the stored data unchanged"):
            fs.calibrations["wavefront_superpixel"] = ramp_data((0.013, 0.021))
            fs.wavefront_calibration_superpixel_process(**kwargs)
            np.testing.assert_equal(
                fs.calibrations["wavefront_superpixel"], ramp_data((0.013, 0.021))
            )

        with subtests.test("a calibration under the legacy key processes identically"):
            fs.calibrations["wavefront_superpixel"] = ramp_data((0.013, 0.021))
            current = fs.wavefront_calibration_superpixel_process(**kwargs)
            fs.calibrations["wavefront"] = fs.calibrations.pop("wavefront_superpixel")
            legacy = fs.wavefront_calibration_superpixel_process(**kwargs)
            for key in ("phase", "amplitude"):
                np.testing.assert_array_equal(as_numpy(legacy[key]), as_numpy(current[key]))

        with subtests.test("missing or empty data, and negative smoothing, raise"):
            del fs.calibrations["wavefront"]
            with pytest.raises(RuntimeError, match="Could not find"):
                fs.wavefront_calibration_superpixel_process(**kwargs)
            fs.calibrations["wavefront_superpixel"] = {}
            with pytest.raises(RuntimeError, match="No raw wavefront data"):
                fs.wavefront_calibration_superpixel_process(**kwargs)
            fs.calibrations["wavefront_superpixel"] = ramp_data((0.013, 0.021))
            with pytest.raises(ValueError, match="non-negative"):
                fs.wavefront_calibration_superpixel_process(**{**kwargs, "smooth": -1})

    @pytest.mark.slow
    def test_wavefront_calibrate_zernike(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.wavefront_calibrate_zernike."""
        # The points are passed to CompressedSpotHologram in the zernike basis (radians),
        # not in ij pixels.
        calibration_points = convert_vector(
            fourierslm_calibrated.wavefront_calibration_points(pitch=120),
            from_units="ij",
            to_units="zernike",
            hardware=fourierslm_calibrated,
        )

        with subtests.test("no perturbation projects the spots and calibrates nothing"):
            fourierslm_calibrated.wavefront_calibrate_zernike(
                calibration_points=calibration_points,
                zernike_indices=4,
                perturbation=0,
                optimize_position=False,
                optimize_weights=False,
                plot=-1,
            )
            assert "wavefront_zernike" not in fourierslm_calibrated.calibrations

        with subtests.test("a sweep stores the corrected spots in their basis"):
            fourierslm_calibrated.wavefront_calibrate_zernike(
                calibration_points=calibration_points,
                zernike_indices=4,
                perturbation=0.5,
                optimize_position=False,
                optimize_weights=False,
                plot=-1,
            )
            cal = fourierslm_calibrated.calibrations["wavefront_zernike"]
            assert cal["corrected_spots"].shape[1] == calibration_points.shape[1]
            # An integer basis expands to that many terms, tilt and focus first.
            assert np.array_equal(np.ravel(cal["zernike_indices"]), [2, 1, 4, 3])

        with subtests.test("iterating starts from the spots of the last calibration"):
            fourierslm_calibrated.wavefront_calibrate_zernike(
                perturbation=0.3,
                optimize_position=False,
                optimize_weights=False,
                plot=-1,
            )
            cal = fourierslm_calibrated.calibrations["wavefront_zernike"]
            assert cal["corrected_spots"].shape[1] == calibration_points.shape[1]

    def test_wavefront_calibrate_zernike_planted(self, simulated_system_factory, subtests):
        """A planted astigmatism is measured by the sweep, then corrected by the spots."""
        weight = 0.8

        def calibrate(global_correction=False):
            # Spots a few pixels wide, so that the spot-area metric resolves the aberration.
            fs = simulated_system_factory("matched", ratio=4.0)
            fs.cam.set_exposure(0.3)
            install_ground_truth_calibration(fs)
            # Planted as the very term the sweep perturbs, so the optimum is -weight.
            fs.slm.source["phase_sim"] = fs.slm.source["phase_sim"] + fs.slm.xp.asarray(
                weight * zernike(fs.slm, 3, use_mask=False)
            )
            points_kxy = view_kxy_grid(fs, count=2, frac=0.4)
            return fs.wavefront_calibrate_zernike(
                calibration_points=convert_vector(points_kxy, "kxy", "zernike", hardware=fs),
                zernike_indices=[2, 1, 3],
                perturbation=np.linspace(-1.5, 1.5, 11),
                global_correction=global_correction,
                optimize_position=False,
                optimize_weights=False,
                plot=-1,
            )

        cal = calibrate()
        measured = cal["corrected_spots"][2]

        with subtests.test("the correction opposes the planted term and nearly matches it"):
            # A parabola fit over a sweep wider than the well pulls the optimum toward zero.
            assert np.all(np.abs(measured / -weight - 1) < 0.25), measured

        with subtests.test("the corrected spots are tighter than the uncorrected ones"):
            stats = np.array(cal["metric_stats"])
            assert np.mean(stats[-1]) < 0.95 * np.mean(stats[0]), stats

        with subtests.test("global_correction applies the mean correction to every spot"):
            shared = calibrate(global_correction=True)["corrected_spots"][2]
            np.testing.assert_allclose(shared, np.mean(measured))

    def test_wavefront_calibrate_zernike_options(self, simulated_system_factory, capfd, subtests):
        """How wavefront_calibrate_zernike takes its perturbation, metric, basis and weights."""
        fs = simulated_system_factory("fov_smaller")
        install_ground_truth_calibration(fs)
        points = convert_vector(view_kxy_grid(fs, count=2, frac=0.4), "kxy", "zernike", hardware=fs)
        kwargs = {"optimize_position": False, "optimize_weights": False, "plot": -1}

        with subtests.test("a metric sees each spot's window at unit total, and is recorded"):
            seen = []

            def metric(images):
                seen.append(np.array(images))
                return -np.max(images, axis=(1, 2))

            cal = fs.wavefront_calibrate_zernike(
                calibration_points=points,
                zernike_indices=[2, 1, 4],
                perturbation=np.linspace(-0.4, 0.4, 5),
                metric=metric,
                **kwargs,
            )
            width = cal["spot_integration_width_ij"]
            assert len(seen) == 1 + 5 + 1  # Before the sweep, during it, and after.
            assert all(images.shape == (4, width, width) for images in seen)
            np.testing.assert_allclose([np.sum(images) for images in seen], 1)
            np.testing.assert_allclose(
                cal["metric_stats"], [-np.max(seen[i], axis=(1, 2)) for i in (0, -1)]
            )

        with subtests.test("a three-point sweep fits without an OptimizeWarning"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fs.wavefront_calibrate_zernike(
                    calibration_points=points,
                    zernike_indices=[2, 1, 4],
                    perturbation=[-0.3, 0, 0.3],
                    **kwargs,
                )
            assert not [w for w in caught if issubclass(w.category, OptimizeWarning)]

        with subtests.test("plot=-1 prints no optimize_weights progress bar"):
            capfd.readouterr()
            fs.wavefront_calibrate_zernike(
                calibration_points=points,
                zernike_indices=[2, 1, 4],
                perturbation=0,
                **{**kwargs, "optimize_weights": 2},
            )
            assert "optimize_weights" not in capfd.readouterr().err

        with subtests.test("optimize_weights must be a positive count"):
            with pytest.raises(ValueError, match="positive integer"):
                fs.wavefront_calibrate_zernike(
                    calibration_points=points,
                    zernike_indices=[2, 1, 4],
                    perturbation=0,
                    **{**kwargs, "optimize_weights": -1},
                )

        with subtests.test("fewer than two points raise"):
            with pytest.raises(ValueError, match="at least 2"):
                fs.wavefront_calibrate_zernike(
                    calibration_points=points[:, :1],
                    zernike_indices=[2, 1, 4],
                    perturbation=0,
                    **kwargs,
                )

    def test_wavefront_calibrate_zernike_smooth(self, simulated_system_factory, caplog, subtests):
        """Test FourierSLM.wavefront_calibrate_zernike_smooth."""
        (fs, _) = _zernike_calibrated(simulated_system_factory)
        spots = fs.calibrations["wavefront_zernike"]["corrected_spots"]

        with subtests.test("a smoothed copy is returned, leaving the calibration alone"):
            identity = fs.wavefront_calibrate_zernike_smooth(smoothing=0, smoothing_xy=0)
            assert identity.shape == spots.shape
            assert fs.calibrations["wavefront_zernike"]["corrected_spots"] is spots
            assert np.allclose(identity, spots)

        with subtests.test("full smoothing replaces a term with its neighborhood"):
            # At smoothing=1 a coordinate is entirely its neighbors' average.
            spots[2, 4] = 5.0
            high = fs.wavefront_calibrate_zernike_smooth(smoothing=1, smoothing_xy=0)
            spots[2, 4] = -5.0
            low = fs.wavefront_calibrate_zernike_smooth(smoothing=1, smoothing_xy=0)

            assert high[2, 4] == pytest.approx(low[2, 4])
            assert abs(high[2, 4]) < 1.0
            assert np.allclose(high[:2], spots[:2])  # Tilt is left to smoothing_xy.

        with subtests.test("a spot with no neighbors is left alone"):
            points_ij = fs.calibrations["wavefront_zernike"]["calibration_points_ij"]
            original = np.copy(points_ij)
            points_ij[:2, 0] += 10 * np.min(fs.cam.shape)
            try:
                smoothed = fs.wavefront_calibrate_zernike_smooth(smoothing=0.5, smoothing_xy=0.5)
            finally:
                points_ij[:] = original
            assert np.allclose(smoothed[:, 0], spots[:, 0])

        with subtests.test("a changed camera window is flagged"):
            fs.cam.set_woi((0, 64, 0, 48))
            try:
                with caplog.at_level(logging.WARNING):
                    caplog.clear()
                    fs.wavefront_calibrate_zernike_smooth()
                    smooth_warned = "does not follow the window" in caplog.text
                    caplog.clear()
                    fs._wavefront_calibration_zernike_interpolator_2d = None
                    fs.wavefront_calibrate_zernike_get((0, 0))
                    get_warned = "does not follow the window" in caplog.text
            finally:
                fs.cam.set_woi(None)
            assert smooth_warned and get_warned

        with subtests.test("a factor outside [0, 1] raises"):
            for kwargs in ({"smoothing": 1.5}, {"smoothing_xy": -0.1}):
                with pytest.raises(ValueError, match="Smoothing factor must be between 0 and 1"):
                    fs.wavefront_calibrate_zernike_smooth(**kwargs)

        with subtests.test("focus smoothing is not implemented"):
            with pytest.raises(RuntimeError):
                fs.wavefront_calibrate_zernike_smooth(smoothing_z=0.5)

    def test_wavefront_calibrate_zernike_get(self, simulated_system_factory, subtests):
        """Test FourierSLM.wavefront_calibrate_zernike_get."""
        (fs, points_ij) = _zernike_calibrated(simulated_system_factory)
        indices = np.ravel(fs.calibrations["wavefront_zernike"]["zernike_indices"])

        with subtests.test("a vector yields one coefficient per index per point"):
            coefficients = fs.wavefront_calibrate_zernike_get(points_ij[:, :3], from_units="ij")
            assert np.shape(coefficients) == (len(indices), 3)
            assert np.all(np.isfinite(coefficients))

        with subtests.test("no vector yields the interpolator itself"):
            interpolator = fs.wavefront_calibrate_zernike_get()
            assert callable(interpolator)

        with subtests.test("the interpolator agrees with evaluating directly"):
            kxy = fs.ijcam_to_kxyslm(points_ij[:, :3])
            assert np.allclose(
                interpolator(kxy),
                fs.wavefront_calibrate_zernike_get(points_ij[:, :3], from_units="ij"),
            )

        cal = fs.calibrations["wavefront_zernike"]
        calibrated_ij = cal["calibration_points_ij"][:2]

        with subtests.test("the calibrated spots are reproduced at their own points"):
            np.testing.assert_allclose(
                fs.wavefront_calibrate_zernike_get(calibrated_ij, from_units="ij"),
                cal["corrected_spots"],
                rtol=1e-6,
                atol=1e-9,
            )

        with subtests.test("an edited affine field is reproduced exactly, even off the grid"):
            A = np.array([[40.0, -15.0], [8.0, 25.0], [-30.0, 12.0]])[: len(indices)]
            c = np.array([[0.5], [-1.0], [0.25]])[: len(indices)]
            base = convert_vector(calibrated_ij, "ij", "norm", hardware=fs)
            cal["corrected_spots"] = A @ base[:2] + c

            # Between the points, and outside both their hull and the camera.
            off_grid = np.array([[3.5, 40.25, 200.0, -30.0], [7.75, 90.5, -15.0, 150.0]])
            expected = A @ convert_vector(off_grid, "ij", "norm", hardware=fs)[:2] + c
            np.testing.assert_allclose(
                fs.wavefront_calibrate_zernike_get(off_grid, from_units="ij"),
                expected,
                rtol=1e-6,
                atol=1e-9,
            )

        with subtests.test("a 3D vector raises"):
            with pytest.raises(ValueError, match="3D Zernike"):
                fs.wavefront_calibrate_zernike_get(np.zeros((3, 2)))

        with subtests.test("without a calibration, a RuntimeError names what is missing"):
            del fs.calibrations["wavefront_zernike"]
            with pytest.raises(RuntimeError, match="Could not find Zernike"):
                fs.wavefront_calibrate_zernike_get((0, 0))

    def test_wavefront_calibration_points(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.wavefront_calibration_points."""
        fs = fourierslm_calibrated
        (h, w) = fs.cam.shape
        exclusion = 60
        points = fs.wavefront_calibration_points(pitch=60, field_exclusion=exclusion)

        with subtests.test("every point lies on the camera"):
            assert points.shape[0] == 2 and points.shape[1] > 0
            assert np.all(points[0] >= 0) and np.all(points[0] < w)
            assert np.all(points[1] >= 0) and np.all(points[1] < h)

        with subtests.test("no point lies within field_exclusion of the 0th order"):
            zeroth = fs.kxyslm_to_ijcam([0, 0])
            assert np.min(np.linalg.norm(points - zeroth, axis=0)) >= exclusion

        with subtests.test("a coarser pitch asks for fewer points"):
            assert fs.wavefront_calibration_points(pitch=120).shape[1] < points.shape[1]

        with subtests.test("turning the avoidance rules off keeps the points they removed"):
            assert (
                fs.wavefront_calibration_points(
                    pitch=60, field_exclusion=0, avoid_mirrors=False, avoid_nyquist=False, plot=1
                ).shape[1]
                > points.shape[1]
            )

        with subtests.test("the lattice is centered in the addressable region"):
            for pitch in np.arange(20, 400):
                lattice = fs.wavefront_calibration_points(
                    pitch, field_exclusion=0, avoid_mirrors=False
                )
                box = fs.get_farfield_extent(inscribe=True, margin=pitch / 2)
                box_center = (np.min(box, axis=1) + np.max(box, axis=1)) / 2
                lattice_center = (np.min(lattice, axis=1) + np.max(lattice, axis=1)) / 2
                assert np.all(np.abs(lattice_center - box_center) <= 1), pitch

        with subtests.test("a per-axis pitch spaces each axis by its own pitch"):
            lattice = fs.wavefront_calibration_points(
                pitch=(40, 80), field_exclusion=0, avoid_mirrors=False
            )
            for axis, pitch in enumerate((40, 80)):
                spacing = np.diff(np.unique(np.rint(lattice[axis])))
                assert spacing.size > 0 and np.all(spacing >= pitch), (axis, spacing)

        with subtests.test("avoid_points prunes the points within field_exclusion of them"):
            kwargs = {"pitch": 60, "field_exclusion": 40, "avoid_mirrors": False}
            unpruned = fs.wavefront_calibration_points(**kwargs)
            avoid = unpruned[:, [1]] + [[5], [-5]]
            pruned = fs.wavefront_calibration_points(avoid_points=avoid, **kwargs)
            assert pruned.shape[1] == unpruned.shape[1] - 1
            assert np.min(np.linalg.norm(pruned - avoid, axis=0)) >= 40

    def test_wavefront_calibration_points_parse(self, fourierslm_calibrated, subtests):
        """How the calibration routines turn a count, a list, or nothing into camera points."""
        fs = fourierslm_calibrated
        generated = np.rint(fs.wavefront_calibration_points(pitch=60)).astype(int)

        with subtests.test("None takes every generated point"):
            np.testing.assert_array_equal(
                fs._wavefront_calibration_points_parse(None, pitch=60), generated
            )

        with subtests.test("an integer takes that many of them"):
            np.testing.assert_array_equal(
                fs._wavefront_calibration_points_parse(2, pitch=60), generated[:, :2]
            )
            assert (
                fs._wavefront_calibration_points_parse(2 * generated.shape[1], pitch=60).shape[1]
                == generated.shape[1]
            ), "a count beyond the grid asks for no more"

        with subtests.test("a count must be positive"):
            for count in (0, -3):
                with pytest.raises(ValueError, match="must be positive"):
                    fs._wavefront_calibration_points_parse(count, pitch=60)

        with subtests.test("points are rounded to whole camera pixels"):
            parsed = fs._wavefront_calibration_points_parse([[60.4], [70.6]])
            np.testing.assert_array_equal(parsed, [[60], [71]])

        with subtests.test("points off the sensor are refused"):
            (h, w) = fs.cam.shape
            with pytest.raises(ValueError, match="field of view"):
                fs._wavefront_calibration_points_parse([[w + 10], [h + 10]])

        with subtests.test("a pitch too coarse for any point raises"):
            with pytest.raises(ValueError, match="No calibration points fit"):
                fs._wavefront_calibration_points_parse(None, pitch=10 * max(fs.cam.shape))

    def test_wavefront_calibrate(self, fourierslm_calibrated, monkeypatch, subtests):
        """The dispatcher in front of the superpixel and Zernike implementations."""
        fs = fourierslm_calibrated
        forwarded = {}

        for method in ("superpixel", "zernike"):
            monkeypatch.setattr(
                type(fs),
                f"wavefront_calibrate_{method}",
                lambda self, method=method, **kwargs: forwarded.setdefault(method, kwargs),
            )

        with subtests.test("an unknown method raises"):
            with pytest.raises(ValueError, match="not recognized"):
                fs.wavefront_calibrate(method="nonsense")

        with subtests.test("the method chooses the implementation, superpixel by default"):
            fs.wavefront_calibrate(calibration_points=3)
            assert forwarded["superpixel"] == {"calibration_points": 3}

            fs.wavefront_calibrate(method="zernike", calibration_points=3)
            assert forwarded["zernike"] == {"calibration_points": 3}

        with subtests.test("the retired point arguments warn and forward under the new name"):
            for retired in ("interference_point", "calibration_point"):
                forwarded.clear()
                with pytest.warns(UserWarning, match="deprecated"):
                    fs.wavefront_calibrate(method="superpixel", **{retired: [[60], [70]]})
                assert forwarded["superpixel"] == {"calibration_points": [[60], [70]]}

    def test_pixel_calibrate(self, simulated_system_factory, caplog, subtests):
        """Test FourierSLM.pixel_calibrate on a crosstalk-free simulated system."""
        fs = simulated_system_factory("fov_much_smaller")
        install_ground_truth_calibration(fs)

        with subtests.test("sweeps and stores raw data"):
            cal = fs.pixel_calibrate(levels=8, periods=2, orders=1, directions="x", plot=False)
            assert cal["data"].shape == (2, 2, 8, 8, 3)
            assert np.all(cal["periods"] % 2 == 0)
            assert np.any(cal["data"] > 0)

        with subtests.test("only the swept direction is populated"):
            assert np.all(cal["data"][1] == 0)

        with subtests.test("intensity depends only on the level difference"):
            # An ideal SLM diffracts on phase difference alone, so each slice is circulant.
            for order in range(cal["data"].shape[-1]):
                data = cal["data"][0, 0, :, :, order]
                assert np.allclose(data, np.array([np.roll(data[0], k) for k in range(8)]))

        def sampling_warnings(levels):
            """Warnings that the levels are too few to resolve the swept phase range."""
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                fs.pixel_calibrate(
                    levels=levels,
                    periods=1,
                    orders=1,
                    directions="x",
                    test_index=0,
                    plot=False,
                )
            return [r.getMessage() for r in caplog.records if "cycles" in r.getMessage()]

        with subtests.test("silent when the levels densely sample the phase range"):
            assert not sampling_warnings(32)

        with subtests.test("warns when they do not"):
            assert sampling_warnings(8)

        for case in ("matched", "fov_larger", "fov_much_smaller", "camera_wide", "fov_extreme"):
            with subtests.test(f"{case}: the periods chosen keep every order on camera"):
                geometry = simulated_system_factory(case)
                install_ground_truth_calibration(geometry)
                swept = geometry.pixel_calibrate(levels=2, periods=2, plot=False)
                assert np.all(swept["periods"] >= 2)

        with subtests.test("a one-directional sweep ignores the other direction's orders"):
            # A wide camera has far less room in y, which must not veto an x-only sweep.
            wide = simulated_system_factory("camera_wide")
            install_ground_truth_calibration(wide)
            wide.pixel_calibrate(levels=2, periods=[4], orders=1, directions="x", plot=False)

        with subtests.test("rejects orders that fall off the negative side of the sensor"):
            # Negative indices wrap rather than raising, so they would otherwise be
            # integrated from the opposite edge of the image.
            offset = simulated_system_factory("zeroth_outside")
            install_ground_truth_calibration(offset)
            with pytest.raises(ValueError, match="short of the camera"):
                offset.pixel_calibrate(levels=2, periods=[6], orders=1, directions="x", plot=False)

        with subtests.test("test_index booleans"):
            # True tests the first len(levels) points of the sweep; False is not a test at all.
            for true in (True, np.True_):  # numpy booleans are not bool instances.
                result = fs.pixel_calibrate(
                    levels=4, periods=1, orders=1, directions="x", test_index=true, plot=False
                )
                np.testing.assert_array_equal(result["indices"], np.arange(4))
            result = fs.pixel_calibrate(
                levels=2, periods=1, orders=1, directions="x", test_index=False, plot=False
            )
            assert "data" in result

        with subtests.test("rejects a test_index selecting nothing"):
            with pytest.raises(ValueError, match="no points"):
                fs.pixel_calibrate(levels=2, periods=1, orders=1, test_index=[], plot=False)

        with subtests.test("rejects odd periods"):
            with pytest.raises(ValueError, match="even"):
                fs.pixel_calibrate(levels=2, periods=[25], orders=1, plot=False)

        with subtests.test("rejects repeated orders"):
            with pytest.raises(ValueError, match="Repeated orders"):
                fs.pixel_calibrate(levels=2, periods=1, orders=[-1, 1, 1], plot=False)

        with subtests.test("rejects a sweep without the 1st order"):
            with pytest.raises(ValueError, match="1st order"):
                fs.pixel_calibrate(levels=2, periods=[26], orders=[0], plot=False)

        sweep = {"periods": [26], "orders": 1, "directions": "x", "plot": False}

        with subtests.test("a count of levels rounds up to a power of two, spread evenly"):
            levels = fs.pixel_calibrate(levels=3, **sweep)["levels"]
            np.testing.assert_array_equal(levels, [0, 64, 128, 192])

        with subtests.test("more levels than the SLM has truncate to its bitresolution"):
            B = fs.slm.bitresolution
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                # The last index of a one-period, one-direction sweep is (levels**2 - 1).
                result = fs.pixel_calibrate(levels=4 * B, test_index=-1, **sweep)
            np.testing.assert_array_equal(result["indices"], [B * B - 1])
            assert "more than the bitresolution" in caplog.text

        with subtests.test("levels outside the SLM's range are dropped, and none left raises"):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                levels = fs.pixel_calibrate(levels=[-1, 0, 128, 300], **sweep)["levels"]
            np.testing.assert_array_equal(levels, [0, 128])
            assert "Omitting requested levels" in caplog.text
            with pytest.raises(ValueError, match="No valid levels"):
                fs.pixel_calibrate(levels=[-5, 999], **sweep)

        with subtests.test("integer directions name the axes, and an unknown one raises"):
            data = fs.pixel_calibrate(levels=2, **{**sweep, "directions": [1]})["data"]
            assert np.all(data[0] == 0) and np.any(data[1] > 0)
            with pytest.raises(ValueError, match="No valid directions"):
                fs.pixel_calibrate(levels=2, **{**sweep, "directions": "z"})

        with subtests.test("rejects repeated and non-positive periods"):
            for periods, message in (
                ([26, 26], "Repeated periods"),
                ([0], "positive"),
                ([-26], "positive"),
            ):
                with pytest.raises(ValueError, match=message):
                    fs.pixel_calibrate(levels=2, **{**sweep, "periods": periods})

        with subtests.test("rejects periods longer than half the window"):
            with pytest.raises(ValueError, match="half of the window"):
                fs.pixel_calibrate(levels=2, window=(32, 32, 32, 32), **sweep)

        with subtests.test("autoexposing test indices reports each exposure, keeping the least"):
            try:
                result = fs.pixel_calibrate(levels=2, test_index=[0, 1], autoexpose=True, **sweep)
                np.testing.assert_array_equal(result["indices"], [0, 1])
                assert len(result["results"]) == 2
                assert fs.cam.get_exposure() == pytest.approx(min(result["results"]))
            finally:
                fs.cam.set_exposure(0.1)

        with subtests.test("plots"):
            fs.pixel_calibrate(levels=2, periods=1, orders=2, directions="x", plot=2)
            fs.pixel_calibration_plot(summed=True)
            fs.pixel_calibration_plot(orders=[1])  # A single order must not squeeze away.
            with pytest.raises(ValueError, match="were measured"):
                fs.pixel_calibration_plot(orders=[99])

    def test_pixel_calibration_process(self, simulated_system_factory, temp_dir, caplog, subtests):
        """Test FourierSLM.pixel_calibration_process, which fits the phase response."""
        fs = simulated_system_factory("fov_much_smaller")
        install_ground_truth_calibration(fs)
        cal = fs.pixel_calibrate(levels=8, periods=2, orders=1, directions="x", plot=False)

        with subtests.test("gamma recovers the linear phase response"):
            gamma = fs.pixel_calibration_process(plot=False)
            expected = cal["levels"] / fs.slm.bitresolution
            assert np.allclose(gamma, expected - np.min(expected), atol=0.02)
            # The fit is seeded with the linear response, so agreement alone proves
            # nothing; the fit must also actually describe the data.
            assert fs.calibrations["pixel"]["gamma_r2"] > 0.9

        with subtests.test("the sampled fit is applied across every level"):
            B = fs.slm.bitresolution
            # The sweep measures a handful of levels; the SLM carries all of them.
            assert len(gamma) < B
            assert fs.slm.gamma.shape == (B,)
            np.testing.assert_allclose(as_numpy(fs.slm.gamma), np.arange(B) / B, atol=0.02)

        with subtests.test("apply=False leaves the SLM alone"):
            fs.slm.set_gamma(None)
            fs.pixel_calibration_process(plot=False, apply=False)
            assert fs.slm.gamma is None and fs.slm.lut is None

        with subtests.test("loading the calibration restores the response"):
            path = fs.save_calibration("pixel", path=temp_dir, name="test_gamma")
            fs.slm.set_gamma(None)
            fs.load_calibration("pixel", file_path=path)
            assert fs.slm.gamma is not None and fs.slm.lut is not None

        with subtests.test("a calibration from another bitdepth is not applied"):
            fs.slm.set_gamma(None)
            meta = fs.calibrations["pixel"].setdefault("__meta__", {}).setdefault("slm", {})
            bitresolution = meta.get("bitresolution", fs.slm.bitresolution)
            meta["bitresolution"] = 4 * fs.slm.bitresolution
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                fs._pixel_calibration_apply_gamma()
            assert any("level SLM" in r.message for r in caplog.records)
            assert fs.slm.gamma is None and fs.slm.lut is None
            meta["bitresolution"] = bitresolution

        with subtests.test("loading onto a retuned SLM warns"):
            wav_um = fs.slm.wav_um
            fs.slm.wav_um = 2 * wav_um
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                fs.load_calibration("pixel", file_path=path)
            assert any("was taken at" in r.message for r in caplog.records)
            fs.slm.wav_um = wav_um

        with subtests.test("a fit that does not describe the data is flagged"):
            fs.calibrations["pixel"]["data"] = np.random.default_rng(0).uniform(
                size=fs.calibrations["pixel"]["data"].shape
            )
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                caplog.clear()
                fs.pixel_calibration_process(plot=False, apply=False)
            assert fs.calibrations["pixel"]["gamma_r2"] < 0.9
            assert "Low R^2" in caplog.text

        with subtests.test("refuses signal-free data"):
            fs.calibrations["pixel"]["data"][:] = 0
            with pytest.raises(RuntimeError, match="no signal"):
                fs.pixel_calibration_process(plot=False)

    def test_pixel_calibration_process_planted(self, simulated_system_factory, subtests):
        """A simulated non-linear response is measured, then corrected by the table."""
        fs = simulated_system_factory("fov_much_smaller")
        install_ground_truth_calibration(fs)

        B = fs.slm.bitresolution
        truth = np.square(np.arange(B) / (B - 1)) * (B - 1) / B
        fs.slm.gamma_sim = truth

        fs.pixel_calibrate(levels=16, periods=2, orders=1, directions="x", plot=False)
        gamma = fs.pixel_calibration_process(plot=False)

        with subtests.test("the sweep measures the simulated response"):
            levels = fs.calibrations["pixel"]["levels"].astype(int)
            expected = truth[levels] - np.min(truth[levels])
            assert np.allclose(gamma, expected, atol=0.02)
            assert not np.allclose(gamma, levels / B, atol=0.02)  # Not the ideal ramp.

        with subtests.test("the table corrects the response"):
            target = np.arange(B) * (2 * np.pi / B)

            def error(levels):
                realized = np.mod(fs.slm._gamma_sign * 2 * np.pi * truth[levels], 2 * np.pi)
                deviation = (realized - target + np.pi) % (2 * np.pi) - np.pi
                return np.sqrt(np.mean(np.square(deviation)))

            uncorrected = error((-np.rint(target * B / (2 * np.pi)).astype(int)) % B)
            corrected = error(
                as_numpy(fs.slm.lut)[np.floor(target * fs.slm._phase_to_lut).astype(int)]
            )
            assert corrected < uncorrected / 8

        with subtests.test("a response beyond one cycle is unwrapped"):
            # The fit resolves each level only modulo a cycle.
            truth = 2.4 * np.arange(B) / B
            fs.slm.gamma_sim = truth
            fs.pixel_calibrate(levels=16, periods=2, orders=1, directions="x", plot=False)
            gamma = fs.pixel_calibration_process(plot=False)

            levels = fs.calibrations["pixel"]["levels"].astype(int)
            assert np.max(gamma) > 1  # More than one cycle recovered.
            assert np.allclose(gamma, truth[levels] - np.min(truth[levels]), atol=0.02)

        with subtests.test("a mirrored fit is returned on the increasing branch"):
            assert np.all(np.diff(gamma) > 0)

        with subtests.test("levels are canonically ordered"):
            # Unwrapping the fit follows the order of the levels it was measured at.
            fs.pixel_calibrate(
                levels=np.array([96, 0, 192, 32]),
                periods=2,
                orders=1,
                directions="x",
                plot=False,
            )
            stored = fs.calibrations["pixel"]["levels"]
            np.testing.assert_array_equal(stored, np.sort(stored))

    def test_pixel_kernel(self, subtests):
        """The pixel kernel is normalized, peaked, and shaped by its two sides."""
        x = (np.arange(401) - 200) / 50  # +-4 pixels at 50 samples per pixel.

        with subtests.test("normalized"):
            assert np.isclose(np.sum(FourierSLM.pixel_kernel(x, a_pix=0.5)), 1)

        with subtests.test("symmetric by default"):
            kernel = FourierSLM.pixel_kernel(x, a_pix=0.5, n=2)
            assert np.allclose(kernel, np.flip(kernel))

        with subtests.test("asymmetric widths and exponents"):
            for kwargs in ({"a_minus_pix": 0.05}, {"n_minus": 4}):
                kernel = FourierSLM.pixel_kernel(x, a_pix=0.5, n=1, **kwargs)
                assert not np.allclose(kernel, np.flip(kernel))

        with subtests.test("swapping the two sides mirrors the kernel"):
            kernel = FourierSLM.pixel_kernel(x, a_pix=0.5, n=1, a_minus_pix=0.05, n_minus=4)
            mirror = FourierSLM.pixel_kernel(x, a_pix=0.05, n=4, a_minus_pix=0.5, n_minus=1)
            assert np.allclose(kernel, np.flip(mirror))

        with subtests.test("peaked at the origin and decaying"):
            kernel = FourierSLM.pixel_kernel(x, a_pix=0.5, n=1, a_minus_pix=0.05)
            assert np.argmax(kernel) == np.argmin(np.abs(x))
            assert np.all(np.diff(kernel[x >= 0]) <= 0)
            assert np.all(np.diff(kernel[x <= 0]) >= 0)

        with subtests.test("x0_pix displaces the peak"):
            kernel = FourierSLM.pixel_kernel(x, a_pix=0.5, n=2, x0_pix=0.4)
            assert np.isclose(x[np.argmax(kernel)], 0.4)
            assert np.allclose(kernel, FourierSLM.pixel_kernel(x - 0.4, a_pix=0.5, n=2))

    def test_pixel_crosstalk_simulate(self, subtests):
        """Test FourierSLM._pixel_crosstalk_simulate, the forward model of the
        crosstalk kernel's effect on the diffraction orders of a grating."""
        simulate = FourierSLM._pixel_crosstalk_simulate
        (grating_50, grating_25) = (
            _binary_grating(16, np.pi, 0),
            _binary_grating(16, np.pi, 0, duty_cycle=0.25),
        )

        with subtests.test("crosstalk-free limit matches the analytic binary grating"):
            # A 50% duty grating of phase difference d diffracts (2/(pi m))^2 sin^2(d/2)
            # into odd order m, and nothing into even orders.
            for delta in (np.pi, 0.6 * np.pi):
                orders = simulate(_binary_grating(16, delta, 0), a_pix=1e-4)
                for m in (1, 3):
                    assert np.isclose(
                        orders[m], (2 / (np.pi * m)) ** 2 * np.sin(delta / 2) ** 2, rtol=1e-2
                    )
                assert np.allclose(orders[[2, 4]], 0, atol=1e-6)

        with subtests.test("a kernel narrower than the sampling stays finite"):
            # Such a kernel underflows to zero, and must fall back to a delta function.
            for period in (15, 16):
                for supersample in (1, 3, 16):
                    orders = simulate(
                        _binary_grating(period, np.pi, 0),
                        supersample=supersample,
                        a_pix=1e-6,
                    )
                    assert np.all(np.isfinite(orders)) and np.isclose(np.sum(orders), 1)

        with subtests.test("kernel is applied on the SLM pixel scale"):
            # Weakly-phased orders report the kernel's transfer function at m/p, which
            # pins a_pix absolutely.
            (delta, period) = (0.01, 16)
            for a_pix in (0.25, 0.5):
                for supersample in (32, 64):
                    orders = simulate(
                        _binary_grating(period, delta, 0),
                        supersample=supersample,
                        a_pix=a_pix,
                        n=1,
                    )
                    for m in (1, 3):
                        transfer = 1 / (1 + (2 * np.pi * m * a_pix / period) ** 2)
                        assert np.isclose(
                            orders[m],
                            (delta / (np.pi * m)) ** 2 * transfer**2,
                            rtol=1e-2,
                        )

        with subtests.test("constant parameters reduce to a convolution"):
            # Moser Eq. (12) with fixed parameters is exactly the usual crosstalk model.
            # The reference tiles the grating, so that its kernel is untruncated too.
            (supersample, tiles) = (32, 5)
            commanded = np.repeat(np.tile(grating_50, tiles), supersample)
            size = commanded.size - 1
            for kwargs in ({"a_minus_pix": 0.05}, {"n_minus": 4}, {"a_pix": 2.0}):
                kwargs = {"a_pix": 0.5, **kwargs}
                kernel = FourierSLM.pixel_kernel(
                    (np.arange(size) - (size - 1) / 2) / supersample, **kwargs
                )
                blurred = ndimage.convolve1d(commanded, kernel, mode="grid-wrap")
                expected = np.square(np.abs(np.fft.fft(np.exp(1j * blurred)) / commanded.size))
                orders = simulate(grating_50, supersample=supersample, **kwargs)
                assert np.allclose(orders[:9], expected[::tiles][:9], atol=1e-12)

        with subtests.test("independent of how many periods are supplied"):
            # The kernel's support must be set by the kernel, not by the pattern.
            for kwargs in ({"a_pix": 0.5, "a_minus_pix": 0.05}, {"a_pix": 2.0}):
                for period in (4, 16):
                    grating = _binary_grating(period, np.pi, 0, duty_cycle=0.25)
                    one = simulate(grating, supersample=64, **kwargs)
                    four = simulate(np.tile(grating, 4), supersample=64, **kwargs)
                    assert np.allclose(one[:5], four[::4][:5], atol=1e-12)

        with subtests.test("50% duty gratings cannot show a constant kernel's asymmetry"):
            # phi(x + p/2) = a + b - phi(x) survives any constant kernel and forces
            # |E_m| = |E_-m|.  This is the blind spot of pixel_calibrate().
            for kwargs in ({"a_minus_pix": 0.05}, {"n_minus": 4}):
                for delta in (np.pi, 0.6 * np.pi):
                    orders = simulate(_binary_grating(16, delta, 0), a_pix=0.5, **kwargs)
                    assert np.isclose(orders[1], orders[-1], rtol=1e-12)

        with subtests.test("level-dependent width breaks the 50% duty symmetry"):
            # Moser's mechanism: LC pre-tilt steepens the transition for one sign of the
            # level step and flattens it for the other, which no convolution can do.
            # Signed, so that swapping the (phi0, phi1) convention fails.
            orders = simulate(grating_50, a_pix=lambda phi0, phi1: 0.25 if phi1 > phi0 else 0.75)
            assert (orders[1] - orders[-1]) / (orders[1] + orders[-1]) < -0.1

            # Removing the level dependence removes the asymmetry.
            for a_pix in (0.25, 0.75):
                orders = simulate(grating_50, a_pix=a_pix)
                assert np.isclose(orders[1], orders[-1], rtol=1e-12)

        with subtests.test("a duty cycle other than 50% does expose it"):
            assert np.isclose(*simulate(grating_25, a_pix=0.5)[[1, -1]], rtol=1e-12)
            for kwargs in ({"a_minus_pix": 0.05}, {"n_minus": 4}):
                orders = simulate(grating_25, a_pix=0.5, **kwargs)
                assert np.abs((orders[1] - orders[-1]) / (orders[1] + orders[-1])) > 0.01

        with subtests.test("mirroring the kernel mirrors the first orders"):
            orders = simulate(grating_25, a_pix=0.5, n=1, a_minus_pix=0.05, n_minus=4)
            mirror = simulate(grating_25, a_pix=0.05, n=4, a_minus_pix=0.5, n_minus=1)
            assert np.isclose(orders[1], mirror[-1], rtol=1e-12)
            assert np.isclose(orders[-1], mirror[1], rtol=1e-12)

        with subtests.test("plots"):
            simulate(grating_25, a_pix=0.5, a_minus_pix=0.05, plot=True)

    def test_settle_calibration_process(self, fourierslm, caplog, subtests):
        """The settle fit returns positive, in-range times matching a planted response."""
        times = np.linspace(0, 0.2, 201)

        for communication, relaxation in ((0.03, 0.02), (0.01, 0.005), (0.05, 0.01)):
            with subtests.test(f"communication={communication} relaxation={relaxation}"):
                fourierslm.calibrations["settle"] = {
                    "times": times,
                    "data": np.where(
                        times >= communication,
                        1 - np.exp(-(times - communication) / relaxation),
                        0,
                    ),
                }
                result = fourierslm.settle_calibration_process(plot=0)

                assert 0 < result["settle_time"] <= np.max(times)
                assert result["communication_time"] == pytest.approx(
                    communication, abs=2 * np.diff(times)[0]
                )
                assert result["relax_time"] == pytest.approx(relaxation, rel=0.02)
                assert result["settle_time"] == pytest.approx(
                    communication + 4 * relaxation, rel=0.02
                )

        with subtests.test("a flat frame has no step to fit"):
            fourierslm.calibrations["settle"] = {"times": times, "data": np.zeros_like(times)}
            with pytest.raises(RuntimeError, match="no signal"):
                fourierslm.settle_calibration_process()

        with subtests.test("a response slower than the sweep is reported as unresolved"):
            fourierslm.calibrations["settle"] = {
                "times": times,
                "data": 1 - np.exp(-times / (10 * np.ptp(times))),
            }
            with caplog.at_level(logging.WARNING):
                result = fourierslm.settle_calibration_process(plot=1)

            assert result["settle_time"] > np.max(times)
            assert "outside the swept range" in caplog.text

    def test_settle_calibrate(self, fourierslm_calibrated, monkeypatch, subtests):
        """The sweep writes both the raw response and the fit into the calibration."""
        fs = fourierslm_calibrated
        fs.cam.set_exposure(0.1)
        # The fit has four parameters, so fewer probed times leave it undetermined.
        times = [0, 0.005, 0.01, 0.015, 0.02]
        settles = []
        set_phase = fs.slm.set_phase

        def spy(phase, *args, settle=None, **kwargs):
            settles.append(settle)
            return set_phase(phase, *args, settle=settle, **kwargs)

        monkeypatch.setattr(fs.slm, "set_phase", spy)

        calibration = fs.settle_calibrate(times=times, plot=-1)

        with subtests.test("the raw response and the fit are stored"):
            assert calibration is fs.calibrations["settle"]
            np.testing.assert_allclose(calibration["times"], times)
            assert len(calibration["data"]) == len(times)
            assert np.all(calibration["data"] > 0), "the blazed spot should reach the camera"
            assert {"settle_time", "relax_time", "communication_time"} <= set(calibration)

        with subtests.test("every reset settles for the longest probed time"):
            assert set(settles) == {max(times), False}

        with subtests.test("the camera exposure is restored"):
            assert fs.cam.get_exposure() == pytest.approx(0.1)

    def test_fourier_calibrate_geometries(
        self,
        simulated_system,
        simulated_system_name,
        simulated_system_source,
        calibration_results,
        calibration_plot_level,
        request,
        caplog,
    ):
        """fourier_calibrate with a fixed array, across every simulated geometry."""
        (error, tolerance, note, failure) = _run_calibration(
            simulated_system,
            simulated_system_name,
            simulated_system_source,
            calibration_results,
            calibration_plot_level,
        )

        if simulated_system_name not in DEFAULT_CALIBRATION_OK:
            # Strict, so a geometry that starts calibrating is reported rather than
            # staying quietly green in the expected-failure column.
            request.node.add_marker(
                pytest.mark.xfail(
                    strict=True,
                    reason=(
                        f"Fixed array_shape/array_pitch produce a wrong or failed "
                        f"calibration for this geometry ({note})."
                    ),
                )
            )

        if failure is not None:
            raise failure
        assert error < tolerance, (
            f"Calibrated mapping is {error:.2f} px off ground truth (tolerance {tolerance:.2f})."
        )

        # An array with unequal sides may be mis-oriented only with a warning.
        n = simulated_system.calibrations["fourier"]["array"]["array_shape"][0]
        for shape in ((n - 2, n), (n - 1, n + 1)):
            seed_for(f"orientation-{simulated_system_name}-{shape}")
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                simulated_system.fourier_calibrate(
                    array_shape=shape, array_pitch=BATTERY_ARRAY_PITCH, plot=False
                )
            M = simulated_system.calibrations["fourier"]["M"]
            R = np.rint(np.linalg.solve(ground_truth_affine(simulated_system)[0], M))
            assert np.array_equal(R, np.eye(2)) or "not verified" in caplog.text, (
                f"A {shape} array is silently mis-oriented by {R.tolist()}."
            )


@pytest.mark.parametrize("name", GEOMETRY_CASES)
def test_simulated_blaze_lands_at_prediction(simulated_system_factory, name):
    """A blaze, and the 0th order it degenerates to, land where the truth predicts."""
    fs = simulated_system_factory(name)
    tolerance = max(3.0, 2 * np.max(spot_size_ij(fs)))

    for kxy in (np.zeros((2, 1)), in_view_kxy(fs, frac=0.5)):
        predicted = np.squeeze(ground_truth_kxy_to_ij(fs, kxy))
        if np.any(predicted < 0) or np.any(predicted >= np.flip(fs.cam.shape)):
            continue  # Steered off the sensor, so there is nothing to find.

        fs.slm.set_phase(blaze(fs.slm, vector=kxy))
        fs.cam.autoexpose(verbose=False)
        img = fs.cam.get_image().astype(float)
        peak = np.flip(np.unravel_index(np.argmax(img), img.shape))

        assert np.linalg.norm(peak - predicted) < tolerance, (
            f"peak {peak} is {np.linalg.norm(peak - predicted):.2f} px from "
            f"predicted {predicted} (tolerance {tolerance:.2f})."
        )


@pytest.mark.parametrize("name", GEOMETRY_CASES)
def test_simulated_speckle_confined_to_farfield(simulated_system_factory, name):
    """Random phase fills the farfield, and the k-space limits crop it there."""
    fs = simulated_system_factory(name)
    mask = farfield_support_mask(fs)

    # Dilated, so spot-sized blur at the aperture edge does not count as escape.
    blur = int(np.ceil(2 * np.max(spot_size_ij(fs)))) * 2 + 1
    outside = ~(cv2.dilate(mask.astype(np.uint8), np.ones((blur, blur), np.uint8)) > 0)
    if fs.cam.noise is not None or not outside.any():
        pytest.skip("noise, or no camera outside the farfield, leaves nothing to compare")

    seed_for(name)
    fs.slm.set_phase(np.random.uniform(0, 2 * np.pi, fs.slm.shape))
    fs.cam.autoexpose(verbose=False)
    img = fs.cam.get_image().astype(float)

    assert img[mask].mean() > 100 * img[outside].mean(), "Speckle escaped the farfield."
