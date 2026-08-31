"""
Unit tests for FourierSLM and the calibrations mixed into it.
"""
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import ndimage

from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.holography import analysis
from slmsuite.holography.algorithms import SpotHologram
from slmsuite.holography.toolbox import convert_vector
from slmsuite.holography.toolbox.phase import blaze, zernike_sum

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


# Enough geometry to exercise the simulator: a contained farfield with aperture edges in
# view, a cropped one, rotation, and a 0th order steered off the sensor.
GEOMETRY_CASES = ("matched", "fov_much_larger", "rotated", "zeroth_outside")

# Geometries whose calibration depends on the random draw. A single seed hides that:
# what passes at one draw can be silently wrong at the next.
MARGINAL_CASES = ("offset", "zeroth_corner", "zeroth_outside", "anisotropic")

# A farfield that fills the camera, one contained inside it, and one both rotated and
# cropped: the three ways the measured efficiency map meets the aperture.
FARFIELD_CASES = ("matched", "fov_larger", "rotated")

BATTERY_ARRAY_SHAPE = 10
BATTERY_ARRAY_PITCH = 10

# Cases where fourier_calibrate with a fixed array recovers the correct affine. The
# remaining cases silently produce a *wrong* calibration (or raise) --- this is the
# motivation for fourier_calibrate_auto().
DEFAULT_CALIBRATION_OK = {
    "identity", "matched", "fov_much_larger", "fov_larger",
    "mirrored", "pitch_anisotropic", "defocus",
}

# Geometries that fourier_calibrate_auto() cannot yet handle, and why. Keyed by case
# name, or by (case, source) where only one illumination is affected. All but the last
# refuse rather than returning a calibration, so the failure is loud.
AUTO_LIMITATIONS = {
    "zeroth_outside": (
        "With the 0th order off-camera the array can only be placed relative to the "
        "lit region, and every array that fits there is fitted wrongly, which "
        "verification rejects"
    ),
    "fov_extreme": (
        "The survey measures one scale for both axes, which a farfield three times "
        "larger than the camera along one axis and three times smaller along the "
        "other is not described by"
    ),
    ("noisy_severe", "gaussian2d"): (
        "Severe noise on an apodized source leaves an unbounded fit residual, which the "
        "calibration refuses rather than returning; the same noise on a uniform source "
        "calibrates"
    ),
    # TODO: fourier_calibrate_auto should be able to move the center of the array.
    ("zeroth_corner", "gaussian2d"): (
        "A 0th order 2% of the way across the sensor leaves no room to place an array "
        "around it, so most of it falls off the camera and too few spots remain to fit"
    ),
    ("noisy", "gaussian2d"): (
        "A marginal draw rather than a broken geometry: most draws of this case "
        "calibrate, and the seeded draw of this battery lands just outside tolerance"
    ),
}


def _binary_grating(period, a, b, duty_cycle=.5):
    """One period of a binary grating, as per-pixel phase in radians."""
    return np.where(np.arange(period) < round(period * duty_cycle), a, b)


def _ground_truth_error(fs):
    """
    Largest disagreement between the installed calibration and the ground-truth affine,
    together with the tolerance it is held to, both in camera pixels. Measured over the
    area the camera views, not along one ray through its center: a ray is blind to any
    error orthogonal to itself.
    """
    grid = view_kxy_grid(fs)
    error = float(np.max(np.linalg.norm(
        fs.kxyslm_to_ijcam(grid) - ground_truth_kxy_to_ij(fs, grid), axis=0
    )))
    return (error, max(2.0, np.max(spot_size_ij(fs))))


@pytest.fixture(scope="module")
def calibration_plot_level(request):
    """
    Plot level for the calibration batteries: 2 emits the diagnostics that explain *why*
    a geometry fails, which cost more to render than the calibration costs to run.
    """
    return 2 if request.config.getoption("--save-plots") else 0


@pytest.fixture(scope="module")
def calibration_summaries(test_output_dir, request):
    """Per-routine accumulators, each written at teardown as one montage of every case."""
    summaries = {}
    yield summaries
    for (routine, results) in summaries.items():
        _write_summary(results, test_output_dir, request, routine)


def _write_summary(results, test_output_dir, request, routine):
    """Montage of every case's calibration against the ground truth."""
    if not results or test_output_dir is None:
        return
    if not request.config.getoption("--save-plots"):
        return

    columns = 4
    rows = int(np.ceil(len(results) / columns))
    (fig, axs) = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axs = np.atleast_1d(axs).ravel()

    for (ax, (name, result)) in zip(axs, results.items()):
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
            color=color, fontsize="medium",
        )
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axs[len(results):]:
        ax.axis("off")

    fig.suptitle(
        f"{routine} vs ground truth\n"
        f"circles = true spot positions, crosses = calibrated prediction"
    )
    fig.tight_layout()
    fig.savefig(test_output_dir / f"{routine}_summary.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved summary: {test_output_dir / routine}_summary.png")


def _run_calibration(fs, case, source, calibrate, summaries, routine):
    """
    Calibrates, compares against ground truth, and records the diagnostics.

    Failures are returned rather than raised: the geometries that defeat a calibration
    are the ones whose plots are worth having. Returns the error in camera pixels
    (infinite if nothing was produced), the tolerance, a description, and any exception.
    """
    seed_for(case)      # The hologram starts from a random phase.
    name = f"{case}-{source or 'uniform'}"

    failure = None
    try:
        calibrate(fs)
    except Exception as e:
        failure = e

    (error, tolerance) = (np.inf, max(2.0, np.max(spot_size_ij(fs))))
    if failure is None:
        (error, tolerance) = _ground_truth_error(fs)

    # The array that the calibration settled on, for the diagnostic overlays.
    array = fs.calibrations.get("fourier", {}).get("array", {})
    spots_kxy = array_kxy(
        fs,
        array.get("array_shape", BATTERY_ARRAY_SHAPE),
        array.get("array_pitch", BATTERY_ARRAY_PITCH),
        array.get("array_center"),
    )
    note = (
        f"{type(failure).__name__} raised" if failure is not None
        else f"max error {error:.2f} px (tolerance {tolerance:.2f} px)"
    )
    img = fs.cam.last_image if fs.cam.last_image is not None else fs.cam.get_image()
    plot_calibration_diagnostic(
        fs, img=img, spots_kxy=spots_kxy, name=f"{routine}_{name}", note=note,
    )

    summaries.setdefault(routine, {})[name] = {
        "img": img,
        "truth": ground_truth_kxy_to_ij(fs, spots_kxy),
        "calibrated": (
            fs.kxyslm_to_ijcam(spots_kxy) if "fourier" in fs.calibrations else None
        ),
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
            with pytest.raises(ValueError):
                fourierslm.fourier_calibrate(array_pitch=-1, array_shape=5, plot=False)

    @pytest.mark.slow
    def test_fourier_calibrate_noise(self, slm, subtests):
        """Camera noise and a rotated sensor still calibrate to the ground truth."""
        noise = {
            "dark": lambda img: np.random.normal(0.005 * img, 0.002 * img),
            "read": lambda img: np.random.poisson(0.03 * img),
        }
        for (theta, nz) in [(0.0, None), (0.0, noise), (0.2, noise), (-0.3, noise)]:
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
            with pytest.raises(ValueError):
                fourierslm.fourier_calibrate_analytic(np.eye(3), b)

    def test_fourier_grid_project(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.fourier_grid_project."""
        hologram = fourierslm_calibrated.fourier_grid_project(array_shape=3, array_pitch=35)
        img = fourierslm_calibrated.cam.get_image()

        with subtests.test("two spots are omitted as an orientation check"):
            assert hologram.spot_kxy_rounded.shape == (2, 3 * 3 - 2)

        with subtests.test("every projected spot is among the brightest pixels"):
            ij = np.rint(hologram.spot_ij).astype(int)
            assert np.min(img[ij[1], ij[0]]) > 0.5 * np.max(img)

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
                for (rot, binning, woi) in [
                    ("0", 1, None),
                    ("90", 1, None),
                    ("180", 2, None),
                    ("0", 2, (20, 60, 30, 50)),
                ]:
                    fs.cam.transform = analysis.get_orientation_transformation(rot)
                    fs.cam.set_binning(binning)
                    fs.cam.set_woi(woi)

                    size = fs.get_farfield_spot_size((Wx, Wy), basis="ij")
                    expected = np.sqrt(np.abs(fs.fourier_affine.det())) * np.array(
                        [1 / Wx, 1 / Wy]
                    )
                    assert np.allclose(size, expected), (
                        f"rot={rot} binning={binning} woi={woi}: {size} != {expected}"
                    )
            finally:
                fs.cam.transform = analysis.get_orientation_transformation("0")
                fs.cam.set_binning(1)
                fs.cam.set_woi(None)

        with subtests.test("bad basis raises"):
            with pytest.raises(ValueError):
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
            with pytest.raises(ValueError):
                fourierslm_calibrated.get_camera_extent(units="kxy", return_mask=True)

    def test_simulate(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.simulate."""

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

    def test_save_load_calibration(self, fourierslm_calibrated, temp_dir, subtests):
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

        with subtests.test("saving a calibration that was never taken raises"):
            with pytest.raises(ValueError):
                fourierslm_calibrated.save_calibration("nonexistent", path=temp_dir)

    def test_load(self, fourierslm_calibrated, temp_dir, subtests):
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
            assert np.allclose(
                fs.kxyslm_to_ijcam(kxy), fourierslm_calibrated.kxyslm_to_ijcam(kxy)
            )
            assert np.allclose(fs.simulate().kxyslm_to_ijcam(kxy), fs.kxyslm_to_ijcam(kxy))

    def test_plot(self, fourierslm):
        """Test FourierSLM.plot, which shows the nearfield beside the farfield."""
        assert len(fourierslm.plot(phase=blaze(fourierslm.slm, vector=(1e-3, 2e-3)))) == 2
        assert len(fourierslm.plot(image=fourierslm.cam.get_image())) == 2

    @pytest.mark.slow
    def test_wavefront_calibrate_superpixel(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.wavefront_calibrate_superpixel against a planted aberration."""
        fs = fourierslm_calibrated
        cal_point = [150, 150]
        sp_size = fs.slm.shape[0] // 6

        # An aberrated source, so that there is something to measure.
        fs.slm.set_source_analytic(
            phase_offset=zernike_sum(
                fs.slm, indices=(3, 4, 5, 7, 8), weights=(1, -2, 3, 1, 1),
                aperture=None, use_mask=False,
            ),
            sim=True,
        )
        fs.cam.set_exposure(.1)

        with subtests.test("interference happens where the blaze lands"):
            fs.slm.set_phase(blaze(fs.slm, vector=fs.ijcam_to_kxyslm(cal_point)))
            img = fs.cam.get_image()
            assert img[140:160, 140:160].mean() > img.mean()

        for (phase_steps, name) in [(None, "amplitude only"), (1, "one shot"), (5, "many shot")]:
            fs.slm.source["phase"] = None    # Clear any previous calibration.

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
                assert error < .5, f"amplitude is {error:.2f} off the simulated source."

        with subtests.test("smoothing the correction"):
            fs.wavefront_calibration_superpixel_process(smooth=True, plot=True)

        with subtests.test("test_index measures one point and restores the source"):
            before = np.array(fs.slm.source["amplitude"])
            result = fs.wavefront_calibrate_superpixel(
                calibration_points=cal_point,
                superpixel_size=sp_size,
                phase_steps=5,
                test_index=-2,
                plot=-1,
            )
            assert "power" in result
            assert np.array_equal(fs.slm.source["amplitude"], before)

        with subtests.test("requires a Fourier calibration"):
            with pytest.raises((RuntimeError, KeyError)):
                FourierSLM(fs.cam, fs.slm).wavefront_calibrate_superpixel(
                    calibration_points=cal_point, superpixel_size=sp_size, plot=-1,
                )

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

    def test_wavefront_calibration_superpixel_process(self, simulated_system_factory, subtests):
        """
        Superpixels below the r2 threshold are filled in from their neighbours, so a pure
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
            for (hx, hy) in holes:
                r2[hy, hx] = 0
            return {
                "NX": NX, "NY": NY, "nxref": NX // 2, "nyref": NY // 2,
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
                truth = 2 * np.pi * (
                    vector[0] * np.asarray(x_grid) + vector[1] * np.asarray(y_grid)
                )
                fs.calibrations["wavefront_superpixel"] = ramp_data(vector)
                phase = fs.wavefront_calibration_superpixel_process(
                    smooth=False,
                    r2_threshold=.5,
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
            result = fourierslm_calibrated.wavefront_calibrate_zernike(
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

    @pytest.mark.parametrize("name", FARFIELD_CASES)
    def test_farfield_calibrate(self, simulated_system_factory, name, subtests):
        """Test FourierSLM.farfield_calibrate, which fills the farfield with speckle and
        captures the 0th order with a flat phase pattern."""
        fs = simulated_system_factory(name)
        install_ground_truth_calibration(fs)

        with subtests.test("stores one raw frame per requested realization"):
            cal = fs.farfield_calibrate(averaging=3)
            assert cal["efficiency_raw"].shape == (3,) + fs.cam.shape
            assert cal["exposure_zeroth"] > 0 and cal["exposure_raw"] > 0

        with subtests.test("captures the 0th order at the offset of the affine"):
            zeroth = fs.calibrations["farfield"]["zeroth"]
            peak = np.flip(np.unravel_index(np.argmax(zeroth), zeroth.shape))
            b = np.squeeze(ground_truth_affine(fs)[1])
            assert np.linalg.norm(peak - b) < max(3.0, 2 * np.max(spot_size_ij(fs)))

    @pytest.mark.parametrize("name", FARFIELD_CASES)
    def test_farfield_calibration_process(self, simulated_system_factory, name, subtests):
        """Test FourierSLM.farfield_calibration_process."""
        fs = simulated_system_factory(name)
        install_ground_truth_calibration(fs)
        # Ten speckle realizations, blurred wider than their grain, else this maps speckle.
        fs.farfield_calibrate()
        efficiency = fs.farfield_calibration_process(size_blur=5)
        mask = fs.get_farfield_extent(return_mask=True)
        # Erode and dilate to stay clear of speckle blur at the aperture edge.
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

        with subtests.test("normalized to unity at the brightest usable pixel"):
            # The 0th order is excluded from the normalization, so it may exceed unity.
            (yy, xx) = np.indices(fs.cam.shape)
            b = np.squeeze(ground_truth_affine(fs)[1])
            usable = np.hypot(xx - b[0], yy - b[1]) > 5 * np.max(spot_size_ij(fs))

            assert efficiency.shape == fs.cam.shape
            assert np.nanmax(efficiency[usable]) == pytest.approx(1.0)
            assert np.array_equal(efficiency, fs.get_farfield_efficiency(fourier_crop=False))

        with subtests.test("dark outside the farfield the SLM can address"):
            inside = efficiency[eroded].mean()
            assert inside > 0, "the efficiency map is not illuminated."
            if (~dilated).any():
                outside = efficiency[~dilated].mean()
                assert outside < 0.1 * inside, (
                    f"efficiency outside the aperture ({outside:.3f}) should be far "
                    f"below inside ({inside:.3f})."
                )

        with subtests.test("speckle averages away inside the farfield"):
            values = efficiency[eroded]
            cv = np.std(values) / np.mean(values)
            assert cv < 0.5, f"the averaged efficiency map is too nonuniform (CV={cv:.2f})."

        with subtests.test("raises without raw data, which is not saved to file"):
            with pytest.raises(RuntimeError):
                FourierSLM(fs.cam, fs.slm).farfield_calibration_process()

    def test_farfield_calibration_process_saturation(self, simulated_system_factory, subtests):
        """A railed pixel must not set the efficiency normalization."""
        fs = simulated_system_factory("matched")
        fs.farfield_calibrate(averaging=3)

        # Averaging sums frames, so it lifts saturation as well as the bitresolution.
        fs.cam.averaging = 16
        saturation = fs.cam.bitresolution - fs.cam.averaging

        efficiency = fs.calibrations["farfield"]["efficiency"]
        (y, x) = np.unravel_index(np.argmin(np.abs(efficiency - 0.5)), fs.cam.shape)
        raw = fs.calibrations["farfield"]["efficiency_raw"]
        resolved = raw[:, y, x].copy()

        def peak_away_from(y, x):
            """The efficiency peak ignoring one pixel and everything the blur touched."""
            efficiency = np.array(fs.get_farfield_efficiency(fourier_crop=False))
            efficiency[max(0, y-2):y+3, max(0, x-2):x+3] = 0
            return np.nanmax(efficiency)

        with subtests.test("a saturated pixel does not set the peak"):
            raw[:, y, x] = saturation
            fs.farfield_calibration_process(size_blur=3)
            assert peak_away_from(y, x) == pytest.approx(1.0)

        with subtests.test("and is reported above unity"):
            assert fs.get_farfield_efficiency(fourier_crop=False)[y, x] > 1

        with subtests.test("saturating one realization is enough to exclude it"):
            raw[:, y, x] = resolved
            raw[0, y, x] = saturation
            fs.farfield_calibration_process(size_blur=3)
            assert peak_away_from(y, x) == pytest.approx(1.0)

        with subtests.test("full saturation falls back to the whole frame"):
            raw[:] = saturation
            efficiency = fs.farfield_calibration_process(size_blur=3)
            assert np.all(np.isfinite(efficiency))
            assert np.nanmax(efficiency) == pytest.approx(1.0)

    @pytest.mark.parametrize("name", ("fov_much_larger", "fov_larger"))
    def test_get_farfield_efficiency(self, simulated_system_factory, name, subtests):
        """Test FourierSLM.get_farfield_efficiency: the blind support measurement that
        fourier_calibrate_auto() bootstraps from, and the crops applied to it."""
        fs = simulated_system_factory(name)
        fs.farfield_calibrate(averaging=3)
        # Smoothing stays below the farfield itself, tens of pixels across when cropped.
        fs.farfield_calibration_process(size_blur=3)

        with subtests.test("the thresholded support matches the ground truth"):
            assert "fourier" not in fs.calibrations, "the support is measured blind."
            support = fs.get_farfield_efficiency(efficiency_threshold=0.1)
            expected = farfield_support_mask(fs)
            iou = (support & expected).sum() / (support | expected).sum()
            assert iou > 0.8, f"support IoU {iou:.2f} too low."

        with subtests.test("fourier_crop masks off everything outside the extent"):
            install_ground_truth_calibration(fs)
            mask = fs.get_farfield_extent(return_mask=True)
            cropped = fs.get_farfield_efficiency(fourier_crop=True)
            whole = fs.get_farfield_efficiency(fourier_crop=False)
            assert np.array_equal(cropped[mask], whole[mask])
            assert (~mask).any() and np.all(np.isnan(cropped[~mask]))

        with subtests.test("raises before the calibration is processed"):
            with pytest.raises(RuntimeError):
                FourierSLM(fs.cam, fs.slm).get_farfield_efficiency()

    def test_farfield_products(self, simulated_system_factory, temp_dir, subtests):
        """The zeroth order, background, and file products of a farfield calibration:
        get_farfield_zeroth, get_farfield_background, and save_calibration."""
        fs = simulated_system_factory("fov_larger")
        install_ground_truth_calibration(fs)
        fs.farfield_calibrate(averaging=3)

        with subtests.test("the background is bounded by the frames it came from"):
            background = fs.get_farfield_background()
            assert background.shape == fs.cam.shape
            assert np.all(background >= 0)
            raw = np.asarray(fs.calibrations["farfield"]["efficiency_raw"], float)
            assert np.nanmax(background) <= np.nanmax(np.mean(raw, axis=0))

        with subtests.test("raw stacks are not saved to file"):
            path = fs.save_calibration("farfield", path=temp_dir, name="test_farfield")
            fs.load_calibration("farfield", file_path=path)
            loaded = fs.calibrations["farfield"]
            assert "efficiency_raw" not in loaded and "background_raw" not in loaded
            for key in ("efficiency", "background", "zeroth", "exposure_saturating"):
                assert key in loaded

        with subtests.test("a loaded calibration still serves its products"):
            with pytest.raises(RuntimeError):
                fs.farfield_calibration_process()
            assert np.all(np.isfinite(fs.get_farfield_zeroth()))
            assert np.all(np.isfinite(fs.get_farfield_background()))

    def test_get_farfield_weights(self, simulated_system_factory):
        """
        Weights ask for more amplitude where the measured farfield delivers less
        power, are bounded, and fall back to uniform when nothing was measured.
        """
        fs = simulated_system_factory("fov_larger")
        install_ground_truth_calibration(fs)
        fs.farfield_calibrate(averaging=1)
        fs.farfield_calibration_process()

        efficiency = fs.get_farfield_efficiency()
        lit = np.argwhere(np.nan_to_num(efficiency) > 0)
        assert len(lit) > 0, "The farfield calibration measured no efficiency."

        # Brightest and dimmest measured pixels, in (x, y).
        values = efficiency[lit[:, 0], lit[:, 1]]
        ij = np.flip(lit[[np.argmax(values), np.argmin(values)]].T, axis=0)

        weights = fs.get_farfield_weights(ij, np.ones(2))
        assert weights.shape == (2,)
        assert np.all(weights > 0) and np.max(weights) == pytest.approx(1.0)
        assert weights[1] > weights[0], (
            "The dimmer part of the farfield should be asked for more amplitude."
        )
        # Power goes as the square of amplitude, and is floored so that a dead
        # region cannot ask for unbounded power.
        assert np.max(weights) / np.min(weights) <= 1 / np.sqrt(0.05) + 1e-9

        # An unmeasured farfield weights every spot the same rather than by NaN.
        fs.calibrations["farfield"]["efficiency"] = np.zeros_like(efficiency)
        assert np.array_equal(fs.get_farfield_weights(ij, np.ones(2)), np.ones(2))

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
                    levels=levels, periods=1, orders=1, directions="x",
                    test_index=0, plot=False,
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
            # True tests every point of the sweep; False is not a test at all.
            for true in (True, np.True_):        # numpy booleans are not bool instances.
                result = fs.pixel_calibrate(
                    levels=2, periods=1, orders=1, directions="x", test_index=true, plot=False
                )
                assert len(result["indices"]) == 4
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

        with subtests.test("plots"):
            fs.pixel_calibrate(levels=2, periods=1, orders=2, directions="x", plot=2)
            fs.pixel_calibration_plot(summed=True)
            fs.pixel_calibration_plot(orders=[1])   # A single order must not squeeze away.
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
            assert np.allclose(gamma, expected - np.min(expected), atol=.02)
            # The fit is seeded with the linear response, so agreement alone proves
            # nothing; the fit must also actually describe the data.
            assert fs.calibrations["pixel"]["gamma_r2"] > .9

        with subtests.test("the sampled fit is applied across every level"):
            B = fs.slm.bitresolution
            # The sweep measures a handful of levels; the SLM carries all of them.
            assert len(gamma) < B
            assert fs.slm.gamma.shape == (B,)
            np.testing.assert_allclose(fs.slm.gamma, np.arange(B) / B, atol=.02)

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
            assert np.allclose(gamma, expected, atol=.02)
            assert not np.allclose(gamma, levels / B, atol=.02)   # Not the ideal ramp.

        with subtests.test("the table corrects the response"):
            target = np.arange(B) * (2 * np.pi / B)

            def error(levels):
                realized = np.mod(fs.slm._gamma_sign * 2 * np.pi * truth[levels], 2 * np.pi)
                deviation = (realized - target + np.pi) % (2 * np.pi) - np.pi
                return np.sqrt(np.mean(np.square(deviation)))

            uncorrected = error((-np.rint(target * B / (2 * np.pi)).astype(int)) % B)
            corrected = error(
                np.asarray(fs.slm.lut)[np.floor(target * fs.slm._phase_to_lut).astype(int)]
            )
            assert corrected < uncorrected / 8

        with subtests.test("a response beyond one cycle is unwrapped"):
            # The fit resolves each level only modulo a cycle.
            truth = 2.4 * np.arange(B) / B
            fs.slm.gamma_sim = truth
            fs.pixel_calibrate(levels=16, periods=2, orders=1, directions="x", plot=False)
            gamma = fs.pixel_calibration_process(plot=False)

            levels = fs.calibrations["pixel"]["levels"].astype(int)
            assert np.max(gamma) > 1                      # More than one cycle recovered.
            assert np.allclose(gamma, truth[levels] - np.min(truth[levels]), atol=.02)

        with subtests.test("a mirrored fit is returned on the increasing branch"):
            assert np.all(np.diff(gamma) > 0)

        with subtests.test("levels are canonically ordered"):
            # Unwrapping the fit follows the order of the levels it was measured at.
            fs.pixel_calibrate(
                levels=np.array([96, 0, 192, 32]), periods=2, orders=1,
                directions="x", plot=False,
            )
            stored = fs.calibrations["pixel"]["levels"]
            np.testing.assert_array_equal(stored, np.sort(stored))

    def test_pixel_kernel(self, subtests):
        """The pixel kernel is normalized, peaked, and shaped by its two sides."""
        x = (np.arange(401) - 200) / 50    # +-4 pixels at 50 samples per pixel.

        with subtests.test("normalized"):
            assert np.isclose(np.sum(FourierSLM.pixel_kernel(x, a_pix=.5)), 1)

        with subtests.test("symmetric by default"):
            kernel = FourierSLM.pixel_kernel(x, a_pix=.5, n=2)
            assert np.allclose(kernel, np.flip(kernel))

        with subtests.test("asymmetric widths and exponents"):
            for kwargs in ({"a_minus_pix": .05}, {"n_minus": 4}):
                kernel = FourierSLM.pixel_kernel(x, a_pix=.5, n=1, **kwargs)
                assert not np.allclose(kernel, np.flip(kernel))

        with subtests.test("swapping the two sides mirrors the kernel"):
            kernel = FourierSLM.pixel_kernel(x, a_pix=.5, n=1, a_minus_pix=.05, n_minus=4)
            mirror = FourierSLM.pixel_kernel(x, a_pix=.05, n=4, a_minus_pix=.5, n_minus=1)
            assert np.allclose(kernel, np.flip(mirror))

        with subtests.test("peaked at the origin and decaying"):
            kernel = FourierSLM.pixel_kernel(x, a_pix=.5, n=1, a_minus_pix=.05)
            assert np.argmax(kernel) == np.argmin(np.abs(x))
            assert np.all(np.diff(kernel[x >= 0]) <= 0)
            assert np.all(np.diff(kernel[x <= 0]) >= 0)

        with subtests.test("x0_pix displaces the peak"):
            kernel = FourierSLM.pixel_kernel(x, a_pix=.5, n=2, x0_pix=.4)
            assert np.isclose(x[np.argmax(kernel)], .4)
            assert np.allclose(kernel, FourierSLM.pixel_kernel(x - .4, a_pix=.5, n=2))

    def test_pixel_crosstalk_simulate(self, subtests):
        """Test FourierSLM._pixel_crosstalk_simulate, the forward model of the
        crosstalk kernel's effect on the diffraction orders of a grating."""
        simulate = FourierSLM._pixel_crosstalk_simulate
        (grating_50, grating_25) = (
            _binary_grating(16, np.pi, 0), _binary_grating(16, np.pi, 0, duty_cycle=.25)
        )

        with subtests.test("crosstalk-free limit matches the analytic binary grating"):
            # A 50% duty grating of phase difference d diffracts (2/(pi m))^2 sin^2(d/2)
            # into odd order m, and nothing into even orders.
            for delta in (np.pi, .6 * np.pi):
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
                        supersample=supersample, a_pix=1e-6,
                    )
                    assert np.all(np.isfinite(orders)) and np.isclose(np.sum(orders), 1)

        with subtests.test("kernel is applied on the SLM pixel scale"):
            # Weakly-phased orders report the kernel's transfer function at m/p, which
            # pins a_pix absolutely.
            (delta, period) = (.01, 16)
            for a_pix in (.25, .5):
                for supersample in (32, 64):
                    orders = simulate(
                        _binary_grating(period, delta, 0),
                        supersample=supersample, a_pix=a_pix, n=1,
                    )
                    for m in (1, 3):
                        transfer = 1 / (1 + (2 * np.pi * m * a_pix / period) ** 2)
                        assert np.isclose(
                            orders[m],
                            (delta / (np.pi * m)) ** 2 * transfer ** 2,
                            rtol=1e-2,
                        )

        with subtests.test("constant parameters reduce to a convolution"):
            # Moser Eq. (12) with fixed parameters is exactly the usual crosstalk model.
            # The reference tiles the grating, so that its kernel is untruncated too.
            (supersample, tiles) = (32, 5)
            commanded = np.repeat(np.tile(grating_50, tiles), supersample)
            size = commanded.size - 1
            for kwargs in ({"a_minus_pix": .05}, {"n_minus": 4}, {"a_pix": 2.}):
                kwargs = {"a_pix": .5, **kwargs}
                kernel = FourierSLM.pixel_kernel(
                    (np.arange(size) - (size - 1) / 2) / supersample, **kwargs
                )
                blurred = ndimage.convolve1d(commanded, kernel, mode="grid-wrap")
                expected = np.square(np.abs(
                    np.fft.fft(np.exp(1j * blurred)) / commanded.size
                ))
                orders = simulate(grating_50, supersample=supersample, **kwargs)
                assert np.allclose(orders[:9], expected[::tiles][:9], atol=1e-12)

        with subtests.test("independent of how many periods are supplied"):
            # The kernel's support must be set by the kernel, not by the pattern.
            for kwargs in ({"a_pix": .5, "a_minus_pix": .05}, {"a_pix": 2.}):
                for period in (4, 16):
                    grating = _binary_grating(period, np.pi, 0, duty_cycle=.25)
                    one = simulate(grating, supersample=64, **kwargs)
                    four = simulate(np.tile(grating, 4), supersample=64, **kwargs)
                    assert np.allclose(one[:5], four[::4][:5], atol=1e-12)

        with subtests.test("50% duty gratings cannot show a constant kernel's asymmetry"):
            # phi(x + p/2) = a + b - phi(x) survives any constant kernel and forces
            # |E_m| = |E_-m|.  This is the blind spot of pixel_calibrate().
            for kwargs in ({"a_minus_pix": .05}, {"n_minus": 4}):
                for delta in (np.pi, .6 * np.pi):
                    orders = simulate(_binary_grating(16, delta, 0), a_pix=.5, **kwargs)
                    assert np.isclose(orders[1], orders[-1], rtol=1e-12)

        with subtests.test("level-dependent width breaks the 50% duty symmetry"):
            # Moser's mechanism: LC pre-tilt steepens the transition for one sign of the
            # level step and flattens it for the other, which no convolution can do.
            # Signed, so that swapping the (phi0, phi1) convention fails.
            orders = simulate(grating_50, a_pix=lambda phi0, phi1: .25 if phi1 > phi0 else .75)
            assert (orders[1] - orders[-1]) / (orders[1] + orders[-1]) < -.1

            # Removing the level dependence removes the asymmetry.
            for a_pix in (.25, .75):
                orders = simulate(grating_50, a_pix=a_pix)
                assert np.isclose(orders[1], orders[-1], rtol=1e-12)

        with subtests.test("a duty cycle other than 50% does expose it"):
            assert np.isclose(*simulate(grating_25, a_pix=.5)[[1, -1]], rtol=1e-12)
            for kwargs in ({"a_minus_pix": .05}, {"n_minus": 4}):
                orders = simulate(grating_25, a_pix=.5, **kwargs)
                assert np.abs((orders[1] - orders[-1]) / (orders[1] + orders[-1])) > .01

        with subtests.test("mirroring the kernel mirrors the first orders"):
            orders = simulate(grating_25, a_pix=.5, n=1, a_minus_pix=.05, n_minus=4)
            mirror = simulate(grating_25, a_pix=.05, n=4, a_minus_pix=.5, n_minus=1)
            assert np.isclose(orders[1], mirror[-1], rtol=1e-12)
            assert np.isclose(orders[-1], mirror[1], rtol=1e-12)

        with subtests.test("plots"):
            simulate(grating_25, a_pix=.5, a_minus_pix=.05, plot=True)

    def test_settle_calibration_process(self, fourierslm, subtests):
        """The settle fit returns positive, in-range times matching a planted response."""
        times = np.linspace(0, .2, 201)

        for (communication, relaxation) in ((.03, .02), (.01, .005), (.05, .01)):
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
                assert result["relax_time"] == pytest.approx(relaxation, rel=.02)
                assert result["settle_time"] == pytest.approx(
                    communication + 4 * relaxation, rel=.02
                )

    def test_fourier_calibrate_geometries(
        self, simulated_system, simulated_system_name, simulated_system_source,
        calibration_summaries, calibration_plot_level, request,
    ):
        """fourier_calibrate with a fixed array, across every simulated geometry."""
        (error, tolerance, note, failure) = _run_calibration(
            simulated_system, simulated_system_name, simulated_system_source,
            lambda system: system.fourier_calibrate(
                array_shape=BATTERY_ARRAY_SHAPE, array_pitch=BATTERY_ARRAY_PITCH,
                plot=calibration_plot_level, verbose=False,
            ),
            calibration_summaries, "fourier_calibrate",
        )

        if simulated_system_name not in DEFAULT_CALIBRATION_OK:
            auto_fails_too = AUTO_LIMITATIONS.get(
                (simulated_system_name, simulated_system_source),
                AUTO_LIMITATIONS.get(simulated_system_name),
            )
            # Strict, so a geometry that starts calibrating is reported rather than
            # staying quietly green in the expected-failure column.
            request.node.add_marker(pytest.mark.xfail(
                strict=True,
                reason=(
                    f"Fixed array_shape/array_pitch produce a wrong or failed "
                    f"calibration for this geometry ({note}); "
                    + (
                        "fourier_calibrate_auto() does not rescue it either."
                        if auto_fails_too else "fourier_calibrate_auto() handles it."
                    )
                ),
            ))

        if failure is not None:
            raise failure
        assert error < tolerance, (
            f"Calibrated mapping is {error:.2f} px off ground truth "
            f"(tolerance {tolerance:.2f})."
        )

    def test_fourier_calibrate_auto(
        self, simulated_system, simulated_system_name, simulated_system_source,
        calibration_summaries, calibration_plot_level, request,
    ):
        """fourier_calibrate_auto, which chooses its own array, across every geometry."""
        (error, tolerance, note, failure) = _run_calibration(
            simulated_system, simulated_system_name, simulated_system_source,
            lambda system: system._fourier_calibrate_auto(plot=calibration_plot_level),
            calibration_summaries, "fourier_calibrate_auto",
        )

        limitation = AUTO_LIMITATIONS.get(
            (simulated_system_name, simulated_system_source),
            AUTO_LIMITATIONS.get(simulated_system_name),
        )
        if limitation is not None:
            request.node.add_marker(
                pytest.mark.xfail(strict=True, reason=f"{limitation} ({note})")
            )

        if failure is not None:
            raise failure
        assert error < tolerance, (
            f"fourier_calibrate_auto is {error:.2f} px off ground truth "
            f"(tolerance {tolerance:.2f})."
        )

    def test_fourier_calibrate_auto_failure(
        self, simulated_system_factory, monkeypatch, subtests
    ):
        """
        A calibration that cannot be verified must not be installed, and must not destroy
        a calibration that was already there: the alternative is a system that silently
        uses a calibration its own check rejected, or one that loses a working
        calibration because the beam was blocked during a re-run. Every failure is forced
        rather than found, so that this tests the guarantee rather than whichever
        geometry happens to be failing.
        """
        def calibrated():
            """A system carrying the ground-truth calibration, and that calibration."""
            fs = simulated_system_factory("matched")
            install_ground_truth_calibration(fs)
            return (fs, np.array(fs.calibrations["fourier"]["M"]))

        def raise_injected(*args, **kwargs):
            raise ValueError("injected failure")

        with subtests.test("no light reaches the camera"):
            (fs, existing) = calibrated()
            fs.slm.source["amplitude_sim"] = np.zeros_like(fs.slm.source["amplitude_sim"])
            with pytest.raises(RuntimeError):
                fs._fourier_calibrate_auto()
            assert np.array_equal(fs.calibrations["fourier"]["M"], existing)

        with subtests.test("a calibration that fails verification is not installed"):
            fs = simulated_system_factory("matched")
            with pytest.raises(RuntimeError):
                fs._fourier_calibrate_auto(tolerance=-1)
            assert "fourier" not in fs.calibrations

        with subtests.test("and does not displace the previous calibration"):
            (fs, existing) = calibrated()
            with pytest.raises(RuntimeError):
                fs._fourier_calibrate_auto(tolerance=-1)
            assert np.array_equal(fs.calibrations["fourier"]["M"], existing)

        with subtests.test("an unexpected failure before any array is projected"):
            (fs, existing) = calibrated()
            fs.get_farfield_zeroth = raise_injected
            with pytest.raises(ValueError):
                fs._fourier_calibrate_auto()
            assert np.array_equal(fs.calibrations["fourier"]["M"], existing)

        with subtests.test("an unexpected failure after an array is installed"):
            (fs, existing) = calibrated()
            monkeypatch.setattr(analysis, "_score_array_orientation", raise_injected)
            with pytest.raises(ValueError):
                fs._fourier_calibrate_auto()
            assert np.array_equal(fs.calibrations["fourier"]["M"], existing)


@pytest.mark.parametrize("name", GEOMETRY_CASES)
def test_simulated_blaze_lands_at_prediction(simulated_system_factory, name):
    """A blaze, and the 0th order it degenerates to, land where the truth predicts."""
    fs = simulated_system_factory(name)
    tolerance = max(3.0, 2 * np.max(spot_size_ij(fs)))

    for kxy in (np.zeros((2, 1)), in_view_kxy(fs, frac=0.5)):
        predicted = np.squeeze(ground_truth_kxy_to_ij(fs, kxy))
        if np.any(predicted < 0) or np.any(predicted >= np.flip(fs.cam.shape)):
            continue    # Steered off the sensor, so there is nothing to find.

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


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("name", MARGINAL_CASES)
def test_fourier_calibrate_auto_survives_any_draw(
    simulated_system_factory, simulated_system_source, name, seed
):
    """
    A calibration that passes verification must be correct whatever the random draw.
    Refusing is allowed --- returning a wrong affine that reports a sub-pixel
    residual is the failure this whole routine exists to prevent.
    """
    fs = simulated_system_factory(name, source_function=simulated_system_source)
    np.random.seed(seed)
    try:
        import cupy
        cupy.random.seed(seed)
    except ImportError:
        pass

    try:
        fs._fourier_calibrate_auto()
    except RuntimeError:
        return

    (error, tolerance) = _ground_truth_error(fs)
    assert error < tolerance, (
        f"seed {seed}: accepted a calibration {error:.1f} px off ground truth "
        f"(tolerance {tolerance:.1f}), reporting "
        f"{fs.calibrations['fourier']['array']['residual']:.2f} px residual."
    )
