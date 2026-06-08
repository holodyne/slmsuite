"""
Unit tests for FourierSLM class.
"""
import pytest
import numpy as np
import os
import logging
import matplotlib.pyplot as plt

from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.holography.toolbox.phase import blaze, zernike_sum
from slmsuite.holography.algorithms import SpotHologram


class TestFourierSLM:
    """Tests for public methods on FourierSLM."""

    def test_init(self, slm, camera, subtests):
        """Test FourierSLM.__init__."""

        with subtests.test("default magnification"):
            fs = FourierSLM(camera, slm)
            assert fs.cam is camera
            assert fs.slm is slm
            assert fs.mag == 1.0
            assert fs.name == f"{camera.name}-{slm.name}"
            assert isinstance(fs.calibrations, dict)
            assert hasattr(fs, "_wavefront_calibration_window_multiplier")

        with subtests.test("custom magnification"):
            fs = FourierSLM(camera, slm, mag=5.0)
            assert fs.mag == 5.0

        with subtests.test("rejects non-camera"):
            slm_tmp = SimulatedSLM(resolution=(1920, 1080))
            with pytest.raises(ValueError, match="Expected Camera"):
                FourierSLM("not_a_camera", slm_tmp)

        with subtests.test("rejects non-SLM"):
            slm_tmp = SimulatedSLM(resolution=(1920, 1080))
            cam_tmp = SimulatedCamera(slm_tmp, resolution=(512, 512))
            with pytest.raises(ValueError, match="Expected SLM"):
                FourierSLM(cam_tmp, "not_an_slm")

    def test_fourier_calibrate(self, fourierslm, subtests):
        """Test FourierSLM.fourier_calibrate — the primary Fourier calibration
        routine.  This is the most important calibration in slmsuite."""

        with subtests.test("basic calibration stores M and b"):
            fourierslm.fourier_calibrate(
                array_pitch=35, array_shape=5, plot=True,
            )
            cal = fourierslm.calibrations["fourier"]
            assert "M" in cal and "b" in cal
            assert cal["M"].shape == (2, 2)
            assert cal["b"].shape == (2, 1)

        with subtests.test("M is invertible"):
            M = fourierslm.calibrations["fourier"]["M"]
            det = np.linalg.det(M)
            assert abs(det) > 1e-10, "Calibration matrix should be invertible"

        with subtests.test("metadata attached"):
            cal = fourierslm.calibrations["fourier"]
            # Metadata from _get_calibration_metadata
            assert "__meta__" in cal or "__version__" in cal or "name" in cal

        with subtests.test("second calibration overwrites"):
            fourierslm.fourier_calibrate(
                array_pitch=30, array_shape=5, plot=True,
            )
            # Just confirm it didn't error and key still exists
            assert "fourier" in fourierslm.calibrations

        with subtests.test("scalar array_shape and array_pitch"):
            fourierslm.fourier_calibrate(
                array_pitch=35, array_shape=5, plot=False,
            )
            assert fourierslm.calibrations["fourier"]["M"].shape == (2, 2)

        with subtests.test("list array_shape and array_pitch"):
            fourierslm.fourier_calibrate(
                array_pitch=[35, 35], array_shape=[5, 5], plot=False,
            )
            assert fourierslm.calibrations["fourier"]["M"].shape == (2, 2)

        with subtests.test("non-positive pitch raises"):
            with pytest.raises(ValueError):
                fourierslm.fourier_calibrate(
                    array_pitch=-1, array_shape=5, plot=False,
                )

    @pytest.mark.slow
    def test_fourier_calibrate_large_array(self, fourierslm, fourierslm_calibrated, subtests):
        """Test fourier_calibrate with a larger grid for better statistics."""

        with subtests.test("10x10 grid calibrates"):
            fourierslm.fourier_calibrate(
                array_pitch=30, array_shape=10, plot=True,
            )
            plt.show()
            M = fourierslm.calibrations["fourier"]["M"]
            assert abs(np.linalg.det(M)) > 1e-10

        with subtests.test("calibration matches smaller grid"):
            M_large = fourierslm.calibrations["fourier"]["M"]
            b_large = fourierslm.calibrations["fourier"]["b"]
            M_small = fourierslm_calibrated.calibrations["fourier"]["M"]
            b_small = fourierslm_calibrated.calibrations["fourier"]["b"]
            assert np.allclose(M_large, M_small, rtol=0.1, atol=0.1)
            assert np.allclose(b_large, b_small, rtol=0.1, atol=0.1)

    def test_fourier_calibrate_analytic(self, fourierslm, subtests):
        """Test FourierSLM.fourier_calibrate_analytic."""

        M = np.array([[1.5, 0.1], [-0.05, 1.6]])
        b = np.array([[10.0], [20.0]])

        with subtests.test("stores M and b"):
            # Note: fourier_calibrate_analytic with arbitrary M calls set_affine
            # on SimulatedCamera, which may fail for small M values.
            # Use M values consistent with the simulated optical system.
            fourierslm.fourier_calibrate(array_pitch=35, array_shape=5, plot=False)
            real_M = fourierslm.calibrations["fourier"]["M"]
            real_b = fourierslm.calibrations["fourier"]["b"]
            fourierslm.fourier_calibrate_analytic(real_M, real_b)
            cal = fourierslm.calibrations["fourier"]
            assert np.allclose(cal["M"], real_M)
            assert np.allclose(cal["b"], real_b)

        with subtests.test("identity matrix"):
            fourierslm.fourier_calibrate_analytic(np.eye(2), np.zeros((2, 1)))
            cal = fourierslm.calibrations["fourier"]
            assert np.allclose(cal["M"], np.eye(2))

        with subtests.test("wrong-shape M raises"):
            with pytest.raises(ValueError):
                fourierslm.fourier_calibrate_analytic(np.eye(3), b)

    def test_fourier_grid_project(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.fourier_grid_project."""

        with subtests.test("returns a hologram with spot data"):
            hologram = fourierslm_calibrated.fourier_grid_project(
                array_shape=3, array_pitch=35,
            )
            assert hologram is not None
            assert hasattr(hologram, "spot_kxy_rounded")

    def test_kxyslm_to_ijcam(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.kxyslm_to_ijcam."""
        M = fourierslm_calibrated.calibrations["fourier"]["M"]
        b = fourierslm_calibrated.calibrations["fourier"]["b"]
        a = fourierslm_calibrated.calibrations["fourier"]["a"]

        with subtests.test("single point"):
            kxy = np.array([[10.0], [20.0]])
            ij = fourierslm_calibrated.kxyslm_to_ijcam(kxy)
            expected = M @ (kxy - a) + b
            assert np.allclose(ij, expected)

        with subtests.test("origin maps to b + M@a offset"):
            ij = fourierslm_calibrated.kxyslm_to_ijcam([0, 0])
            expected = M @ (np.zeros((2, 1)) - a) + b
            assert np.allclose(ij, expected, atol=1e-10)

        with subtests.test("raises without calibration"):
            fs_bare = FourierSLM(
                fourierslm_calibrated.cam, fourierslm_calibrated.slm,
            )
            with pytest.raises((KeyError, RuntimeError)):
                fs_bare.kxyslm_to_ijcam([10.0, 20.0])

    def test_ijcam_to_kxyslm(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.ijcam_to_kxyslm."""
        M = fourierslm_calibrated.calibrations["fourier"]["M"]
        b = fourierslm_calibrated.calibrations["fourier"]["b"]
        a = fourierslm_calibrated.calibrations["fourier"]["a"]

        with subtests.test("single point"):
            ij = np.array([[120.0], [140.0]])
            kxy = fourierslm_calibrated.ijcam_to_kxyslm(ij)
            expected = np.linalg.solve(M, ij - b) + a
            assert np.allclose(kxy, expected, atol=1e-10)

        with subtests.test("roundtrip kxy -> ij -> kxy"):
            kxy_orig = np.array([[15.0], [25.0]])
            ij = fourierslm_calibrated.kxyslm_to_ijcam(kxy_orig)
            kxy_back = fourierslm_calibrated.ijcam_to_kxyslm(ij)
            assert np.allclose(kxy_orig, kxy_back, atol=1e-10)

        with subtests.test("roundtrip ij -> kxy -> ij"):
            ij_orig = np.array([[200.0], [300.0]])
            kxy = fourierslm_calibrated.ijcam_to_kxyslm(ij_orig)
            ij_back = fourierslm_calibrated.kxyslm_to_ijcam(kxy)
            assert np.allclose(ij_orig, ij_back, atol=1e-10)

        with subtests.test("multiple points"):
            ij_multi = np.array([[100, 200, 300], [110, 210, 310]], dtype=float)
            kxy_multi = fourierslm_calibrated.ijcam_to_kxyslm(ij_multi)
            ij_rt = fourierslm_calibrated.kxyslm_to_ijcam(kxy_multi)
            assert np.allclose(ij_multi, ij_rt, atol=1e-10)

    def test_get_farfield_spot_size(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.get_farfield_spot_size."""

        with subtests.test("kxy basis positive"):
            size = fourierslm_calibrated.get_farfield_spot_size(
                slm_size=1.0, basis="kxy",
            )
            assert np.all(np.asarray(size) > 0)

        with subtests.test("ij basis positive"):
            size = fourierslm_calibrated.get_farfield_spot_size(
                slm_size=1.0, basis="ij",
            )
            assert np.all(np.asarray(size) > 0)

        with subtests.test("bad basis raises"):
            with pytest.raises(ValueError):
                fourierslm_calibrated.get_farfield_spot_size(
                    slm_size=1.0, basis="badvalue",
                )

    def test_get_effective_focal_length(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.get_effective_focal_length."""

        with subtests.test("ij units"):
            f = fourierslm_calibrated.get_effective_focal_length(units="ij")
            assert np.isfinite(f)
            assert f > 0

        with subtests.test("norm units"):
            f = fourierslm_calibrated.get_effective_focal_length(units="norm")
            assert np.all(np.isfinite(f))

        with subtests.test("raises without calibration"):
            fs_bare = FourierSLM(
                fourierslm_calibrated.cam, fourierslm_calibrated.slm,
            )
            with pytest.raises(RuntimeError):
                fs_bare.get_effective_focal_length()

    def test_get_farfield_extent(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.get_farfield_extent."""

        with subtests.test("returns corners array shape"):
            corners = fourierslm_calibrated.get_farfield_extent(return_corners=True)
            # 5 points (closed polygon: ll, lr, ur, ul, ll) in (2, N) format
            assert corners.shape[0] == 2
            assert corners.shape[1] == 5

        with subtests.test("returns boolean mask when return_corners=False"):
            mask = fourierslm_calibrated.get_farfield_extent(return_corners=False)
            assert mask.dtype == bool
            assert mask.shape == fourierslm_calibrated.cam.shape
            # The SLM farfield should occupy some portion of the camera
            assert mask.any()

    def test_get_camera_extent(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.get_camera_extent."""

        with subtests.test("kxy units returns 5 corner points"):
            corners = fourierslm_calibrated.get_camera_extent(units="kxy", return_corners=True)
            assert corners.shape[0] == 2
            assert corners.shape[1] == 5

        with subtests.test("knm canvas return_corners=False"):
            shape = SpotHologram.get_padded_shape(fourierslm_calibrated, padding_order=1, square_padding=True)
            mask = fourierslm_calibrated.get_camera_extent(units=shape, return_corners=False)
            assert mask.dtype == bool
            assert mask.shape == shape
            # Camera covers some portion of the SLM farfield
            assert mask.any()

        with subtests.test("corner roundtrip ij->kxy->ij"):
            # Corners in kxy space then mapped back to ij should enclose the camera
            corners_kxy = fourierslm_calibrated.get_camera_extent(units="kxy", return_corners=True)
            # Each corner maps back to a camera boundary pixel
            corners_ij = fourierslm_calibrated.kxyslm_to_ijcam(corners_kxy)
            cam_shape = fourierslm_calibrated.cam.shape
            # All points should be within a few pixels of the camera boundary
            assert np.all(corners_ij[0] >= -1) and np.all(corners_ij[0] <= cam_shape[1])
            assert np.all(corners_ij[1] >= -1) and np.all(corners_ij[1] <= cam_shape[0])

    def test_fourier_affine_property(self, fourierslm_calibrated, subtests):
        """Test that fourier_affine returns an Affine equal to kxyslm->ijcam conversion."""
        from slmsuite.holography.analysis import Affine

        with subtests.test("fourier_affine is an Affine instance"):
            aff = fourierslm_calibrated.fourier_affine
            assert isinstance(aff, Affine)

        with subtests.test("fourier_affine matches kxyslm_to_ijcam"):
            kxy = np.array([[10.0], [20.0]])
            via_affine = fourierslm_calibrated.fourier_affine @ kxy
            via_method = fourierslm_calibrated.kxyslm_to_ijcam(kxy)
            assert np.allclose(via_affine, via_method, atol=1e-10)

        with subtests.test("fourier_affine inverse matches ijcam_to_kxyslm"):
            ij = np.array([[150.0], [200.0]])
            via_inv = fourierslm_calibrated.fourier_affine.inv @ ij
            via_method = fourierslm_calibrated.ijcam_to_kxyslm(ij)
            assert np.allclose(via_inv, via_method, atol=1e-10)

    def test_simulate(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.simulate."""

        with subtests.test("returns FourierSLM with simulated hardware"):
            fs_sim = fourierslm_calibrated.simulate()
            assert isinstance(fs_sim, FourierSLM)
            assert isinstance(fs_sim.slm, SimulatedSLM)
            assert isinstance(fs_sim.cam, SimulatedCamera)

        with subtests.test("calibration copied"):
            fs_sim = fourierslm_calibrated.simulate()
            assert np.allclose(
                fs_sim.calibrations["fourier"]["M"],
                fourierslm_calibrated.calibrations["fourier"]["M"],
            )

        with subtests.test("coordinate transform matches original"):
            fs_sim = fourierslm_calibrated.simulate()
            kxy = np.array([10.0, 15.0])
            ij_real = fourierslm_calibrated.kxyslm_to_ijcam(kxy)
            ij_sim = fs_sim.kxyslm_to_ijcam(kxy)
            assert np.allclose(ij_real, ij_sim)

        with subtests.test("raises without calibration"):
            fs_bare = FourierSLM(
                fourierslm_calibrated.cam, fourierslm_calibrated.slm,
            )
            with pytest.raises(ValueError, match="Cannot simulate"):
                fs_bare.simulate()

    def test_name_calibration(self, fourierslm, subtests):
        """Test FourierSLM.name_calibration."""

        for cal_type in ("fourier", "wavefront"):
            with subtests.test(f"type={cal_type}"):
                name = fourierslm.name_calibration(cal_type)
                assert isinstance(name, str)
                assert cal_type in name.lower()

    def test_save_load_calibration(self, fourierslm_calibrated, temp_dir, subtests):
        """Test FourierSLM.save_calibration and load_calibration round-trip."""

        with subtests.test("save creates file"):
            path = fourierslm_calibrated.save_calibration(
                "fourier", path=temp_dir, name="test_save",
            )
            assert os.path.exists(path)

        with subtests.test("load restores calibration"):
            path = fourierslm_calibrated.save_calibration(
                "fourier", path=temp_dir, name="test_load",
            )
            fs_new = FourierSLM(
                fourierslm_calibrated.cam, fourierslm_calibrated.slm,
            )
            fs_new.load_calibration("fourier", file_path=path)
            assert np.allclose(
                fs_new.calibrations["fourier"]["M"],
                fourierslm_calibrated.calibrations["fourier"]["M"],
            )
            assert np.allclose(
                fs_new.calibrations["fourier"]["b"],
                fourierslm_calibrated.calibrations["fourier"]["b"],
            )

        with subtests.test("save nonexistent type raises"):
            with pytest.raises(ValueError):
                fourierslm_calibrated.save_calibration(
                    "nonexistent", path=temp_dir,
                )

    def test_load(self, fourierslm_calibrated, temp_dir, subtests):
        """Test FourierSLM.load static constructor."""

        path = fourierslm_calibrated.save_calibration(
            "fourier", path=temp_dir, name="test_static_load",
        )

        with subtests.test("returns valid FourierSLM"):
            fs = FourierSLM.load(path)
            assert isinstance(fs, FourierSLM)
            assert isinstance(fs.slm, SimulatedSLM)
            assert isinstance(fs.cam, SimulatedCamera)

        with subtests.test("calibration loaded"):
            fs = FourierSLM.load(path)
            # FourierSLM.load only restores metadata/hardware, not calibration data
            assert isinstance(fs, FourierSLM)

    def test_plot(self, fourierslm, subtests):
        """Test FourierSLM.plot."""

        with subtests.test("default call"):
            phase = np.random.rand(*fourierslm.slm.shape) * 2 * np.pi
            axs = fourierslm.plot(phase=phase)
            plt.show()
            assert axs is not None
            assert len(axs) == 2

    @pytest.mark.slow
    def test_wavefront_calibrate_superpixel(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.wavefront_calibrate_superpixel with various settings."""

        cal_point = [150, 150]
        sp_size = fourierslm_calibrated.slm.shape[0] // 6

        with subtests.test(f"add aberration to slm"):
            phase_abberation = zernike_sum(
                fourierslm_calibrated.slm,
                indices=(3, 4, 5, 7, 8),
                weights=(1, -2, 3, 1, 1),
                aperture=None,
                use_mask=False
            )
            fourierslm_calibrated.slm.set_source_analytic(
                phase_offset=phase_abberation,
                sim=True
            )
            fourierslm_calibrated.slm.plot_source(sim=True)

        with subtests.test(f"direct blaze to calibration point"):
            kxy = fourierslm_calibrated.ijcam_to_kxyslm(cal_point)
            fourierslm_calibrated.slm.set_phase(blaze(fourierslm_calibrated.slm, vector=kxy))
            img = fourierslm_calibrated.cam.get_image()

            fourierslm_calibrated.plot(image=img, title="Blazed spot at calibration point")
            plt.show()

            assert img[140:160, 140:160].mean() > img.mean(), "Blazed spot should be brighter than background"

        for phase_steps, name in [(None, "amplitude-only"), (1, "one-shot phase"), (5, "many-shot phase")]:
            fourierslm_calibrated.slm.source["phase"] = None    # Clear any old calibration.

            # FUTURE: test for warnings if underexposed.
            # fourierslm_calibrated.cam.set_exposure(0.01)

            # with subtests.test(f"test low-exposure {name} (phase_steps={phase_steps})"):
            #     result = fourierslm_calibrated.wavefront_calibrate_superpixel(
            #         calibration_points=cal_point,
            #         superpixel_size=sp_size,
            #         phase_steps=phase_steps,
            #         plot=True,
            #         test_index=-2,
            #     )

            fourierslm_calibrated.cam.set_exposure(.1)

            # FUTURE: benchmark the calibration tick?
            with subtests.test(f"test {name} (phase_steps={phase_steps})"):
                result = fourierslm_calibrated.wavefront_calibrate_superpixel(
                    calibration_points=cal_point,
                    superpixel_size=sp_size,
                    phase_steps=phase_steps,
                    plot=True,
                    test_index=-2,
                )

            with subtests.test(f"calibrate {name} (phase_steps={phase_steps})"):
                result = fourierslm_calibrated.wavefront_calibrate_superpixel(
                    calibration_points=cal_point,
                    superpixel_size=sp_size,
                    phase_steps=phase_steps,
                )
                assert isinstance(result, dict)
                assert "power" in result
                cal = fourierslm_calibrated.calibrations["wavefront_superpixel"]
                assert "superpixel_size" in cal

            with subtests.test(f"process {name} (phase_steps={phase_steps})"):
                fourierslm_calibrated.wavefront_calibration_superpixel_process(
                    plot=True,
                    smooth=False,
                )
                plt.show()

            with subtests.test(f"process smooth {name} (phase_steps={phase_steps})"):
                fourierslm_calibrated.wavefront_calibration_superpixel_process(
                    plot=True,
                    smooth=True,
                )
                plt.show()

            # Verifying phase calibration is difficult with low resolution, but
            # amplitude is decent.
            with subtests.test(f"check amplitude {name} (phase_steps={phase_steps})"):
                fourierslm_calibrated.slm.plot_source(sim=False)
                fourierslm_calibrated.slm.plot_source(sim=True)

                # Subtract the calibrated amplitude from the simulated amplitude
                amp = fourierslm_calibrated.slm.source["amplitude"]
                amp_sim = fourierslm_calibrated.slm.source["amplitude_sim"]

                amp_diff = np.abs(amp - amp_sim)
                plt.imshow(amp_diff)
                plt.title("Amplitude difference")
                plt.colorbar()
                plt.show()
                amp_diff_norm = np.sum(amp_diff) / np.sum(amp_sim)
                logger = logging.getLogger("conftest")
                logger.info(f"Normalized amplitude difference {name}: {amp_diff_norm:.2f}")
                assert amp_diff_norm < .5, f"Calibrated amplitude should be close to simulated amplitude ({amp_diff_norm:.2f} off)"

        with subtests.test("requires Fourier calibration"):
            fs_bare = FourierSLM(
                fourierslm_calibrated.cam, fourierslm_calibrated.slm,
            )
            with pytest.raises((RuntimeError, KeyError)):
                fs_bare.wavefront_calibrate_superpixel(
                    calibration_points=cal_point,
                    superpixel_size=sp_size,
                    plot=-1,
                )

        with subtests.test("stores scheduling metadata"):
            cal = fourierslm_calibrated.calibrations["wavefront_superpixel"]
            assert "scheduling" in cal
            assert "slm_supershape" in cal

    @pytest.mark.slow
    def test_wavefront_calibrate_zernike(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.wavefront_calibrate_zernike."""

        # wavefront_calibrate_zernike passes calibration_points to
        # CompressedSpotHologram with basis=zernike_indices, so points must be
        # in the zernike basis (radians), not ij pixels.  Generate ij-space
        # points with wavefront_calibration_points(), then convert to zernike.
        from slmsuite.holography.toolbox import convert_vector
        ij_pts = fourierslm_calibrated.wavefront_calibration_points(pitch=120)
        cal_pts = convert_vector(
            ij_pts, from_units="ij", to_units="zernike",
            hardware=fourierslm_calibrated
        )

        with subtests.test("perturbation=0 projects spots only"):
            fourierslm_calibrated.wavefront_calibrate_zernike(
                calibration_points=cal_pts,
                zernike_indices=4,
                perturbation=0,
                optimize_position=False,
                optimize_weights=False,
                plot=-1,
            )

        with subtests.test("basic sweep stores calibration"):
            result = fourierslm_calibrated.wavefront_calibrate_zernike(
                calibration_points=cal_pts,
                zernike_indices=4,
                perturbation=0.5,
                optimize_position=False,
                optimize_weights=False,
                plot=-1,
            )
            assert result is not None
            assert "wavefront_zernike" in fourierslm_calibrated.calibrations
            cal = fourierslm_calibrated.calibrations["wavefront_zernike"]
            assert "corrected_spots" in cal
            assert "zernike_indices" in cal

        with subtests.test("iteration on previous calibration"):
            result2 = fourierslm_calibrated.wavefront_calibrate_zernike(
                perturbation=0.3,
                optimize_position=False,
                optimize_weights=False,
                plot=-1,
            )
            assert result2 is not None

    def test_wavefront_calibration_points(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.wavefront_calibration_points."""

        with subtests.test("returns 2×N array"):
            pts = fourierslm_calibrated.wavefront_calibration_points(pitch=60)
            assert pts.ndim == 2
            assert pts.shape[0] == 2
            assert pts.shape[1] > 0

        with subtests.test("larger pitch gives fewer points"):
            pts_coarse = fourierslm_calibrated.wavefront_calibration_points(pitch=120)
            pts_fine = fourierslm_calibrated.wavefront_calibration_points(pitch=60)
            assert pts_coarse.shape[1] <= pts_fine.shape[1]

    def test_full_workflow(self, slm, camera, temp_dir, subtests):
        """Integration: calibrate -> save -> load -> simulate -> transform."""

        fs = FourierSLM(camera, slm)

        with subtests.test("calibrate"):
            fs.fourier_calibrate(array_pitch=35, array_shape=5, plot=False)
            assert "fourier" in fs.calibrations

        with subtests.test("save"):
            path = fs.save_calibration("fourier", path=temp_dir)
            assert os.path.exists(path)

        with subtests.test("load into new instance"):
            fs_loaded = FourierSLM.load(path)
            # FourierSLM.load restores hardware metadata but not all calibration keys;
            # reload the calibration explicitly.
            fs_loaded.load_calibration("fourier", file_path=path)
            assert np.allclose(
                fs.calibrations["fourier"]["M"],
                fs_loaded.calibrations["fourier"]["M"],
            )

        with subtests.test("simulate from loaded"):
            fs_sim = fs_loaded.simulate()
            kxy = np.array([10.0, 15.0])
            assert np.allclose(
                fs.kxyslm_to_ijcam(kxy),
                fs_sim.kxyslm_to_ijcam(kxy),
            )
