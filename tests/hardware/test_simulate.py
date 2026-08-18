"""
Unit tests for FourierSLM.simulate(), which clones a hardware-based experiment into
a simulation so that the same algorithm can be run against either and compared.

Cloning a simulation is deterministic: the clone copies the simulated truth rather
than deriving it, so (absent camera noise) it must reproduce the original frame for
frame. That makes a simulated system a usable stand-in for hardware here, and lets
the geometry assertions be exact rather than approximate.
"""
import numpy as np
import pytest

from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.holography.toolbox.phase import blaze, zernike_sum

from conftest import (
    build_simulated_system,
    in_view_kxy,
    install_ground_truth_calibration,
    seed_for,
    view_kxy_grid,
)


# Geometries spanning the ways the camera can sit in the SLM's k-space. "identity"
# is excluded: its camera samples the knm grid directly with no affine, so the clone
# (which is always built from the Fourier calibration's affine) interpolates instead
# and is only approximately equal. See test_identity_case_is_approximate.
CLONE_CASES = (
    "matched",
    "fov_much_larger",
    "fov_smaller",
    "rotated",
    "sheared",
    "offset",
    "zeroth_outside",
    "anisotropic",
    "mirrored",
    "camera_wide",
    "pitch_anisotropic",
    "defocus",
)


def _calibrated(name, **overrides):
    """A simulated system standing in for hardware, with its ground truth installed."""
    seed_for(name)
    fs = build_simulated_system(name, **overrides)
    fs.cam.set_exposure(0.1)
    install_ground_truth_calibration(fs)
    return fs


def _display(fs, kxy=None):
    """Puts a blaze on the SLM so that the camera sees something off-axis."""
    if kxy is None:
        kxy = in_view_kxy(fs, frac=0.4)
    fs.slm.set_phase(blaze(fs.slm, np.squeeze(kxy)))
    return kxy


class TestSimulateFidelity:
    """The clone reproduces the system it was cloned from."""

    @pytest.mark.parametrize("name", CLONE_CASES)
    def test_image_matches(self, name, subtests):
        """The clone renders the same image as the system it cloned."""
        fs = _calibrated(name)
        kxy = _display(fs)

        fs_sim = fs.simulate()

        with subtests.test("camera geometry"):
            assert fs_sim.cam.shape == fs.cam.shape
            assert fs_sim.cam.bitresolution == fs.cam.bitresolution
            assert np.allclose(fs_sim.cam.pitch_um, fs.cam.pitch_um)
            assert fs_sim.cam.exposure_s == fs.cam.exposure_s

        with subtests.test("image is identical"):
            assert np.array_equal(fs.cam.get_image(), fs_sim.cam.get_image())

        with subtests.test("coordinate frames agree"):
            probe = view_kxy_grid(fs, count=4)
            assert np.allclose(
                fs.kxyslm_to_ijcam(probe), fs_sim.kxyslm_to_ijcam(probe), atol=1e-9
            )
            ij = fs.kxyslm_to_ijcam(kxy)
            assert np.allclose(fs.ijcam_to_kxyslm(ij), fs_sim.ijcam_to_kxyslm(ij), atol=1e-12)

        with subtests.test("clone writes phase the same way"):
            phase = blaze(fs.slm, np.squeeze(in_view_kxy(fs, frac=0.7)))
            fs.slm.set_phase(phase)
            fs_sim.slm.set_phase(phase)
            assert np.array_equal(
                np.asarray(fs.slm.display), np.asarray(fs_sim.slm.display)
            )
            assert np.array_equal(fs.cam.get_image(), fs_sim.cam.get_image())

    def test_identity_case_is_approximate(self):
        """
        The "identity" camera samples the knm grid directly; the clone always goes
        through the calibrated affine, so it interpolates. The two agree on where the
        light lands, but not pixel for pixel.
        """
        fs = _calibrated("identity")
        _display(fs)

        fs_sim = fs.simulate()

        img = fs.cam.get_image()
        img_sim = fs_sim.cam.get_image()

        assert img.shape == img_sim.shape
        assert np.allclose(
            np.unravel_index(np.argmax(img), img.shape),
            np.unravel_index(np.argmax(img_sim), img_sim.shape),
            atol=1,
        )

    def test_requires_fourier_calibration(self):
        """Without the Fourier relationship there is nothing to place the camera by."""
        seed_for("matched")
        fs = build_simulated_system("matched")
        assert "fourier" not in fs.calibrations
        with pytest.raises(ValueError, match="Cannot simulate"):
            fs.simulate()


class TestSimulateCameraFraming:
    """
    The stored Fourier calibration maps k-space onto *raw* sensor pixels, while the
    images a camera delivers are cropped to its WOI, binned, and reoriented. The clone
    is built as the delivered image, so it inherits all three through
    ``fourier_affine``; taking the stored calibration at face value instead would offset
    the clone by the WOI origin and transpose its frames.
    """

    def test_woi(self, subtests):
        fs = _calibrated("matched")
        fs.cam.set_woi((20, 64, 30, 48))
        _display(fs)

        fs_sim = fs.simulate()

        with subtests.test("the clone is the window, and needs none of its own"):
            assert fs_sim.cam.shape == fs.cam.shape == (48, 64)
            assert fs_sim.cam.woi == (0, 64, 0, 48)

        with subtests.test("affine is not offset by the woi origin"):
            probe = view_kxy_grid(fs, count=4)
            assert np.allclose(
                fs.kxyslm_to_ijcam(probe), fs_sim.kxyslm_to_ijcam(probe), atol=1e-9
            )

        with subtests.test("image is identical"):
            assert np.array_equal(fs.cam.get_image(), fs_sim.cam.get_image())

    def test_binning(self, subtests):
        fs = _calibrated("matched")
        fs.cam.set_binning((2, 2))
        _display(fs)

        fs_sim = fs.simulate()

        with subtests.test("shape and pitch are the binned ones"):
            assert fs_sim.cam.shape == fs.cam.shape
            assert np.allclose(fs_sim.cam.pitch_um, fs.cam.pitch_um)

        with subtests.test("binned pixels sum, so the clone reads out the wider range"):
            assert fs_sim.cam.bitresolution == fs.cam.bitresolution

        with subtests.test("affine agrees"):
            probe = view_kxy_grid(fs, count=4)
            assert np.allclose(
                fs.kxyslm_to_ijcam(probe), fs_sim.kxyslm_to_ijcam(probe), atol=1e-9
            )

        with subtests.test("image agrees"):
            # Not identical: the hardware quantizes each pixel and then sums four of
            # them, while the clone renders the binned pixel directly.
            (img, img_sim) = (fs.cam.get_image(), fs_sim.cam.get_image())
            assert np.unravel_index(np.argmax(img), img.shape) == np.unravel_index(
                np.argmax(img_sim), img_sim.shape
            )
            assert np.isclose(img_sim.sum(), img.sum(), rtol=0.5)

    @pytest.mark.parametrize(
        "orientation",
        ({"rot": "90"}, {"rot": "180"}, {"fliplr": True}, {"rot": "270", "flipud": True}),
        ids=("rot90", "rot180", "fliplr", "rot270_flipud"),
    )
    def test_orientation_transform(self, orientation, subtests):
        """A rotated or flipped camera must not have its transform applied twice."""
        fs = _calibrated("camera_wide")

        # Rebuild the stand-in camera with an orientation, keeping the same placement.
        cam = SimulatedCamera(
            fs.slm,
            resolution=np.flip(fs.cam._shape),
            M=fs.cam.M,
            b=fs.cam.b,
            pitch_um=fs.cam._pitch_um,
            bitdepth=fs.cam.bitdepth,
            **orientation,
        )
        cam.set_exposure(0.1)
        fs = FourierSLM(cam, fs.slm)
        install_ground_truth_calibration(fs)
        _display(fs)

        fs_sim = fs.simulate()

        with subtests.test("shape is not transposed"):
            assert fs_sim.cam.shape == fs.cam.shape

        with subtests.test("image is identical"):
            assert np.array_equal(fs.cam.get_image(), fs_sim.cam.get_image())

        with subtests.test("coordinate frames agree"):
            probe = view_kxy_grid(fs, count=4)
            assert np.allclose(
                fs.kxyslm_to_ijcam(probe), fs_sim.kxyslm_to_ijcam(probe), atol=1e-9
            )


class TestSimulateSLMCharacteristics:
    """The clone carries the SLM's phase response, aperture, and source."""

    def test_gamma(self, subtests):
        """
        A coarsely-quantized SLM with a nonlinear response (e.g. a PLM's 16 non-uniform
        phase states) must be simulated as such, not as an ideal linear one.
        """
        fs = _calibrated("matched")
        gamma = np.sort(np.random.rand(fs.slm.bitresolution))
        fs.slm.gamma_sim = gamma           # What the stand-in hardware realizes.
        fs.slm.set_gamma(gamma)            # What it quantizes through.
        _display(fs)

        fs_sim = fs.simulate()

        with subtests.test("realized response is cloned"):
            assert np.allclose(fs_sim.slm.gamma_sim, gamma)

        with subtests.test("quantization table is cloned"):
            assert np.array_equal(np.asarray(fs_sim.slm.gamma), np.asarray(fs.slm.gamma))
            assert np.array_equal(np.asarray(fs_sim.slm.lut), np.asarray(fs.slm.lut))

        with subtests.test("image is identical"):
            assert np.array_equal(fs.cam.get_image(), fs_sim.cam.get_image())

        with subtests.test("an ideal SLM stays ideal"):
            fs_ideal = _calibrated("matched")
            assert fs_ideal.simulate().slm.gamma_sim is None

    def test_aperture(self, subtests):
        fs = _calibrated("matched")
        fs.slm.set_aperture(radius=0.3, center=(70, 60), units="frac")
        _display(fs)

        fs_sim = fs.simulate()

        with subtests.test("spec and center are cloned"):
            assert np.allclose(fs_sim.slm.aperture.spec, fs.slm.aperture.spec)
            assert np.allclose(fs_sim.slm.aperture.center, fs.slm.aperture.center)

        with subtests.test("mask is cloned"):
            assert np.array_equal(
                np.asarray(fs_sim.slm.aperture_mask), np.asarray(fs.slm.aperture_mask)
            )

        with subtests.test("a default aperture is left alone"):
            fs_bare = _calibrated("matched")
            assert not fs_bare.simulate().slm.aperture.crops

    def test_measured_source_becomes_the_simulated_truth(self, subtests):
        """
        A wavefront-calibrated SLM carries a measured amplitude and a phase
        *correction*. The clone must simulate the aberration that correction undoes,
        so that applying it flattens the wavefront exactly as on hardware.
        """
        fs = _calibrated("matched")
        aberration = zernike_sum(fs.slm, (4, 7), (2.0, -1.5))
        amplitude = np.abs(np.asarray(fs.slm.source["amplitude_sim"]))

        # Stand in for a wavefront calibration: the measurement is the correction.
        fs.slm.source["phase"] = -aberration
        fs.slm.source["amplitude"] = amplitude
        del fs.slm.source["amplitude_sim"], fs.slm.source["phase_sim"]

        fs_sim = fs.simulate()

        with subtests.test("truth is the aberration, not the correction"):
            assert np.allclose(fs_sim.slm.source["phase_sim"], aberration)
            assert np.allclose(fs_sim.slm.source["amplitude_sim"], amplitude)

        with subtests.test("applying the correction flattens the wavefront"):
            # The unquantized render, since a corrected spot saturates the readout.
            fs_sim.slm.set_phase(None, phase_correct=True)
            corrected = fs_sim.cam._get_image_hw(0, quantize=False)
            fs_sim.slm.set_phase(None, phase_correct=False)
            uncorrected = fs_sim.cam._get_image_hw(0, quantize=False)
            assert corrected.max() > 2 * uncorrected.max()

    def test_source_is_not_aliased(self, subtests):
        """Editing the clone's source in place must not corrupt the hardware."""
        fs = _calibrated("matched")
        fs_sim = fs.simulate()

        for key in ("amplitude_sim", "phase_sim"):
            with subtests.test(key):
                assert fs_sim.slm.source[key] is not fs.slm.source[key]
                before = np.array(fs.slm.source[key])
                fs_sim.slm.source[key] *= 0
                assert np.array_equal(fs.slm.source[key], before)

    def test_source_override(self):
        """``source=`` injects a known truth, e.g. to test what a calibration recovers."""
        fs = _calibrated("matched")
        injected = zernike_sum(fs.slm, (4, 8), (3.0, 1.0))

        fs_sim = fs.simulate(source={"phase_sim": injected})

        assert np.allclose(fs_sim.slm.source["phase_sim"], injected)
        assert not np.allclose(fs.slm.source["phase_sim"], injected)

    def test_settle_time(self, subtests):
        fs = _calibrated("matched")
        fs.slm.settle_time_s = 0.05

        with subtests.test("dropped by default, so simulation is not slowed"):
            assert fs.simulate().slm.settle_time_s == 0

        with subtests.test("cloned on request"):
            assert fs.simulate(settle=True).slm.settle_time_s == 0.05


class TestSimulateIdentity:
    """The clone is the same kind of object as what it cloned."""

    def test_preserves_subclass(self):
        """
        A FourierSLM subclass (e.g. one adding a calibration routine) must clone into
        its own class, or the clone cannot run the algorithm being compared.
        """
        class _Subclass(FourierSLM):
            def only_here(self):
                return True

        fs = _calibrated("matched")
        fs.__class__ = _Subclass

        fs_sim = fs.simulate()

        assert type(fs_sim) is _Subclass
        assert fs_sim.only_here()

    def test_preserves_mag_and_calibrations(self, subtests):
        fs = _calibrated("matched")
        fs.mag = 4.0
        fs.calibrations["settle"] = {"times": np.arange(3.0), "data": np.ones((3, 2))}

        fs_sim = fs.simulate()

        with subtests.test("mag"):
            assert fs_sim.mag == 4.0

        with subtests.test("calibrations are deep-copied"):
            assert set(fs_sim.calibrations) == set(fs.calibrations)
            assert np.allclose(
                fs_sim.calibrations["fourier"]["M"], fs.calibrations["fourier"]["M"]
            )
            fs_sim.calibrations["settle"]["data"] *= 0
            assert np.all(fs.calibrations["settle"]["data"] == 1)

        with subtests.test("hardware is untouched"):
            assert fs_sim.slm is not fs.slm
            assert fs_sim.cam is not fs.cam

    def test_hardware_is_simulated(self):
        fs = _calibrated("matched")
        fs_sim = fs.simulate()
        assert isinstance(fs_sim.slm, SimulatedSLM)
        assert isinstance(fs_sim.cam, SimulatedCamera)


class TestLoad:
    """
    ``FourierSLM.load()`` rebuilds a system from a file with no hardware present. The
    camera it builds must be *placed* by the Fourier calibration it reads, or the
    system's images and its own ``kxyslm_to_ijcam()`` describe different geometries.
    """

    def _spot_ij(self, fs, kxy):
        """Where the brightest pixel of a blaze at ``kxy`` actually lands."""
        fs.slm.set_phase(blaze(fs.slm, np.squeeze(kxy)))
        img = fs.cam.get_image()
        return np.flip(np.unravel_index(np.argmax(img), img.shape)).astype(float)

    @pytest.mark.parametrize("woi", (None, (20, 64, 30, 48)), ids=("full", "woi"))
    def test_camera_is_placed_by_the_calibration(self, woi, temp_dir, subtests):
        # The WOI goes on before the calibration, since a calibration's metadata is a
        # snapshot of the hardware as it was when the calibration was taken.
        seed_for("matched")
        fs = build_simulated_system("matched")
        fs.cam.set_exposure(0.1)
        if woi is not None:
            fs.cam.set_woi(woi)
        install_ground_truth_calibration(fs)

        # Somewhere inside the frame the camera actually delivers, which for a WOI is
        # not the same thing as somewhere inside the farfield.
        (height, width) = fs.cam.shape
        kxy = fs.ijcam_to_kxyslm([0.6 * width, 0.4 * height])

        path = fs.save_calibration("fourier", path=temp_dir, name="load_place")
        fs_loaded = FourierSLM.load(path)
        fs_loaded.cam.set_exposure(fs.cam.exposure_s)

        with subtests.test("shape"):
            assert fs_loaded.cam.shape == fs.cam.shape

        with subtests.test("coordinate frame survives the round trip"):
            assert np.allclose(
                fs_loaded.kxyslm_to_ijcam(kxy), fs.kxyslm_to_ijcam(kxy), atol=1e-9
            )

        with subtests.test("images agree with the frame they are described in"):
            # The regression: a camera left unplaced samples the SLM's knm grid
            # directly, so its spots ignore the calibration that was just loaded.
            expected = fs_loaded.kxyslm_to_ijcam(kxy).ravel()
            assert np.allclose(self._spot_ij(fs_loaded, kxy), expected, atol=2)

        with subtests.test("agrees with the system it was saved from"):
            assert np.allclose(self._spot_ij(fs_loaded, kxy), self._spot_ij(fs, kxy), atol=2)

    def test_restores_wavelength_and_geometry(self, temp_dir, subtests):
        """Every calibration is wavelength specific, so the wavelengths must survive."""
        fs = _calibrated("matched")
        path = fs.save_calibration("fourier", path=temp_dir, name="load_meta")

        fs_loaded = FourierSLM.load(path)

        with subtests.test("wavelength"):
            assert fs_loaded.slm.wav_um == fs.slm.wav_um
            assert fs_loaded.slm.wav_design_um == fs.slm.wav_design_um

        with subtests.test("geometry"):
            assert fs_loaded.slm.shape == fs.slm.shape
            assert np.allclose(fs_loaded.slm.pitch_um, fs.slm.pitch_um)
            assert np.allclose(fs_loaded.cam.pitch_um, fs.cam.pitch_um)
            assert fs_loaded.slm.bitdepth == fs.slm.bitdepth
            assert fs_loaded.cam.bitdepth == fs.cam.bitdepth

        with subtests.test("mag"):
            assert fs_loaded.mag == fs.mag

    def test_restores_source_and_calibrations_from_a_pickle(self, temp_dir, subtests):
        """A full pickle carries the measured source and every calibration."""
        fs = _calibrated("matched")
        fs.calibrations["settle"] = {"times": np.arange(3.0), "data": np.ones((3, 2))}
        path = fs.save(path=temp_dir, name="load_pickle")

        fs_loaded = FourierSLM.load(path)

        with subtests.test("every calibration"):
            assert set(fs_loaded.calibrations) == set(fs.calibrations)

        with subtests.test("measured source"):
            assert np.allclose(
                fs_loaded.slm.source["amplitude_sim"], fs.slm.source["amplitude_sim"]
            )

        with subtests.test("camera is still placed"):
            kxy = in_view_kxy(fs, frac=0.4)
            assert np.allclose(
                fs_loaded.kxyslm_to_ijcam(kxy), fs.kxyslm_to_ijcam(kxy), atol=1e-9
            )


class TestSimulateRadiometry:
    """
    The simulated far-field carries unit total power, so counts are arbitrary until
    they are matched against a hardware frame.
    """

    def test_match_counts(self, subtests):
        fs = _calibrated("fov_larger")
        _display(fs)

        # Stand in for a hardware frame three times brighter than the simulation.
        reference = fs.cam.get_image().astype(float) * 3

        fs_sim = fs.simulate(reference=reference)

        with subtests.test("gain lands on the reference total"):
            # Compare the unquantized render: a dim frame loses sub-count pixels to
            # the integer readout, which is a property of the camera, not the match.
            assert np.isclose(fs_sim.cam._get_image_hw(0, quantize=False).sum(), reference.sum())

        with subtests.test("subtracts the background from the target"):
            background = np.full(fs.cam.shape, 0.5)
            fs_bg = fs.simulate(reference=reference + background, background=background)
            fs_bg.cam.noise = None
            assert np.isclose(fs_bg.cam._get_image_hw(0, quantize=False).sum(), reference.sum())

        with subtests.test("refuses a reference with no signal"):
            with pytest.raises(ValueError, match="positive signal"):
                fs.simulate(reference=np.zeros(fs.cam.shape))

    def test_match_counts_respects_the_delivered_frame(self):
        """
        ``match_counts`` is matched against a reference of the camera's delivered shape,
        so a camera with a window of its own must be measured over that window and not
        over its whole sensor.
        """
        fs = _calibrated("fov_larger")
        _display(fs)
        fs.cam.set_woi((16, 64, 24, 48))

        reference = fs.cam.get_image().astype(float) * 3
        assert reference.shape == fs.cam.shape

        fs.cam.match_counts(reference)

        fs.cam.noise = None
        delivered = fs.cam.transform(fs.cam._crop_to_woi(fs.cam._get_image_hw(0, quantize=False)))
        assert np.isclose(delivered.sum(), reference.sum())

    def test_noise_from_background(self, subtests):
        fs = _calibrated("fov_larger")
        _display(fs)

        background = np.random.normal(0.6, 0.25, size=fs.cam.shape)
        fs_sim = fs.simulate(background=background)

        with subtests.test("noise is fit"):
            assert set(fs_sim.cam.noise) == {"dark", "read"}

        with subtests.test("reproduces the background statistics"):
            # Zero the gain so only the noise terms remain.
            fs_sim.cam.gain = 0
            blank = np.stack([fs_sim.cam._get_image_hw(0, quantize=False) for _ in range(16)])
            assert np.isclose(blank.mean(), background.mean(), rtol=0.05)
            assert np.isclose(blank.std(), background.std(), rtol=0.05)

        with subtests.test("dark term scales with exposure"):
            fs_sim.cam.set_exposure(2 * fs.cam.exposure_s)
            doubled = np.stack([fs_sim.cam._get_image_hw(0, quantize=False) for _ in range(16)])
            assert np.isclose(doubled.mean(), 2 * background.mean(), rtol=0.05)

    def test_simulated_detector_characteristics_are_cloned(self, subtests):
        """Cloning a simulation carries its gain, noise, and pixel efficiency."""
        fs = _calibrated("matched")
        fs.cam.gain = 12.5
        fs.cam.noise = {"read": lambda img: 0.01 * img}
        fs.cam._aperture = np.random.rand(*fs.cam._shape)

        fs_sim = fs.simulate()

        with subtests.test("gain"):
            assert fs_sim.cam.gain == 12.5
        with subtests.test("noise"):
            assert set(fs_sim.cam.noise) == {"read"}
        with subtests.test("pixel efficiency"):
            assert np.array_equal(fs_sim.cam._aperture, fs.cam._aperture)
