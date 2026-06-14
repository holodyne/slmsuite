"""
Datastructures, methods, and calibrations for an SLM monitored by a camera.
"""

import os
import copy
import matplotlib.pyplot as plt
from slmsuite._plotting import _slmsuite_plt_show
import numpy as np
import warnings

from slmsuite import __version__
from slmsuite._logging import _Loggable
from slmsuite.holography.analysis.files import load_h5, save_h5, generate_path, latest_path

from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.slms.simulated import SimulatedSLM

# Import calibrations (separated into different files for readability).
from slmsuite.hardware.cameraslms._fourier import _FourierCalibration
from slmsuite.hardware.cameraslms._pixel import _PixelCalibration
from slmsuite.hardware.cameraslms._settle import _SettleCalibration
from slmsuite.hardware.cameraslms._wavefront import _WavefrontCalibration

class CameraSLM(_Loggable):
    """
    Base class for an SLM with camera feedback.

    Attributes
    ----------
    cam : ~slmsuite.hardware.cameras.camera.Camera
        Instance of :class:`~slmsuite.hardware.cameras.camera.Camera`
        which interfaces with a camera. This camera is
        used to provide closed-loop feedback to an SLM for calibration and holography.
    slm : ~slmsuite.hardware.slms.slm.SLM
        Instance of :class:`~slmsuite.hardware.slms.slm.SLM`
        which interfaces with a phase display.
    name : str
        Stores ``cam.name + '-' + slm.name``.
    mag : float
        Magnification of the camera relative to an experiment plane. For instance,
        ``mag = 10`` could refer to the use of a 10x objective (with appropriate
        imaging lensing) between the experiment plane and the camera.
        In this case, the images apparent on the camera are 10x larger than the true
        objects at the experiment plane.
    """
    _pickle = ["name", "cam", "slm", "mag"]
    _pickle_data = []

    def __init__(self, cam=None, slm=None, mag=1):
        """
        Initialize an SLM linked to a camera, with given magnification between the
        camera and experiment planes.

        Parameters
        ----------
        cam : ~slmsuite.hardware.cameras.camera.Camera OR (int, int) OR None
            Instance of :class:`~slmsuite.hardware.cameras.camera.Camera`
            which interfaces with a camera. This camera is
            used to provide closed-loop feedback to an SLM for calibration and holography.
            If a shape ``(int, int)`` is passed and ``slm=None``,
            then a simulated system is constructed with the desired resolution.
            If ``None``, then the shape defaults to ``(512, 512)``.
        slm : ~slmsuite.hardware.slms.slm.SLM OR None
            Instance of :class:`~slmsuite.hardware.slms.slm.SLM`
            which interfaces with a phase display.
        mag : float
            Magnification of the camera relative to an experiment plane. For instance,
            ``mag = 10`` could refer to the use of a 10x objective (with appropriate
            imaging lensing) between the experiment plane and the camera.
            In this case, the images apparent on the camera are ten times larger than
            the true objects at the experiment plane.

            Note
            ~~~~
            This magnification is currently isotropic. In the future, anisotropy between
            the camera and experiment planes could be implemented.
        """
        # First, handle the case where we want to quickly construct a simulated system.
        if cam is None:
            cam = (512, 512)

        if isinstance(cam, (list, tuple)):
            if slm is not None:
                raise ValueError("When a shape is passed for cam, slm must be None.")
            slm = SimulatedSLM(resolution=cam, pitch_um=8)
            slm.set_source_analytic(sim=True)
            slm.set_source_analytic(sim=False)
            cam = SimulatedCamera(slm=slm, pitch_um=8)

        # Now actually parse the cameras.
        if not hasattr(cam, "get_image"):
            raise ValueError(f"Expected Camera to be passed as cam. Found {type(cam)}")
        self.cam = cam

        if not hasattr(slm, "set_phase"):
            raise ValueError(f"Expected SLM to be passed as slm. Found {type(slm)}")
        self.slm = slm

        self.name = self.cam.name + "-" + self.slm.name
        self.mag = float(mag)

        self.calibrations = {}

        # Initialize logger.
        _Loggable.__init__(self)
        self.log_state()

    def plot(
        self,
        phase=None,
        image=None,
        slm_limits=None,
        cam_limits=None,
        title="",
        axs=None,
        cbar=True,
        **kwargs
    ):
        """
        Plots the provided phase and image for the child hardware on a pair of subplot axes.

        Parameters
        ----------
        phase : ndarray OR None
            Phase to be plotted.
            If ``None``, grabs the last written :attr:`phase` from the SLM.

            Important
            ---------
            Writes this ``phase`` to the SLM if ``image`` is ``None``.
        image : ndarray OR None
            Image to be plotted. If ``None``, grabs an image from the camera.
        slm_limits, cam_limits : None OR float OR [[float, float], [float, float]]
            Scales the limits by a given factor or uses the passed limits directly.
        title : str
            Super title for the axes.
        axs : (matplotlib.pyplot.axis, matplotlib.pyplot.axis) OR None
            Axes to plot upon.
        cbar : bool
            Also plot a colorbar.
        **kwargs
            Passed to :meth:`set_phase()`

        Returns
        -------
        (matplotlib.pyplot.axis, matplotlib.pyplot.axis)
            Axes of the plotted phase and image.
        """
        if image is None and phase is not None and np.shape(phase) == self.slm.shape:
            self.slm.set_phase(phase, **kwargs)


        should_show = False
        if axs is None:
            if len(plt.get_fignums()) > 0:
                fig = plt.gcf()
            else:
                fig = plt.figure(figsize=(20,8))
                should_show = True
            axs = (fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2))
        else:
            fig = None
            if len(axs) != 2:
                raise ValueError(f"Expected axs to be a tuple of two axes. Found length {len(axs)} tuple.")

        self.slm.plot(phase=phase, limits=slm_limits, title="", ax=axs[0], cbar=cbar)
        self.cam.plot(image=image, limits=cam_limits, title="", ax=axs[1], cbar=cbar)

        if fig is not None:
            fig.suptitle(title)
        plt.tight_layout()

        if should_show:
            _slmsuite_plt_show(name="fourierslm_plot")

        return axs


class NearfieldSLM(CameraSLM):
    """
    **(NotImplemented)** Class for an SLM which is not nearly in the Fourier domain of a camera.

    Parameters
    ----------
    mag : number OR None
        Magnification between the plane where the SLM image is created
        and the camera sensor plane.
    """

    def __init__(self, *args, **kwargs):
        """See :meth:`CameraSLM.__init__`."""
        super().__init__(*args, **kwargs)
        self.mag = mag


# Make full class including all calibrations (separated into different files for readability).
class FourierSLM(
    CameraSLM,
    _FourierCalibration,
    _PixelCalibration,
    _SettleCalibration,
    _WavefrontCalibration,
):
    r"""
    Class for an SLM and camera separated by a Fourier transform.
    This class includes methods for system calibration.

    Attributes
    ----------
    calibrations : dict
        "fourier" : dict
            The affine transformation that maps between
            the k-space of the SLM (kxy) and the pixel-space of the camera (ij).

            See :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate()`.

            This data is critical for much of :mod:`slmsuite`'s functionality.
        "wavefront" : dict
            Raw data for correcting aberrations in the optical system (``phase``) and
            measuring the optical amplitude distribution incident on the SLM (``amp``).

            See
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate_zernike()`
            and
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate_superpixel()`
            Usable data for the superpixel implementation is produced by running
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_superpixel_process()`.

            This data is critical for crisp holography.
        "pixel" : dict
            Raw data for measuring the crosstalk and :math:`V_\pi` of sections of the
            SLM via measurements on the diffractive orders of binary gratings.

            See
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.pixel_calibrate()`.
            Usable data is produced by running
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.pixel_calibration_process()`.

            **This data is currently unused; exploring
            computationally-efficient ways to apply the crosstalk without oversampling.**
        "settle" : dict
            Raw data for determining the temporal system response of the SLM.

            See
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.settle_calibrate()`.
            Usable data is produced by running
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.settle_calibration_process()`.

            This data informs the user's choice of `settle_time_s`, the time to wait to
            acquire data after a pattern is displayed. This is, of course, a tradeoff
            between measurement speed and measurement precision.
    """
    _pickle = ["name", "cam", "slm", "mag"]
    _pickle_data = ["calibrations"]

    def __init__(self, *args, **kwargs):
        r"""See :meth:`CameraSLM.__init__`."""
        super().__init__(*args, **kwargs)

        # Size of the calibration point window relative to the spot radius.
        self._wavefront_calibration_window_multiplier = 4

    def simulate(self):
        """
        Clones the hardware-based experiment into a simulation.

        Note
        ~~~~
        Since simulation mode needs the Fourier relationship between the SLM and
        camera, the :class:`~slmsuite.hardware.cameraslms.FourierSLM` should be
        Fourier-calibrated prior to cloning for simulation.

        Returns
        -------
        FourierSLM
            A :class:`~slmsuite.hardware.cameraslms.FourierSLM` object with simulated
            hardware.
        """
        # Make sure we have a Fourier calibration.
        if not "fourier" in self.calibrations:
            raise ValueError("Cannot simulate() a FourierSLM without a Fourier calibration.")

        # Make a simulated SLM
        slm_sim = SimulatedSLM(
            self.slm.shape[::-1],
            source=self.slm.source,
            bitdepth=self.slm.bitdepth,
            name=self.slm.name+"_sim",
            wav_um=self.slm.wav_um,
            wav_design_um=self.slm.wav_design_um,
            pitch_um=self.slm.pitch_um,
        )

        # Make a simulated camera using the current Fourier calibration
        cam_sim = SimulatedCamera(
            slm_sim,
            resolution=self.cam.shape[::-1],
            M=copy.copy(self.calibrations["fourier"]["M"]),
            b=copy.copy(self.calibrations["fourier"]["b"]),
            bitdepth=self.cam.bitdepth,
            averaging=self.cam.averaging,
            hdr=self.cam.hdr,
            pitch_um=self.cam.pitch_um,
            name=self.cam.name+"_sim"
        )
        cam_sim.transform = copy.copy(self.cam.transform)

        #Combine the two and pass FourierSLM attributes from hardware
        fs_sim = FourierSLM(cam_sim, slm_sim)
        fs_sim.calibrations = copy.deepcopy(self.calibrations)
        fs_sim._wavefront_calibration_window_multiplier = self._wavefront_calibration_window_multiplier

        return fs_sim

    @staticmethod
    def load(file_path : str):
        """
        Creates a simulation of a system from a file.

        Returns
        -------
        FourierSLM
            A :class:`~slmsuite.hardware.cameraslms.FourierSLM` object with simulated
            hardware.
        """
        # Read in the file.
        data = load_h5(file_path)

        # Check to see if it has the information we need.
        if not "__meta__" in data:
            raise ValueError(
                f"Cannot interpret file {file_path} without field '__meta__'. "
            )
        if not "cam" in data["__meta__"]:
            raise ValueError(
                f"Cannot interpret file {file_path} without metadata field 'cam'. "
            )
        cam_data = data["__meta__"]["cam"]
        if not "slm" in data["__meta__"]:
            raise ValueError(
                f"Cannot interpret file {file_path} without metadata field 'slm'. "
            )
        slm_data = data["__meta__"]["slm"]

        # Create the SLM and Camera objects.
        slm = SimulatedSLM(
            resolution=np.flip(slm_data["shape"]),
            pitch_um=slm_data["pitch_um"],
        )
        cam = SimulatedCamera(
            slm=slm,
            resolution=np.flip(cam_data["shape"]),
            bitdepth=cam_data["bitdepth"],
            pitch_um=cam_data["pitch_um"],
            name=cam_data["name"],
        )

        fs = FourierSLM(cam, slm, mag=data["__meta__"]["mag"])
        fs.name = data["__meta__"]["name"]

        return fs

    ### Automatic Calibration ###

    def _calibrate(self, verbose=True):
        """
        **(Not Implemented)**
        Attempts to autonomously calibrate the system.
        Conducts any missing calibrations. Also looks for saved calibration files under
        default filenames and loads them if they are found.

        See
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate()`,
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.settle_calibrate()`,
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.pixel_calibrate()`, and
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate_superpixel()`.
        """
        def calibration_detected(calibration_type):
            print(calibration_type.replace("_", " ").capitalize() + " calibration...")
            if calibration_type in self.calibrations:
                if verbose: print(f"Found calibration from {self.calibrations[calibration_type]['timestamp']}.")
                return True
            else:
                try:
                    self.load_calibration(calibration_type)
                    if verbose: print(f"Loaded calibration from {self.calibrations[calibration_type]['timestamp']}.")
                    return True
                except FileNotFoundError:
                    return False
                except Exception as e:
                    warnings.warn(f"Unable to load '{calibration_type}' calibration: {e}")
                    return False

        # Fourier
        if not calibration_detected("fourier"):
            self.fourier_calibrate()

        if not calibration_detected("settle"):
            self.settle_calibrate()

        if not calibration_detected("pixel"):
            self.pixel_calibrate()

        if not calibration_detected("wavefront_superpixel"):
            self.wavefront_calibrate_superpixel()

        print("Fourier calibration (final)...")
        self.fourier_calibrate()

    ### Calibration Helpers ###

    def name_calibration(self, calibration_type):
        """
        Creates ``"{self.name}-{calibration_type}-calibration"``.

        Parameters
        ----------
        calibration_type : str
            The type of calibration to save. See :attr:`calibrations` for supported
            options.

        Returns
        -------
        name : str
            The generated name.
        """
        return f"{self.name}-{calibration_type}-calibration"

    def write_calibration(self, calibration_type, path, name):
        "Backwards-compatibility alias for :meth:`save_calibration()`."
        warnings.warn(
            "The backwards-compatible alias FourierSLM.write_calibration will be depreciated "
            "in favor of FourierSLM.save_calibration in a future release."
        )
        self.save_calibration(calibration_type, path, name)

    def save_calibration(self, calibration_type, path=".", name=None):
        """
        Saves the calibration to a file like ``"path/name_id.h5"``.

        Parameters
        ----------
        calibration_type : str
            The type of calibration to save. See :attr:`calibrations` for supported
            options. Works for any key of :attr:`calibrations`.
        path : str
            Path to directory to save in. Default is current directory.
        name : str OR None
            Name of the save file. If ``None``, will use :meth:`name_calibration`.

        Returns
        -------
        str
            The file path that the calibration was saved to.
        """
        if not calibration_type in self.calibrations:
            raise ValueError(
                f"Could not find calibration '{calibration_type}' in calibrations. Options:\n"
                + str(list(self.calibrations.keys()))
            )

        if name is None:
            name = self.name_calibration(calibration_type)
        file_path = generate_path(path, name, extension="h5")
        save_h5(file_path, self.calibrations[calibration_type])

        return file_path

    def read_calibration(self, calibration_type, file_path=None):
        "Backwards-compatibility alias for :meth:`load_calibration()`."
        warnings.warn(
            "The backwards-compatible alias FourierSLM.read_calibration will be depreciated "
            "in favor of FourierSLM.load_calibration in a future release."
        )
        self.load_calibration(calibration_type, file_path)

    def load_calibration(self, calibration_type, file_path=None):
        """
        Loads the calibration from a file.

        Parameters
        ----------
        calibration_type : str
            The type of calibration to load. See :attr:`calibrations` for supported
            options.
        file_path : str OR None
            Full path to the calibration file. If ``None``, will
            search the current directory for a file with a name like
            the one returned by :meth:`name_calibration`.

        Returns
        -------
        str
            The file path that the calibration was loaded from.

        Raises
        ------
        FileNotFoundError
            If a file is not found.
        """
        if file_path is None:
            path = os.path.abspath(".")

            if len(calibration_type) > 4 and calibration_type[-3:] == ".h5":
                file_path = calibration_type
                split = file_path.split("-")
                if len(split) > 3 and "calibration_" in split[-1]:
                    calibration_type = split[-2]
                else:
                    raise ValueError(
                        f"Could not parse calibration type from '{file_path}'."
                    )
            else:
                name = self.name_calibration(calibration_type)
                file_path = latest_path(path, name, extension="h5")

            if file_path is None:
                raise FileNotFoundError(
                    "Unable to find a calibration file like\n{}"
                    "".format(os.path.join(path, name))
                )

        self.calibrations[calibration_type] = cal = load_h5(file_path)
        cal_ver = "an unknown version" if not "__version__" in cal else cal["__version__"]

        if cal_ver != __version__:
            warnings.warn(
                f"You are using slmsuite {__version__}, but the calibration "
                f"in '{file_path}' was created in {cal_ver}."
            )

        return file_path

    def _get_calibration_metadata(self):
        return self.pickle(attributes=False, metadata=True)      # Pickle without heavy data.

FourierSLM.fourier_calibration_build.__doc__ = SimulatedCamera.build_affine.__doc__
