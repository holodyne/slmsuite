"""
Simulated camera to image the simulated SLM.
"""

import warnings

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
except:
    cp = np
    from scipy.ndimage import map_coordinates

import matplotlib.pyplot as plt

from slmsuite.hardware.cameras.camera import Camera
from slmsuite.holography import toolbox
from slmsuite.holography.algorithms import Hologram
from slmsuite.misc.math import INTEGER_TYPES, REAL_TYPES


class SimulatedCamera(Camera):
    """
    Simulated camera.

    Outputs simulated images (i.e., the far-field of an SLM interpolated to
    camera pixels based on the camera's location and orientation.
    Serves as a future testbed for simulation of other imaging artifacts, including non-affine
    aberrations (e.g. pincushion distortion) and imaging readout noise.

    Note
    ~~~~
    For fastest simulation, initialize :class:`SimulatedCamera` with a
    :class:`~slmsuite.hardware.slms.simulated.SimulatedSLM` *only*. Simulated camera images
    will directly sample the (quickly) computed SLM far-field (``"knm"``) via a one-to-one
    mapping instead of interpolating the SLM's far-field intensity at
    each camera pixel location (i.e. ``"knm"``->``"ij"`` basis change),
    which may also require additional padding (computed automatically upon initialization) for
    sufficient resolution.

    Attributes
    ----------
    grid : (numpy.ndarray, numpy.ndarray)
        Pixel column/row number (``x_grid``, ``y_grid``) in the ``"ij"`` basis used for
        far-field interpolation.
    shape_padded : (int, int)
        Size of the FFT computational space required to faithfully reproduce the far-field at
        full camera resolution.
    noise : dict
        Dictionary of single-argument functions (returning the normalized noise amplitude
        for any normalized input pixel amplitude) to simulate various noise sources. Currently,
        ``'dark'`` and ``'read'``, representing exposure-dependent dark current/background noise
        and exposure-independent readout noise, respectively, are the only accepted keys.

        Example
        ~~~~~~~
        The following code adds a Gaussian background with 50% mean and 5% standard
        deviation (relative to the dynamic range at the default ``self.exposure_s = 1``) and
        a Poisson readout noise (independent of ``self.exposure_s``) with an average value
        of 20% of the camera's dynamic range.

        .. code-block:: python

            self.noise = {
                'dark': lambda img: np.random.normal(0.5*img, 0.05*img),
                'read': lambda img: np.random.poisson(0.2*img)
            }

    """

    def __init__(self, slm, resolution=None, M=None, b=None, noise=None, pitch_um=None, gain=1, **kwargs):
        """
        Initialize simulated camera.

        Parameters
        ----------
        slm : ~slmsuite.hardware.slms.simulated.SimulatedSLM
            Simulated SLM creating the image.
        resolution : (int, int) OR int
            See :attr:`resolution`. If ``None``, defaults to the resolution of ``slm``.
            If positive ``int``, the resolution is multiplied by this factor.
        M, b : array_like
            Passed to :meth:`set_affine()`. Can be set later, but the camera cannot be
            used until then.
        noise : dict
            See :attr:`noise`.
        pitch_um : (float, float) OR None
            Pixel pitch in microns. If ``None``, certain calibrations and conversions
            are not available (e.g. :meth:`build_affine()` for certain units).
        gain : float
            Gain to emulate physical cameras while keeping the same values for exposure time.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """

        # Store a reference to the SLM: we need this to compute the far-field camera images.
        self._slm = slm

        # Don't interpolate (slower) by default unless required.
        self._interpolate = False

        # If resolution is a positive scalar, assume it is a scaling factor
        if resolution is None:
            resolution = 1
        if isinstance(resolution, INTEGER_TYPES):
            resolution = Hologram.get_padded_shape(slm.shape, padding_order=resolution)[::-1]
        elif any([r != s for r, s in zip(resolution, slm.shape[::-1])]):
            self._interpolate = True

        # dtype and other parameters are set here in init.
        super().__init__(resolution, pitch_um=pitch_um, **kwargs)

        # Digital gain emulates exposure
        self.gain = gain

        # Add user-defined noise dictionary
        self.noise = noise

        # Compute the camera pixel grid in `basis` units (currently "ij")
        self.grid = np.meshgrid(
            np.arange(resolution[0]),
            np.arange(resolution[1]),
        )

        # Defaults to alignment with the SLM grid.
        self.set_affine(M, b)

    def close(self):
        pass

        self.set_woi(None)

    def set_affine(self, M=None, b=None, **kwargs):
        """
        Set the camera's placement in the SLM's k-space. ``M`` and/or ``b``, if provided,
        are used to transform the :class:`SimulatedCamera`'s ``"ij"`` grid to a ``"knm"`` grid
        for interpolation against the :class:`~slmsuite.hardware.slms.simulated.SimulatedSLM`'s
        ``"knm"`` grid. Keyword arguments, if provided, are passed to :meth:`.build_affine()`
        to build ``M`` and ``b``.

        Parameters
        ----------
        M : array_like
            2 x 2 affine transform matrix to convert between SLM's :math:`k`-space and the
            simulated camera's pixel basis (``"ij"``). If ``None``, defaults to the
            identity matrix.
        b : array_like
            Lateral displacement (in pixels) of the camera center from the SLM's
            optical axis. If ``None``, defaults to ``(0,0)`` offset.
        **kwargs : dict, optional
            Various orientation parameters passed to :meth:`.build_affine()`
            to build ``M`` and ``b``, if not provided. See options documented in this
            method. ``f_eff`` is a required keyword.
        """

        # If kwargs are passed instead of M and b, use these to build M, b
        if M is None or b is None:
            f_eff = kwargs.pop("f_eff", None)
            if f_eff is not None:
                M, b = self.build_affine(f_eff, **kwargs)

        self._interpolate = not (M is None or b is None)
        self.grid = np.meshgrid(
            np.arange(self.shape[1]),
            np.arange(self.shape[0]),
        )
        self.shape_padded = self._slm.shape

        if self._interpolate:
            self.M = M
            self.b = b

            # Affine transform the camera grid ("ij"->"kxy")
            self.grid = toolbox.transform_grid(self, M, b, direction="rev")

            # Fourier space must be sufficiently padded to resolve the camera pixels.
            dkxy = np.sqrt(
                (self.grid[0][:2, :2] - self.grid[0][0, 0]) ** 2 +
                (self.grid[1][:2, :2] - self.grid[1][0, 0]) ** 2
            )
            dkxy_min = dkxy.ravel()[1:].min()

            self.shape_padded = Hologram.get_padded_shape(self._slm, precision=dkxy_min)

            # Convert kxy -> knm (0,0 at corner): 1/dx -> Npx
            self.knm_cam = cp.array(
                [
                    self.shape_padded[0] * self._slm.pitch[1] * self.grid[1] + self.shape_padded[0] / 2,
                    self.shape_padded[1] * self._slm.pitch[1] * self.grid[0] + self.shape_padded[1] / 2,
                ]
            )

            if (
                cp.amax(cp.abs(self.knm_cam[0] - self.shape_padded[0]/2)) > self.shape_padded[1]/2 or
                cp.amax(cp.abs(self.knm_cam[1] - self.shape_padded[1]/2)) > self.shape_padded[0]/2
            ):
                warnings.warn(
                    "Camera extends beyond the accessible SLM k-space;"
                    " some pixels may not be targetable."
                )

        phase = -self._slm.display.astype(float) * (2 * np.pi / self._slm.bitresolution)

        # Suppress power of 2 warning from Hologram.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)

            self._hologram = Hologram(
                self.shape_padded,
                amp=self._slm.source["amplitude_sim"],
                phase=phase - phase.min() + self._slm.source["phase_sim"],
                slm_shape=self._slm,
            )

    def build_affine(
            self,
            f_eff,
            units="norm",
            theta=0,
            shear_angle=0,
            offset=None,
        ):
        """
        Builds an affine transform defining the SLM to camera transformation as
        detailed in :meth:`~slmsuite.hardware.cameraslms.FourierSLM.kxyslm_to_ijcam`.

        Parameters
        ----------
        f_eff : float OR (float, float)
            Effective focal length of the
            optical train separating the Fourier-domain SLM from the camera. If a ``float`` is
            provided, ``f_eff`` is isotropic; otherwise, ``f_eff`` is defined along the SLM's
            :math:`x` and :math:`y` axes.
        units : str {"norm", "ij", "m", "cm", "mm", "um", "nm"}
            Units for the focal length ``f_eff``.

            -  ``"norm"``
                Normalized focal length in wavelengths according to the SLM's
                :attr:`~slmsuite.hardware.slms.slm.SLM.wav_um`.
                This is the default unit.
            -  ``"ij"``
                Focal length in units of camera pixels.
            -  ``"m"``, ``"cm"``, ``"mm"``, ``"um"``, ``"nm"``
                Focal length in metric units.

        theta : float
            Rotation angle (in radians, ccw) of the camera relative to the SLM orientation.
            Defaults to zero (i.e., aligned with the SLM).
        shear_angle : float OR (float, float)
            Shearing angles (in radians) along the SLM's :math:`x` and :math:`y` axes.
            If a ``float`` is provided, shear is applied isotropically.
            Defaults to zero (i.e., no shear).
        offset : (float, float) OR None
            Lateral displacement (in pixels units) of the SLM's optical axis
            from the camera's origin. If ``None``, defaults to be centered on the center
            of the camera.

        Returns
        -------
        numpy.ndarray
            Affine matrix :math:`M`. Shape ``(2, 2)``.
        numpy.ndarray
            Affine vector :math:`b`. Shape ``(1, 2)``.
        """
        if offset is None:
            offset = np.flip(self.shape) / 2

        return SimulatedCamera._build_affine(
            f_eff,
            units=units,
            theta=theta,
            shear_angle=shear_angle,
            offset=offset,
            cam_pitch_um=self.pitch_um,
            wav_um=self._slm.wav_um,
        )

    @staticmethod
    def _build_affine(
            f_eff,
            units="ij",
            theta=0,
            shear_angle=0,
            offset=(0,0),
            cam_pitch_um=None,
            wav_um=None,
        ):
        """
        See documentation in :meth:`build_affine()` and
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.build_fourier_calibration()`.
        This helper function is shared between those functions.
        """
        # Parse scalars.
        if isinstance(f_eff, REAL_TYPES):
            f_eff = [f_eff, f_eff]
        if isinstance(cam_pitch_um, REAL_TYPES):
            cam_pitch_um = [cam_pitch_um, cam_pitch_um]
        else:
            cam_pitch_um = np.ravel(cam_pitch_um)
        if isinstance(shear_angle, REAL_TYPES):
            shear_angle = [shear_angle, shear_angle]
        if offset is None:
            offset = (0,0)

        f_eff = np.squeeze(f_eff).astype(float)
        shear_angle = np.squeeze(shear_angle)

        # Convert.
        if units == "ij":
            pass
        elif units == "norm":
            if wav_um is None:
                raise ValueError(f"wav_um is required for unit '{units}'")
            if cam_pitch_um is None or cam_pitch_um[0] is None:
                raise ValueError(f"cam_pitch_um is required for unit '{units}'")

            f_eff *= wav_um / np.squeeze(cam_pitch_um)
        elif units in toolbox.LENGTH_FACTORS.keys():
            if cam_pitch_um is None or cam_pitch_um[0] is None:
                raise ValueError(f"cam_pitch_um is required for unit '{units}'")

            f_eff *= toolbox.LENGTH_FACTORS[units] / np.squeeze(cam_pitch_um)
        else:
            raise ValueError(f"Unit '{units}' not recognized as a length.")

        mag = np.array([[f_eff[0], 0], [0, f_eff[1]]])
        shear = np.array([[1, np.tan(shear_angle[0])], [np.tan(shear_angle[1]), 1]])
        rot = np.array([[np.cos(-theta), np.sin(-theta)], [-np.sin(-theta), np.cos(-theta)]])

        M = mag @ shear @ rot
        b = toolbox.format_2vectors(offset)

        return M, b

    def flush(self, timeout_s=1):
        """
        See :meth:`.Camera.flush`.
        """
        pass

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        return self.exposure_s

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        self.exposure_s = exposure_s

    def _get_image_hw(self, timeout_s):
        """
        See :meth:`.Camera._get_image_hw`. Computes and samples the affine-transformed SLM far-field.

        Returns
        -------
        numpy.ndarray
            Array of shape :attr:`shape`
        """
        if not hasattr(self, "_hologram"):
            raise RuntimeError(
                "Cannot display SimulatedCamera before affine transformation is defined."
            )

        # Update phase; calculate the far-field (keep on GPU if using cupy for follow-on interp)
        # FUTURE: in the case where sim is being used inside a GS loop, there could be
        # something clever here to use the existing Hologram's data.

        # Analog phase
        # self._hologram.reset_phase(self._slm.phase + self._slm.source["phase_sim"])

        # Quantized phase
        self._hologram.amp = cp.array(self._slm.source["amplitude_sim"], dtype=self._hologram.dtype)
        phase = -self._slm.display.astype(self._hologram.dtype) * (2 * np.pi / self._slm.bitresolution)
        self._hologram.reset_phase(phase - phase.min() + self._slm.source["phase_sim"].astype(self._hologram.dtype))

        ff = self._hologram.get_farfield(get=False)

        # Use map_coordinates for fastest interpolation
        # Note: by default, map_coordinates sets pixels outside the SLM k-space to 0 as desired
        if self._interpolate:
            img = map_coordinates(cp.abs(ff) ** 2, self.knm_cam, order=0)
        else:
            img = cp.abs(ff) ** 2
            img = toolbox.unpad(img, self.shape)
        if cp != np:
            img = img.get()

        img *= self.exposure_s * self.gain

        # Basic noise sources.
        if self.noise is not None:
            for key in self.noise.keys():
                if key == 'dark':
                    # Background/dark current - exposure dependent
                    dark = self.noise['dark'](np.ones_like(img) * self.bitresolution) / self.exposure_s
                    img = img + dark
                elif key == 'read':
                    # Readout noise - exposure independent
                    read = self.noise['read'](np.ones_like(img) * self.bitresolution)
                    img = img + read
                else:
                    raise RuntimeError('Unknown noise source %s specified!'%(key))

        # Truncate to maximum readout value
        img[img > self.bitresolution-1] = self.bitresolution-1

        # Crop to maximum window of interest
        if self.woi is not None:
            x, width, y, height = self.woi
            img = img[y:height, x:width]

        # Quantize: all power in one pixel (img=1) -> maximum readout value at base exposure=1
        # img = np.rint(img)

        return img.astype(self.dtype)

    def set_woi(self, woi=None):
        """
        Sets the imaging region to a 'window of interest' to simulate
        a limited field of view of the imaging plane.

        Parameters
        ----------
        woi : list, None
            See :attr:`~slmsuite.hardware.cameras.camera.Camera.woi`.
            If ``None``, defaults to largest possible.

        Returns
        ----------
        woi : list
            :attr:`~slmsuite.hardware.cameras.camera.Camera.woi`.
        """
        self.woi = woi

        return woi