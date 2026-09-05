"""
Abstract camera functionality.
"""
import time
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import logging

from slmsuite._plotting import _slmsuite_plt_show
from scipy.optimize import curve_fit

from slmsuite.hardware._common import _Common
from slmsuite.holography import analysis
from slmsuite.holography.toolbox import BLAZE_LABELS, format_shape, window_slice
from slmsuite.holography.toolbox.phase import zernike
from slmsuite.misc.fitfunctions import lorentzian
from slmsuite.misc.math import INTEGER_TYPES, REAL_TYPES
from slmsuite.misc.xp import as_backend, as_numpy, is_gpu_array


class Camera(_Common, ABC):
    """
    Abstract class for cameras.
    Comes with transformations, averaging,  HDR,
    and helper functions like :meth:`.autoexpose()`.

    Attributes
    ----------
    name : str
        Camera identifier.
    shape : (int, int)
        ``(height, width)`` of the image returned by :meth:`.get_image()`, in the same
        order as :attr:`numpy.ndarray.shape`. Read-only property derived from the current
        WOI, binning, and orientation transform (a 90/270 rotation swaps the two), so
        ``get_image().shape == shape`` always holds.
    bitdepth : int
        Depth of a camera pixel well in bits.
    bitresolution : int
        Returns ``(2**bitdepth) * averaging``. The action of averaging here is a sum
        rather than a mean, so the effective bitresolution increases accordingly.
    dtype : np.dtype
        Type returned by :meth:`._get_image_hw()`, probed and cached upon initialization.
        Falls back to the narrowest type fitting :attr:`bitdepth` if the probe fails.
    pitch_um : (float, float) OR None
        Pixel pitch in microns.
        This is a property that updates with binning.
    exposure_s : float
        Caches the last result of :meth:`.get_exposure()`. Can be used if the user wants to
        avoid the overhead of calling the method.
    exposure_bounds_s : (float, float) OR None
        Shortest and longest allowable integration in seconds.
    averaging : int OR None
        Default setting for averaging (sums repeated measurements). See :meth:`.get_image()`.
    hdr : (int, int) OR None
        Default setting for multi-exposure High Dynamic Range imaging. See :meth:`.get_image()`.
    color_channel : None OR int
        For color cameras, selects which channel of a ``(height, width, channels)``
        frame to keep as the grayscale image. See :meth:`.get_image()`.
        If ``int``, returns the corresponding color channel.
        If ``None``, defaults to 0.
    capture_attempts : int
        If the camera returns an error or exceeds a timeout,
        try again for a total of ``capture_attempts`` attempts.
        This is useful for resilience against errors that happen with low probability.
        Defaults to 5.
    binning : (int, int)
        Binning of the camera in the transformed orientation. Defaults to (1, 1) for no binning.
    woi : (int, int, int, int)
        WOI (window of interest) in ``(x, width, y, height)`` form, in the
        **transformed, unbinned** frame (the orientation the user sees, at full sensor
        resolution). Stored unbinned because the WOI marks a physical sensor region
        independent of :attr:`binning`; this keeps it binning-invariant so
        ``set_woi(get_woi())`` round-trips and the Fourier calibration stays valid. The
        binned output size is reported by :attr:`shape` instead.

        Warning
        ~~~~~~~
        This feature is less fleshed out than most. There may be issues
        (e.g. :meth:`.get_image()` with the ``averaging`` or ``hdr`` flags).
    origin : (int, int)
        Read-only ``(x, y)`` of the image's upper-left corner (the WOI offset), in the
        same frame as :attr:`woi`. ``(0, 0)`` for a full-sensor WOI.
    center : (float, float)
        Read-only ``(x, y)`` center of the returned image, ``(shape[1]/2, shape[0]/2)``.
    transform : callable
        Orientation transform (:class:`~slmsuite.holography.analysis.OrientationTransform`)
        of flips and 90 degree rotations, applied to raw camera frames before they are
        returned to the user. Built from the ``rot``, ``fliplr``, and ``flipud`` arguments
        to :meth:`__init__`.
    last_image : numpy.ndarray OR cupy.ndarray OR None
        Last captured image. Note that this is a pointer to the same data that the user
        receives (to avoid copying overhead). Thus, if the user modifies the returned data,
        then this data will be modified also. It sits on whichever backend the last
        :meth:`get_image()` returned, i.e. the device if that call passed ``get=False``.
        This may be of :attr:`dtype`, or may be a float, depending on whether :attr:`hdr` is
        used and the type of :attr:`averaging`.
        Is ``None`` if no image has ever been taken.
    """
    _pickle = [
        "name",
        "shape",
        "bitdepth",
        "bitresolution",
        "pitch_um",
        "exposure_s",
        "exposure_bounds_s",
        "averaging",
        "hdr",
        "woi",
        "_shape",
        "_software_woi",
        "_software_binning",
    ]
    _pickle_data = [
        "last_image",
    ]

    @abstractmethod
    def __init__(
        self,
        resolution,
        bitdepth=8,
        pitch_um=None,
        name="",
        exposure_bounds_s=None,
        averaging=None,
        hdr=None,
        color_channel=None,
        capture_attempts=5,
        rot="0",
        fliplr=False,
        flipud=False,
    ):
        """
        Initializes a camera.

        In addition to the other class attributes, accepts the following parameters
        to set :attr:`transform`. See :meth:`~slmsuite.holography.analysis.get_orientation_transformation()`.

        Parameters
        ----------
        resolution
            The width and height of the camera in ``(width, height)`` form.

            Important
            ~~~~~~~~~
            This is the opposite of the numpy ``(height, width)``
            convention stored in :attr:`shape`.
        bitdepth
            See :attr:`bitdepth`.
        pitch_um : (float, float) OR None
            Extra information about the pitch of a single pixel ``(dx_um, dy_um)``
            to use additional calibrations.
        name : str
            Defaults to the class name.
        exposure_bounds_s : (float, float) OR None
            Exposure bounds in seconds for the camera. If ``None``, no software bounds are applied.
        averaging : int or None
            Number of frames to average. Used to increase the effective bit depth of a camera by using
            pre-quantization noise (e.g. dark current, read-noise, etc.) to "dither" the pixel output
            signal. If ``None``, no averaging is performed.
        hdr : int OR (int, int) OR None OR False
            Exposure information for `Multi-exposure High Dynamic Range (HDR) imaging
            <https://en.wikipedia.org/wiki/Multi-exposure_HDR_capture>`_
        color_channel : None OR int
            For color cameras, selects which channel of a ``(height, width, channels)``
            frame to keep as the grayscale image. If ``int``, returns that channel index.
            If ``None``, defaults to channel ``0``. A weighted combination of channels
            (via a list of floats) is planned but not yet implemented.
        capture_attempts : int
            If the camera returns an error or exceeds a timeout,
            try again for a total of ``capture_attempts`` attempts.
            This is useful for resilience against errors that happen with low probability.
            Defaults to 5.
        rot : str or int
            Rotates returned image by the corresponding degrees in ``["90", "180", "270"]``
            or :meth:`numpy.rot90` code in ``[1, 2, 3]``. Defaults to no rotation.
            Used to determine :attr:`shape` and :attr:`transform`.
        fliplr : bool
            Flips returned image left right.
            Used to determine :attr:`transform`.
        flipud : bool
            Flips returned image up down.
            Used to determine :attr:`transform`.
        """
        width, height = format_shape(resolution)

        # Create image transformation. Now shape properties can be used.
        self.transform = analysis.get_orientation_transformation(rot, fliplr, flipud)

        # Set woi, binning, and shape variables
        # These are stored in **untransformed** raw camera coordinates.
        self._binning = (1, 1)
        self._woi = (0, width, 0, height)
        self._shape = (height, width)

        # Detect hardware WOI / binning support (prefer hardware if
        # defined in camera class, fall back to software).
        self._software_woi = type(self)._set_woi_hw is Camera._set_woi_hw
        self._software_binning = type(self)._set_binning_hw is Camera._set_binning_hw

        # Parse capture_attempts.
        self.capture_attempts = int(capture_attempts)
        if self.capture_attempts <= 0:
            raise ValueError("capture_attempts must be positive.")

        # Set exposure information.
        self.exposure_bounds_s = (
            (np.min(exposure_bounds_s), np.max(exposure_bounds_s))
            if exposure_bounds_s is not None else
            None
        )
        self._exposure_s = 1     # Default to 1s for Simulated cameras.

        # Update other camera attributes from hardware.
        self.get_exposure()
        self.get_binning()
        self.get_woi()

        self.color_channel = color_channel

        # Frame averaging variables.
        self.averaging = self._parse_averaging(averaging, preserve_none=True)
        self.hdr = self._parse_hdr(hdr, preserve_none=True)
        self._flush_iterations = 2  # Hidden variable: how many frames to capture for a flush.

        # Initialize the common hardware attributes.
        _Common.__init__(
            self,
            resolution=resolution,
            bitdepth=bitdepth,
            name=name,
            pitch_um=pitch_um,
            is_slm=False,
        )

        # Variable for storing the last capture; None until one is taken.
        self.last_image = None

    @property
    def bitresolution(self) -> int:
        # This overwrites the _Common bitresolution, as averaging and software binning
        # both sum into a range wider than the bitdepth.
        return self._bitresolution(self._parse_averaging(self.averaging))

    def _bitresolution(self, averaging) -> int:
        """:attr:`bitresolution` for a capture using ``averaging`` instead of the attribute."""
        scale = averaging
        if self._software_binning:
            scale *= int(np.prod(self._binning))
        return (2**self.bitdepth) * scale

    # Binning methods.

    @property
    def pitch_um(self) -> np.ndarray | None:
        """Returns the pixel pitch in micrometers (potentially after binning)."""
        if self._pitch_um is not None:
            # Report in the transformed frame: transform the raw pitch the same way as
            # binning, so a 90/270 swap pairs the correct pitch and binning per axis.
            pitch = self.transform.transform_shape(self._pitch_um)
            binning = self.binning
            return np.array([pitch[0] * binning[0], pitch[1] * binning[1]])
        else:
            return None

    @pitch_um.setter
    def pitch_um(self, value):
        self._pitch_um = value

    @property
    def binning(self) -> tuple[int, int]:
        """Returns the current binning."""
        return self.transform.transform_shape(self._binning)

    @binning.setter
    def binning(self, value):
        # set_binning() handles parsing, the transform, and the no-op short circuit.
        self.set_binning(value)

    def set_binning(self, binning: int | tuple[int, int] = 1, update_woi=True):
        """
        Set pixel binning in the transformed orientation. See :attr:`transform`.

        Parameters
        ----------
        binning : int or (int, int)
            Binning factor as ``(binx, biny)``.
            If a single integer is provided, uses the same binning for both dimensions.
        update_woi : bool
            Whether or not to adjust the WOI according to the new binning.
        """
        # Parse binning.
        if isinstance(binning, INTEGER_TYPES):
            binning = (binning, binning)

        target_binning = binning
        binning = self.transform.transform_shape(binning)

        # Break if no change.
        if self._binning == binning:
            return

        # Save original WOI.
        old_woi = self.woi
        old_shape = self.shape

        if not self._software_binning:
            # Send it off to the hardware.
            self._set_binning_hw(binning)
            # Never trust that it succeeded.
            self._binning = self._get_binning_hw()
        else:
            self._binning = binning

        if self._binning != binning:
            self.logger.warning("Attempted to set binning to %s, but realized %s.", target_binning, self.binning)
        else:
            self.logger.debug("Set binning to %s.", self.binning)

        # Try and retain the same WOI in unbinned camera coordinates.
        if update_woi:
            try:
                self.set_woi(old_woi)
            except Exception:
                pass

        # Erase last_image if the shape or WOI changed, since the old image would no longer be valid.
        if self.shape != old_shape or self.woi != old_woi:
            self.last_image = None

    def get_binning(self):
        """
        Returns the current binning.
        """
        if not self._software_binning:
            self._binning = self._get_binning_hw()

        return self.transform.transform_shape(self._binning)

    def _set_binning_hw(self, binning: tuple[int, int]):
        raise NotImplementedError(f"Camera {self.name} has not implemented binning")

    def _get_binning_hw(self):
        raise NotImplementedError(f"Camera {self.name} has not implemented binning")

    # WOI methods.

    @property
    def shape(self):
        """
        Returns ``(height, width)`` of images returned by :meth:`.get_image()`.

        Accounts for the current WOI, binning, and orientation transform so that
        ``get_image().shape == camera.shape`` always holds.
        """
        h_bin = self._woi[3] // self._binning[1]
        w_bin = self._woi[1] // self._binning[0]
        return self.transform.transform_shape((h_bin, w_bin))

    @shape.setter
    def shape(self, _):
        pass  # derived from _woi and _binning; _Common.__init__ sets this

    @property
    def origin(self):
        """
        Returns the ``(x, y)`` coordinate of the upper-left corner of the image in pixels,
        i.e. the WOI offset in transformed, unbinned coordinates. Equals ``(0, 0)`` when
        the WOI covers the full sensor. Together with :attr:`center`, this lives in the same
        coordinate frame as the :attr:`woi` property.
        """
        woi = self.woi
        return (woi[0], woi[2])

    @property
    def center(self):
        """
        Returns the ``(x, y)`` coordinate of the center of the image in pixels,
        following the ``(shape[1] / 2, shape[0] / 2)`` convention used throughout
        the package (e.g. the default Fourier-calibration offset). Accounts for the
        current WOI, binning, and orientation transform via :attr:`shape`.
        """
        shape = self.shape
        return (shape[1] / 2, shape[0] / 2)

    @property
    def _woi_untransformed_binned(self):
        """
        Returns the WOI ``(x, w, y, h)`` in raw binned camera coordinates.
        """
        return (
            self._woi[0] // self._binning[0],    # x / binx
            self._woi[1] // self._binning[0],    # w / binx
            self._woi[2] // self._binning[1],    # y / biny
            self._woi[3] // self._binning[1],    # h / biny
        )

    @property
    def woi(self):
        """
        Returns the WOI ``(x, w, y, h)`` in transformed, unbinned pixel coordinates.

        This is the same coordinate convention accepted by :meth:`set_woi` and returned
        by :meth:`get_woi`, so it is invariant under changes to :attr:`binning`.
        """
        return self.transform.transform_woi(
            self._woi,
            shape=self._shape,
            binning_in=1,
            binning_out=1,
        )

    def _set_woi_hw(self, woi):
        """
        Sets the WOI on hardware in raw camera coordinates (untransformed, binned).
        If the camera expects unbinned coordinates, the woi should be multiplied by the
        untransformed binning factor ``_binning``.
        """
        raise NotImplementedError(f"Camera {self.name} has not implemented WOI")

    def _get_woi_hw(self):
        """
        Gets the WOI from hardware in raw camera coordinates (untransformed, binned).
        If the camera returns unbinned coordinates, the woi should be multiplied by the
        untransformed binning factor ``_binning``.
        """
        raise NotImplementedError(f"Camera {self.name} has not implemented WOI")

    def set_woi(self, woi=None):
        """
        Set the window of interest (WOI) for the camera.

        Cameras without hardware WOI support use a software crop after each capture.
        Cameras with hardware WOI support (``_set_woi_hw`` overridden) call that instead.

        Parameters
        ----------
        woi : int or (int, int) or (int, int, int, int) or None
            Window of interest in **transformed, unbinned** pixel coordinates:
            - ``None``: resets to the full sensor.
            - ``int``: centered square window ``(size, size)`` on the sensor.
            - ``(w, h)``: centered rectangular window ``(w, h)`` on the sensor.
            - ``(x0, w, y0, h)``: explicitly placed window.

        Returns
        -------
        (int, int, int, int)
            :attr:`~slmsuite.hardware.cameras.camera.Camera.woi` after the update.
        """
        old_woi = self.woi
        old_shape = self.shape

        binx, biny = self._binning[0], self._binning[1]
        transformed_shape = self.transform.transform_shape(self._shape)
        H, W = transformed_shape

        if woi is None:
            # Full sensor in untransformed, unbinned coordinates.
            woi_unt = (0, self._shape[1], 0, self._shape[0])
            # Full WOI in transformed, unbinned coordinates (the frame of self.woi)
            # for the realized-WOI comparison below. self.shape is binned, so using
            # it here spuriously trips the warning whenever binning != 1.
            woi = (0, W, 0, H)
        else:
            if isinstance(woi, (int, np.integer)):
                (w, h) = (int(woi), int(woi))
            elif isinstance(woi, (list, tuple, np.ndarray)):
                if len(woi) == 2:
                    (w, h) = (int(woi[0]), int(woi[1]))
                elif len(woi) == 4:
                    (w, h) = (None, None)
                    woi = tuple(int(v) for v in woi)
                else:
                    raise ValueError(
                        "Expected WOI as int, 2-tuple (w, h), or 4-tuple (x0, w, y0, h). "
                        f"Got {woi}."
                    )
            else:
                raise TypeError(f"Cannot interpret WOI of type {type(woi).__name__}.")

            if w is not None:   # Center the (w, h) window on the sensor.
                woi = (max(0, (W - w) // 2), w, max(0, (H - h) // 2), h)

            # Get the WOI in raw camera coordinates.
            woi_unt = self.transform.inverse_woi(woi, transformed_shape)

            # Clip to sensor bounds.
            x0 = max(0, int(woi_unt[0]))
            y0 = max(0, int(woi_unt[2]))
            x1 = min(self._shape[1], x0 + int(woi_unt[1]))
            y1 = min(self._shape[0], y0 + int(woi_unt[3]))
            clipped = (x0, x1 - x0, y0, y1 - y0)

            if clipped != tuple(int(v) for v in woi_unt):
                warnings.warn(
                    f"Requested WOI {woi} extends beyond the sensor and was clipped."
                )

            woi_unt = clipped

        # Store untransformed, unbinned WOI.
        self._woi = woi_unt

        if not self._software_woi:
            # Pass untransformed, binned coordinates to the subclass.
            # The subclass might muliply by binning again depending on the camera.
            self._set_woi_hw(self._woi_untransformed_binned)
            # Read back (hardware may snap to allowed boundaries)
            woi_hw = self._get_woi_hw()
            self._woi = (
                woi_hw[0] * binx,
                woi_hw[1] * binx,
                woi_hw[2] * biny,
                woi_hw[3] * biny,
            )
        # else: handled by _crop_to_woi()

        new_woi = self.woi

        # Erase last_image if the shape or WOI changed, since the old image would no longer be valid.
        if self.shape != old_shape or new_woi != old_woi:
            self.last_image = None

        if new_woi != woi:
            self.logger.warning("Attempted to set WOI to %s, but realized %s.", woi, new_woi)
        else:
            self.logger.debug("Set WOI to %s.", new_woi)

        return self.woi

    def get_woi(self):
        """
        Get the current WOI in transformed, unbinned pixel coordinates.

        For cameras without hardware WOI support, returns the cached WOI directly.
        For cameras with hardware WOI support, queries the hardware first.
        The returned coordinates match what :meth:`set_woi` accepts, so
        ``set_woi(get_woi())`` is always a valid no-op round-trip.

        Returns
        -------
        (int, int, int, int)
            ``(x0, w, y0, h)`` in transformed, unbinned pixel coordinates —
            the same coordinate system accepted by :meth:`set_woi`.
        """
        if not self._software_woi:
            woi_hw = self._get_woi_hw()
            binx, biny = self._binning[0], self._binning[1]
            self._woi = (
                woi_hw[0] * binx,
                woi_hw[1] * binx,
                woi_hw[2] * biny,
                woi_hw[3] * biny,
            )

        return self.transform.transform_woi(
            self._woi,
            shape=self._shape,
            binning_in=1,
            binning_out=1,
        )

    def _get_ijraw_to_ijcam(self):
        """
        Returns an :class:`~slmsuite.holography.analysis.Affine` mapping raw sensor
        pixel coordinates ``(x=col, y=row)`` to camera-image pixel coordinates,
        accounting for WOI offset, binning, and orientation.

        Stored calibrations use ``ijraw``; this converts them for user-facing ``ijcam``.
        """
        binx, biny = self._binning[0], self._binning[1]
        woi_x = self._woi[0]
        woi_y = self._woi[2]
        w_bin = self._woi[1] // binx
        h_bin = self._woi[3] // biny

        # subtract WOI origin, then divide by binning.
        woi_bin = analysis.Affine(
            np.diag([1.0 / binx, 1.0 / biny]),
            np.array([0.0, 0.0]),
            np.array([float(woi_x), float(woi_y)]),
        )

        # orientation transform with shape-dependent translation.
        return self.transform.affine((h_bin, w_bin)) @ woi_bin

    def _get_ijcam_to_ijraw(self):
        """Inverse of :meth:`_get_ijraw_to_ijcam`."""
        return self._get_ijraw_to_ijcam().inv

    # Info method to discover cameras.

    @staticmethod
    def info(verbose=True):
        """
        Abstract method to load information about what cameras are available.

        Parameters
        ----------
        verbose : bool
            Whether or not to print display information.
        """
        raise NotImplementedError("Camera class has not implemented info()")

    # Exposure methods.

    @property
    def exposure_s(self):
        """Returns the current exposure time in seconds."""
        return self._exposure_s

    @exposure_s.setter
    def exposure_s(self, value : float):
        if self._exposure_s != value:
            self.set_exposure(value)

    def get_exposure(self):
        """
        Get the frame integration time in seconds.
        Used in :meth:`.autoexpose()`.

        Returns
        -------
        float
            Integration time in seconds.
        """
        self._exposure_s = float(self._get_exposure_hw())
        return self._exposure_s

    def set_exposure(self, exposure_s):
        """
        Set the frame integration time in seconds.
        Used in :meth:`.autoexpose()`.

        Parameters
        ----------
        exposure_s : float
            The integration time in seconds.

        Returns
        -------
        float
            The resulting integration time in seconds (pulled from :meth:`.get_exposure()`).
        """
        # Handle exposure bounds if they are defined.
        if self.exposure_bounds_s is not None:
            exposure_s_ = np.clip(exposure_s, *self.exposure_bounds_s)
            if exposure_s_ != exposure_s:
                self.logger.warning(
                    "Requested exposure %s s is out of bounds %s s. Clipping to %s s.",
                    exposure_s, self.exposure_bounds_s, exposure_s_,
                )
                exposure_s = exposure_s_

        # Actually set the exposure on the hardware.
        self._set_exposure_hw(exposure_s)

        # Read back the exposure to see what the hardware actually set.
        self._exposure_s = self.get_exposure()

        # Report to the logger.
        if not np.isclose(self._exposure_s, exposure_s): 
            if abs(self._exposure_s - exposure_s) / self._exposure_s > 0.01:
                warn = self.logger.warning
            else:
                warn = self.logger.debug
            warn("Attempted to set exposure to %s seconds, but realized %s seconds.", exposure_s, self._exposure_s)
        else:
            self.logger.debug("Set exposure to %s s.", self._exposure_s)

        return self._exposure_s

    def _unpickle(self, data):
        """
        Restores pickled state data not restored by constructor. See
        :meth:`~slmsuite._pickling._Picklable._unpickle`. 
        """
        super()._unpickle(data)

        self.set_exposure(data.get("exposure_s", 1))

    @abstractmethod
    def _get_exposure_hw(self):
        """
        Abstract method to interface with hardware and get the frame integration time in seconds.
        Subclasses must implement this.
        """
        raise NotImplementedError(f"Camera {self.name} has not implemented _get_exposure_hw")

    @abstractmethod
    def _set_exposure_hw(self, exposure_s):
        """
        Abstract method to interface with hardware and set the exposure time in seconds.
        Subclasses must implement this.

        Parameters
        ----------
        exposure_s : float
            The integration time in seconds.
        """
        raise NotImplementedError(f"Camera {self.name} has not implemented _set_exposure_hw")

    # Parsers for imaging settings.

    def _parse_averaging(self, averaging=None, preserve_none=False):
        """
        Helper function to get a valid averaging.
        """
        if averaging is None:
            if preserve_none:
                return None
            averaging = 1 if self.averaging is None else self.averaging

        if isinstance(averaging, (bool, np.bool_)) and not averaging:
            return 1

        averaging = int(averaging)

        if averaging <= 0:
            raise ValueError("averaging must be positive.")

        return averaging

    def get_dtype(self, averaging=None, hdr=None, binning=None):
        """
        Return the dtype that :meth:`.get_image()` will produce for the given settings.

        Useful for pre-allocating output buffers with the correct type before capture.

        Parameters
        ----------
        averaging : int or None or False
            Number of frames to sum. ``None`` uses :attr:`averaging` (or 1 if unset).
            ``False`` is equivalent to 1.
        hdr : int or (int, int) or None or False
            HDR exposure settings. ``None`` uses :attr:`hdr`.
            Any active HDR (exposures > 1) forces ``float`` regardless of other settings.
        binning : (int, int) or int or None
            Software binning factor ``(binx, biny)`` to include in the overflow budget.
            ``None`` auto-detects: uses :attr:`_binning` when ``_software_binning`` is
            ``True``, otherwise ``(1, 1)`` (hardware binning does not widen the dtype).
            Pass an explicit value to query a hypothetical configuration.

        Returns
        -------
        dtype
            The numpy dtype of the image returned by :meth:`.get_image()`.
        """
        # HDR always returns float.
        (exposures, _) = self._parse_hdr(hdr)
        if exposures > 1:
            return float

        averaging = self._parse_averaging(averaging)

        # Parse software binning factor.
        if binning is None:
            eff_binning = self._binning if self._software_binning else (1, 1)
        elif np.isscalar(binning):
            eff_binning = (int(binning), int(binning))
        else:
            eff_binning = tuple(int(b) for b in binning)
        bin_factor = eff_binning[0] * eff_binning[1]

        # Integer promotion: check whether averaging * binning fits in the native dtype.
        dtype = np.dtype(self.dtype) if not hasattr(self.dtype, 'kind') else self.dtype
        if dtype.kind in ("i", "u"):
            dtype_bitdepth = 8 * dtype.type(0).nbytes
            if dtype.kind == "i":
                dtype_bitdepth -= 1   # signed integers lose one bit
            extra_bits = int(np.ceil(np.log2(max(1, averaging * bin_factor))))
            if self.bitdepth + extra_bits <= dtype_bitdepth:
                return self.dtype
            else:
                return float
        elif dtype.kind == "f":
            return self.dtype
        else:
            raise ValueError(f"Datatype {self.dtype} does not make sense as a camera return.")

    def _parse_hdr(self, exposures=None, preserve_none=False):
        """
        Helper function to get valid hdr parameters.
        """
        # Parse inputs
        if exposures is None:
            if preserve_none:
                return None
            if not hasattr(self, "hdr") or self.hdr is None:
                (exposures, exposure_power) = (1, 0)
            else:
                (exposures, exposure_power) = self._parse_hdr(self.hdr)
        elif isinstance(exposures, (bool, np.bool_)) and not exposures:
            exposures = 1
            exposure_power = 0
        elif np.isscalar(exposures):
            exposure_power = 2
        else:
            (exposures, exposure_power) = exposures

        # Force int so we have a chance of exposure aligning with camera clock.
        if not float(exposure_power).is_integer():
            warnings.warn(
                f"HDR exposure base {exposure_power} is not an integer; "
                f"truncating to {int(exposure_power)}."
            )
        return (int(exposures), int(exposure_power))

    @property
    def _hw_image_shape(self):
        """
        Shape that :meth:`._get_image_hw` returns before software WOI/binning is applied.

        - Software WOI (``_software_woi=True``): hardware delivers the full sensor,
          so the shape is ``_shape``.
        - Hardware WOI + software binning: hardware crops to the WOI but does not bin,
          so the shape is the unbinned WOI size.
        - Hardware WOI + hardware binning: hardware both crops and bins,
          so the shape is the binned WOI size in the untransformed frame.
        """
        if self._software_woi:
            # Full sensor: crop happens later in _crop_to_woi().
            return self._shape
        # Hardware WOI: image is already cropped to the WOI region.
        h_woi = self._woi[3]
        w_woi = self._woi[1]
        if self._software_binning:
            # Binning happens later in _crop_to_woi(): return unbinned WOI size.
            return (h_woi, w_woi)
        # Hardware binning: return the WOI size after binning.
        return (h_woi // self._binning[1], w_woi // self._binning[0])

    def _get_out(self, shape, out=None, dtype=None):
        """
        Allocate a buffer of ``shape`` and ``dtype``, or check that ``out`` is one.
        Defaults to the raw sensor :attr:`dtype`; software binning widens it.
        """
        shape = tuple(int(s) for s in shape)
        dtype = np.dtype(self.dtype if dtype is None else dtype)
        if out is None:
            return np.empty(shape, dtype=dtype)
        if out.shape != shape:
            raise ValueError(f"Expected out to be of shape {shape}. Found {out.shape}.")
        if out.dtype != dtype:
            raise ValueError(f"Expected out to be of type {dtype}. Found {out.dtype}.")

        return out

    def _crop_to_woi(self, img):
        """
        Software-apply WOI crop and/or binning to ``(H, W)`` image or ``(N, H, W)`` stack.
        """
        # Step 1: Software WOI crop.
        if self._software_woi:
            x0, w, y0, h = self._woi
            if x0 != 0 or y0 != 0 or w != self._shape[1] or h != self._shape[0]:
                img = img[..., y0:y0+h, x0:x0+w]

        # Step 2: Software binning (block-sum of adjacent pixels).
        if self._software_binning:
            binx, biny = self._binning
            if biny != 1 or binx != 1:
                # Promote so the block-sum cannot overflow the raw dtype, never narrowing.
                img = img.astype(np.promote_types(img.dtype, self.get_dtype(averaging=1, hdr=False)))
                # Sum into the promoted type; sum() otherwise re-promotes to uint64.
                if img.ndim == 2:
                    H, W = img.shape
                    Ht, Wt = (H // biny) * biny, (W // binx) * binx
                    img = img[:Ht, :Wt].reshape(
                        Ht // biny, biny, Wt // binx, binx
                    ).sum(axis=(1, 3), dtype=img.dtype)
                else:   # (N, H, W) stack
                    N, H, W = img.shape
                    Ht, Wt = (H // biny) * biny, (W // binx) * binx
                    img = img[:, :Ht, :Wt].reshape(
                        N, Ht // biny, biny, Wt // binx, binx
                    ).sum(axis=(2, 4), dtype=img.dtype)

        return img

    # Core capture methods to be implemented by subclass.

    @abstractmethod
    def _get_image_hw(self, timeout_s):
        """
        Abstract method to capture camera images.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait for the frame to be fetched.

        Returns
        -------
        numpy.ndarray
            The raw, untransformed frame of shape :attr:`_hw_image_shape`.
            WOI cropping, software binning, and :attr:`transform` are applied later by
            :meth:`.get_image()`.
        """
        raise NotImplementedError(f"Camera {self.name} has not implemented _get_image_hw")

    def _get_images_hw(self, image_count, timeout_s, out=None):
        """
        Abstract method to capture a series of image_count images using camera-specific
        batch acquisition features.

        Parameters
        ----------
        image_count : int
            Number of frames to batch collect.
        timeout_s : float
            The time in seconds to wait for **each** frame to be fetched.
        out : None OR numpy.ndarray
            Preallocated memory of shape ``(image_count, *_hw_image_shape)``
            to fill in place, if applicable.

        Returns
        -------
        numpy.ndarray
            The raw, untransformed stack of shape ``(image_count, *_hw_image_shape)``.
            WOI cropping, software binning, and :attr:`transform` are applied later by
            :meth:`.get_images()`.
        """
        # Preallocate memory if necessary
        out = self._get_out((image_count,) + tuple(self._hw_image_shape), out)

        for i in range(image_count):
            out[i, :, :] = self._get_image_hw_tolerant(timeout_s)

        return out

    # Capture methods one level of abstraction above _get_image_hw().

    def _parse_color_image(self, img):
        """
        Reduces a color image to grayscale according to :attr:`color_channel`.

        The color channel is assumed to be the **last** axis, so this works both for a
        single frame of shape ``(height, width, channels)`` and for a stack of frames of
        shape ``(image_count, height, width, channels)``.

        Parameters
        ----------
        img : numpy.ndarray
            Color image (or stack of color images) with the channel as the last axis.

        Returns
        -------
        numpy.ndarray
            The image(s) with the color axis removed.
        """
        color_channel = self.color_channel
        if color_channel is None:
            color_channel = 0

        if isinstance(color_channel, int):
            return img[..., color_channel]
        elif isinstance(color_channel, (list, np.ndarray)):
            raise NotImplementedError("Weighted color handling is not implemented yet.")
        else:
            raise ValueError(f"Expected color_channel to be None or int. Found {self.color_channel}.")

    def _get_image_hw_tolerant(self, *args, **kwargs):
        """
        Wraps :meth:`._get_image_hw()` with error tolerance and color handling.

        Retries the capture up to :attr:`capture_attempts` times, warning on intermediate
        failures and re-raising the last error if every attempt fails. Color frames
        (shape ``(height, width, channels)``) are reduced to grayscale via
        :meth:`._parse_color_image()`.

        Parameters
        ----------
        *args, **kwargs
            Passed through to :meth:`._get_image_hw()`.

        Returns
        -------
        numpy.ndarray
            A 2D image of shape ``(height, width)``.
        """
        err = None
        failures = 0
        get = kwargs.pop("get", True)

        for _ in range(self.capture_attempts):
            try:
                raw = self._get_image_hw(*args, **kwargs)
                if not get:
                    img = raw
                elif is_gpu_array(raw):
                    img = raw.get()
                else:
                    # Copy, rather than view: a driver may reuse its frame buffer.
                    img = np.array(raw)

                if len(img.shape) == 2:     # All good!
                    pass
                elif len(img.shape) == 3:     # Need to convert to grayscale.
                    img = self._parse_color_image(img)
                else:
                    raise ValueError(f"Expected a 2D or 3D (color) image. Found {img.shape}.")

                if failures > 0:
                    self.logger.warning("_get_image_hw() failed %s times before succeeding.", failures)

                return img
            except Exception as e:
                failures += 1
                err = e

        self.logger.error("_get_image_hw() failed %s times before quitting.", failures)

        raise err

    def _get_images_hw_tolerant(self, *args, **kwargs):
        """
        Wraps :meth:`._get_images_hw()` with error tolerance and color handling.

        Retries the capture up to :attr:`capture_attempts` times, warning on intermediate
        failures and re-raising the last error if every attempt fails. Color stacks
        (shape ``(image_count, height, width, channels)``) are reduced to grayscale via
        :meth:`._parse_color_image()`.

        Parameters
        ----------
        *args, **kwargs
            Passed through to :meth:`._get_images_hw()`.

        Returns
        -------
        numpy.ndarray
            A stack of 2D images of shape ``(image_count, height, width)``.
        """
        err = None
        failures = 0
        get = kwargs.pop("get", True)

        for _ in range(self.capture_attempts):
            try:
                raw = self._get_images_hw(*args, **kwargs)
                imgs = as_numpy(raw) if get else raw

                if imgs.ndim == 4:      # Stack of color images; reduce to grayscale.
                    imgs = self._parse_color_image(imgs)

                if failures > 0:
                    self.logger.warning("_get_images_hw() failed %s times before succeeding.", failures)

                return imgs
            except Exception as e:
                failures += 1
                err = e

        self.logger.error("_get_images_hw() failed %s times before quitting.", failures)

        raise err

    # High-level capture methods.

    def get_image(self, timeout_s=1, transform=True, hdr=None, averaging=None, get=True):
        """
        Capture, process, and return images from a camera.

        Tip
        ~~~
        This function includes two advanced capture options:

        -   `Multi-exposure High Dynamic Range (HDR) imaging
            <https://en.wikipedia.org/wiki/Multi-exposure_HDR_capture>`_
            and
        -   Software frame averaging (integrating).

        These methods can aid the user in capturing more precise data, beyond the
        default raw (and bitdepth-limited) output of the camera.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait for the frame to be fetched.
            The frame exposure time is **added** to this timeout
            such that there is always enough time to expose.
        transform : bool
            Whether or not to transform the output image according to
            :attr:`~slmsuite.hardware.cameras.camera.Camera.transform`.
            Defaults to ``True``.
        hdr : int OR (int, int) OR None OR False
            Exposure information for `Multi-exposure High Dynamic Range (HDR) imaging
            <https://en.wikipedia.org/wiki/Multi-exposure_HDR_capture>`_
            If ``None``, the value of :attr:`hdr` is used.
            If ``False``, HDR is not used no matter the state of :attr:`hdr`.

            See Also
            ~~~~~~~~
            :meth:`.get_image_hdr()` for more information.

        averaging : int OR None OR False
            If ``int``, the number of frames to average over.
            If ``None``, the value of :attr:`averaging` is used.
            If ``False``, averaging is not used no matter the state of :attr:`averaging`.

            Tip
            ~~~
            The datatype is promoted to float if necessary but otherwise tries to stick
            with the default datatype.
            For instance, a camera that returns a 12-bit image as a 16-bit type has four
            more bits to use for averaging, i.e. :math:`2^4 = 16` possible averages without
            risk of overflow.
            Requesting more than 16 averages would cause the return type to be promoted
            to ``float``.

            Important
            ~~~~~~~~~
            This feature sums many measurements together (does not mean),
            thereby averaging without floating point operations.
            This is done such that integer datatypes (useful for memory compactness) can still be returned,
            whereas a general mean would need to be floating point.

        get : bool
            If ``True`` (default), ensures returned array is a host :class:`numpy.ndarray`.
            If ``False``, returns the driver or simulation array in its native device format
            (e.g., :class:`cupy.ndarray` for simulated GPU cameras). This lets a feedback
            loop against simulated hardware stay on the GPU. Ignored when ``hdr`` is
            requested, as that analysis is host-only.

            Note
            ~~~~
            The backend is chosen per call here, unlike an
            :class:`~slmsuite.hardware.slms.slm.SLM`, which fixes one for the whole object
            (see :attr:`~slmsuite.hardware.slms.slm.SLM.xp`). An SLM owns persistent arrays
            that many callers read, so the backend has to be a property of the object; a
            frame is a transient value with one consumer, and that consumer is the one who
            knows whether it wants host or device memory. A physical camera also has no
            device-native source --- its driver delivers host memory regardless.

        Returns
        -------
        numpy.ndarray OR cupy.ndarray of int OR float
            Array of shape :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`.
        """
        # Parse acquisition options.
        averaging = self._parse_averaging(averaging)
        (exposures, exposure_power) = self._parse_hdr(hdr)

        # Switch based on what imaging case we're in.
        if exposures > 1:       # Average many images with increasing exposure.
            return self.get_image_hdr(
                (exposures, exposure_power),
                timeout_s=timeout_s,
                transform=transform,
                averaging=averaging,
            )
        elif averaging > 1:     # Average many images.
            averaging_dtype = self.get_dtype(averaging=averaging)

            try:
                if not get:
                    # The batch method fills a host buffer, so it cannot serve a
                    # device-native request; accumulate frame by frame instead.
                    raise NotImplementedError

                # Using the camera-specific batch method if available
                imgs = self._get_images_hw_tolerant(
                    averaging, timeout_s=timeout_s + self.exposure_s
                )

                # Sum into the declared type; np.sum otherwise promotes to uint64.
                img = np.sum(imgs, axis=0, dtype=averaging_dtype)
            except NotImplementedError:
                # Brute-force collection as a backup
                img = None

                for _ in range(averaging):
                    # astype() copies, so the first frame is safe to accumulate into.
                    frame = self._get_image_hw_tolerant(
                        timeout_s=timeout_s + self.exposure_s, get=get
                    ).astype(averaging_dtype)
                    if img is None:
                        img = frame
                    else:
                        img += frame
        else:                   # Normal image
            img = self._get_image_hw_tolerant(
                timeout_s=timeout_s + self.exposure_s, get=get
            )

        # Software WOI crop and/or binning (no-op when handled by hardware).
        img = self._crop_to_woi(img)

        # self.transform implements the flipping and rotating keywords passed to the
        # superclass constructor.
        if transform:
            img = self.transform(img)

        # Store the result locally.
        self.last_image = img

        # Push to viewer if active.
        if self.viewer is not None:
            self.viewer.render(img)

        return img

    def get_images(self, image_count, timeout_s=1, out=None, transform=True, flush=False):
        """
        Grab ``image_count`` images in succession.

        Important
        ~~~~~~~~~
        This method **does not** support averaging or HDR features.
        Rather, it just returns a series of raw images.

        Parameters
        ----------
        image_count : int
            Number of images to grab.
        timeout_s : float
            The time in seconds to wait **for each** frame to be fetched.
            The frame exposure time is **added** to this timeout
            such that there is always enough time to expose.
        out : None OR numpy.ndarray
            If not ``None``, output data in this memory. Must match the returned array:
            shape ``(image_count, *shape)`` sized from :attr:`shape` after
            :attr:`transform` (or before it, if ``transform=False``), and dtype
            ``get_dtype(averaging=1, hdr=False)``, as this method applies neither.
            The capture lands in this buffer directly---avoiding
            any allocation---unless software windowing, binning, or color reduction has
            to reshape the frames on the way out.
        transform : bool
            Whether or not to transform the output image according to
            :attr:`~slmsuite.hardware.cameras.camera.Camera.transform`.
            Defaults to ``True``.
        flush : bool
            Whether to flush before grabbing.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(image_count, height, width)``.
        """
        # Flush if desired.
        if flush:
            self.flush()

        # `shape` is the stack this returns; `raw_shape` is the same stack before transform.
        raw_shape = tuple(self.transform.inverse.transform_shape(self.shape))
        shape = tuple(self.shape) if transform else raw_shape
        if out is not None:
            out = self._get_out(
                (image_count,) + shape, out, dtype=self.get_dtype(averaging=1, hdr=False)
            )

        # Hand the hardware a view of the caller's buffer, unless the frames change shape.
        if out is None or raw_shape != tuple(self._hw_image_shape):
            out_hw = None
        else:
            out_hw = self.transform.inverse(out) if transform else out

        # Grab images (no transformation)
        imgs = self._get_images_hw_tolerant(
            image_count,
            timeout_s=timeout_s + self.exposure_s,
            out=out_hw,
        )
        filled = out_hw is not None and imgs is out_hw

        # Software WOI crop and/or binning (unused when handled by hardware).
        imgs = self._crop_to_woi(imgs)

        # Transform if desired. Future: make more efficient.
        if transform:
            imgs = self.transform(imgs)

        # Fill the caller's buffer, unless the hardware already wrote into it.
        if out is not None:
            if not filled:
                np.copyto(out, imgs)
            imgs = out

        # Store the result locally.
        self.last_image = imgs[-1]

        # Push to viewer if active.
        if self.viewer is not None:
            self.viewer.render(imgs[-1])

        return imgs

    def flush(self, timeout_s=1):
        """
        Cycle the image buffer such that all new :meth:`.get_image()` calls yield fresh frames.
        Without this feature, optimizations could be working on outdated information.

        Defaults to calling :meth:`.get_image()` twice, though cameras can implement
        hardware-specific alternatives.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait **for each** frame.
            The frame exposure time is **added** to this timeout
            such that there is always enough time to expose.
        """
        for _ in range(self._flush_iterations):
            self._get_image_hw_tolerant(timeout_s=timeout_s + self.exposure_s)

    # HDR imaging methods.

    def get_image_hdr(self, exposures=None, return_raw=False, **kwargs):
        r"""
        Often, the necessities of precision applications exceed the bitdepth of a
        camera. One way to recover High Dynamic Range (HDR) imaging is to use
        `multiple exposures <https://en.wikipedia.org/wiki/Multi-exposure_HDR_capture>`_
        each with increasing exposure time. Then, these images can be stitched together
        as **floating-point** data, omitting data which is under- or over- exposed.

        Tip
        ~~~
        This feature can be accessed in :meth:`.get_image()`
        using :attr:`hdr` or the ``hdr=`` flag.
        This function is exposed here also to reveal the raw data using ``return_raw=``
        and to draw attention to this useful feature.

        Caution
        ~~~~~~~
        Camera exposure is sometimes poorly defined. This might cause incorrect
        assumptions of the exposure.
        In general, a larger base exposure will produce more accurate results as a
        greater number of sample clock periods are rounded to for smaller relative variation.
        Future modifications to :meth:`get_image_hdr_analysis()` might improve image stitching.

        Parameters
        ----------
        exposures : int OR (int, int) OR None
            The number of exposures to take.
            Each exposure increases in time multiplicatively from the base value
            (original :meth:`.get_exposure()`) by a factor :math:`p`.
            The :math:`i\text{th}` image has exposure time :math:`\tau \times p^i`, zero-indexed.
            The default base of :math:`p = 2` leads to ``exposures`` being equivalent to
            `stops <https://en.wikipedia.org/wiki/Exposure_value>`_.
            This base can be changed to another number by instead passing a tuple, where
            the second ``int`` defines the desired base.
        return_raw : bool
            If ``True``, returns the raw data (stack of images with count ``exposures``)
            instead of the processed data. The data can be processed using :meth:`.get_image_hdr_analysis`.
        **kwargs
            Passed to :meth:`.get_image()`.

        Returns
        -------
        numpy.ndarray of float
            Array of shape :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`.

            Important
            ~~~~~~~~~
            The scale of the returned image is the same as the original exposure.
        """
        (exposures, exposure_power) = self._parse_hdr(exposures)
        # In the units of the frames this call will actually capture.
        overexposure_threshold = self._bitresolution(
            self._parse_averaging(kwargs.get("averaging"))
        ) / 2

        # Make empty data and grab the original exposure time.
        original_exposure = self.get_exposure()
        # Store frames as float: the analysis immediately casts to float anyway,
        # and this sidesteps any integer overflow from software binning or wide averaging.
        imgs = None
        exposure_times = np.zeros((exposures,), dtype=float)

        try:
            for i in range(exposures):
                # FUTURE: record the get_exposures and use these to do better analysis.
                exposure_times[i] = self.set_exposure(int(exposure_power ** i) * original_exposure)
                self.flush()    # Sometimes, cameras return bad frames after exposure change.
                frame = self.get_image(hdr=False, **kwargs)
                if imgs is None:
                    imgs = np.zeros((exposures,) + tuple(frame.shape), float)
                imgs[i, :, :] = frame

                # Terminate the loop if our image is entirely overexposed.
                if np.all(imgs[i, :, :] > overexposure_threshold):
                    # Drop the unexposed tail so the stack only holds measured frames.
                    imgs = imgs[:i+1, :, :]
                    exposure_times = exposure_times[:i+1]
                    break
        finally:
            # Else a failed capture strands the camera at the elevated exposure.
            self.set_exposure(original_exposure)

        if return_raw:
            return imgs, exposure_times
        else:
            img = self.get_image_hdr_analysis(
                imgs,
                overexposure_threshold=overexposure_threshold,
                exposure_power=exposure_times,
            )
            if np.nanmax(img) >= self.bitresolution:
                self.logger.warning("HDR image is overexposed.")
            # Store the result locally.
            self.last_image = img
            return img

    @staticmethod
    def get_image_hdr_analysis(imgs, overexposure_threshold=None, exposure_power=2):
        r"""
        Analyzes raw data for High Dynamic Range (HDR) imaging
        `multiple exposures <https://en.wikipedia.org/wiki/Multi-exposure_HDR_capture>`_
        each with increasing exposure time.

        Parameters
        ----------
        imgs : array_like
            Stack of images with increasing exposure.
        overexposure_threshold : float OR None
            For each image (except the first), data is thrown out if values are above
            this threshold. If ``None``, the threshold defaults to half the maximum.
        exposure_power : int or list of float
            Each exposure increases in time multiplicatively from the base value
            (original :meth:`.get_exposure()`) by this factor :math:`p`. The :math:`i\text{th}` image has
            exposure time :math:`\tau \times p^i`, zero-indexed.
            The default value of ``2`` leads to ``exposures`` being equivalent to
            `stops <https://en.wikipedia.org/wiki/Exposure_value>`_.

        Returns
        -------
        numpy.ndarray of float
            Array of shape :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`.

            Important
            ~~~~~~~~~
            The scale of the returned image is the same as the original exposure.
        """
        # Parse arguments
        if np.isscalar(exposure_power):
            exposure_power = float(int(exposure_power))
            exposure_times = np.power(exposure_power, np.arange(imgs.shape[0]))
        else:
            exposure_times = np.array(exposure_power)
            if np.all(exposure_times <= 0):
                raise ValueError("exposure_times cannot all be non-positive.")
            exposure_times = exposure_times / np.min(exposure_times[exposure_times > 0])

        if overexposure_threshold is None:
            # Default to half exposure.
            overexposure_threshold = np.max(imgs) / 2

        img = None

        for i in range(imgs.shape[0]):
            img_current = imgs[i, :, :].astype(float)

            if i == 0:
                img = img_current
            elif exposure_times[i] > 0:
                # Overwrite data when greater precision is available.
                mask = img_current < overexposure_threshold
                img[mask] = img_current[mask] / exposure_times[i]

        return img

    # Self-test method to test everything above.

    def test(self):
        """
        Test that the hardware behind this :class:`Camera` responds to every core method.

        Each step is named, so a failure reports which feature is broken. Correctness of
        the returned data is the business of the test suite, not of a hardware check.
        """
        self.logger.info("Testing %s.", self.name)

        (orig_averaging, orig_hdr) = (self.averaging, self.hdr)
        (self.averaging, self.hdr, self.last_image) = (None, None, None)
        exposure = None

        try:
            with self._test_step("read and set the exposure"):
                exposure = self.get_exposure()
                assert exposure > 0, f"exposure is {exposure}"
                self.set_exposure(exposure * 1.5)

            with self._test_step("capture a frame"):
                self.get_image(timeout_s=2)
                assert self.last_image is not None, "the capture was not retained"

            with self._test_step("capture without the transform applied"):
                self.get_image(transform=False, timeout_s=2)

            with self._test_step("capture an averaged frame"):
                self.get_image(averaging=2, timeout_s=5)

            with self._test_step("stack frames, into a given buffer"):
                out = np.empty((3,) + self.shape, dtype=self.get_dtype())
                self.get_images(3, timeout_s=2)
                self.get_images(3, timeout_s=2, out=out)

            with self._test_step("flush pending frames"):
                self.flush(timeout_s=2)

            try:
                with self._test_step("stitch exposures with get_image_hdr"):
                    self.get_image_hdr(exposures=2, timeout_s=3)
                    self.get_image_hdr(exposures=2, return_raw=True, timeout_s=3)
            except AssertionError as error:
                self.logger.warning("%s; treating HDR as unsupported.", error)

            if not self._software_woi:
                with self._test_step("window the sensor with set_woi"):
                    orig_woi = self.get_woi()
                    self.set_woi()
                    self.set_woi((0, self.shape[1] // 2, 0, self.shape[0] // 2))
                    assert self.get_image().shape == self.shape, "set_woi did not resize"
                    self.set_woi(orig_woi)
                    assert self.get_woi() == orig_woi, "set_woi did not round-trip"

            if not self._software_binning:
                with self._test_step("bin the sensor with set_binning"):
                    orig_binning = self.get_binning()
                    self.set_binning(1)
                    assert self.get_binning() == (1, 1), "set_binning did not round-trip"
                    assert self.get_image().shape == self.shape, "set_binning did not resize"
                    self.set_binning(orig_binning)

            with self._test_step("enumerate the connected cameras with info"):
                self.info(verbose=False)

            count = 10
            t0 = time.perf_counter()
            for _ in range(count):
                self.get_image(timeout_s=2)
            elapsed = time.perf_counter() - t0
            self.logger.info(
                "get_image: %.1f Hz (%.2f ms/frame)", count / elapsed, 1e3 * elapsed / count
            )
        finally:
            if exposure is not None:
                self.set_exposure(exposure)
            (self.averaging, self.hdr) = (orig_averaging, orig_hdr)

        return True

    # Display method.

    def plot(self, image=None, limits=None, title="Image", ax=None, cbar=True):
        """
        Plots the provided image.

        Parameters
        ----------
        image : ndarray OR None OR bool
            Image to be plotted.
            If ``None``, grabs an image from the camera.
            If ``False``, uses the :attr:`.last_image` attribute.
        limits : None OR float OR [[float, float], [float, float]]
            Scales the limits by a given factor or uses the passed limits directly.
        title : str
            Title the axis.
        ax : matplotlib.pyplot.axis OR None
            Axis to plot upon.
        cbar : bool
            Also plot a colorbar. Does not work if ``ax`` is passed.

        Returns
        -------
        matplotlib.pyplot.axis
            Axis of the plotted image.
        """
        if image is None:
            self.flush()
            image = self.get_image()
        if image is False:
            image = self.last_image
        # as_numpy: last_image follows get_image(get=...), so it may be device-resident.
        image = as_numpy(image)

        (ax, _, should_show) = self._plot(
            image, limits, title, ax=ax, cbar=cbar, labels=BLAZE_LABELS["ij"]
        )

        if should_show:
            _slmsuite_plt_show(name="camera_plot")
        return ax

    # Automated refinement methods.

    @staticmethod
    def _autoexpose_metric(img):
        return np.max(img)

    def autoexposure(self, *args, **kwargs):
        """Backwards-compatible alias for :meth:`autoexpose()`."""
        return self.autoexpose(*args, **kwargs)

    def autoexpose(
        self,
        set_fraction=0.5,
        tol=0.05,
        exposure_bounds_s=None,
        window=None,
        metric=None,
        timeout_s=10,
        verbose=True,
    ):
        """
        Sets the exposure of the camera such that the maximum value is at ``set_fraction``
        of the dynamic range. Useful for mitigating over- or under- exposure.

        Parameters
        ----------
        set_fraction : float
            Fraction of camera dynamic range to use as a target image maximum.
        tol : float
            Fractional tolerance for exposure adjustment.
        exposure_bounds_s : (float, float) OR None
            Shortest and longest allowable integration in seconds. If ``None``, defaults to
            :attr:`exposure_bounds_s`. If this attribute was not set (or not available on
            a particular camera), then ``None`` instead defaults to unbounded.
        window : array_like OR None
            Passed to :meth:`~slmsuite.holography.toolbox.window_slice()`.
            If ``None``, the full camera frame will be used.
        metric : lambda OR None
            Metric to use for exposure within the chosen window.
            If ``None``, this defaults to ``np.max``, which tries to pin the image
            maximum to the desired exposure ``set_fraction``.
        timeout_s : float
            Stop attempting adjusting exposure after ``timeout_s`` seconds.
        verbose : bool
            If ``True``, progress is logged at ``INFO``; otherwise at ``DEBUG``.
            Visibility is ultimately governed by :func:`slmsuite.configure_logging`.

        Returns
        -------
        float
            Resulting exposure in seconds.
        """
        # Parse set_fraction
        if set_fraction is True:
            set_fraction = 0.5
        set_fraction = float(set_fraction)
        if set_fraction <= 0:
            raise ValueError("set_fraction must be positive.")

        # Parse tol
        tol = float(tol)
        if tol <= 0:
            raise ValueError("tol must be positive.")

        # Parse exposure_bounds_s
        if exposure_bounds_s is None:
            if self.exposure_bounds_s is None:
                exposure_bounds_s = (0, np.inf)
            else:
                exposure_bounds_s = self.exposure_bounds_s

        # Parse window
        sliced = window_slice(window)

        # Parse metric
        if metric is None:
            metric = Camera._autoexpose_metric

        # Initialize loop
        set_val = 0.5 * self.bitresolution
        exp = self.get_exposure()
        self.flush()
        img = self.get_image(hdr=False)
        is_railed = False

        # Calculate the error as a percent of the camera's bitresolution
        status = metric(img[sliced])
        err = np.abs(status - set_val) / self.bitresolution
        t = time.perf_counter()

        # Loop until we meet tolerance, run out of steps, or time out.
        for _ in range(20):
            if err <= tol or time.perf_counter() - t > timeout_s:
                break

            exp_prev = exp

            # Clip exposure steps to 0.1x -> 10x, also avoiding division by 0.
            exp_unclipped = exp * np.clip(set_val / max(status, 1), .1, 10)
            exp = np.clip(exp_unclipped, exposure_bounds_s[0], exposure_bounds_s[1])
            if exp_unclipped != exp:
                # If already railed, handle failure cases (TODO).
                if is_railed:
                    if (exp == exposure_bounds_s[0] and set_fraction < 0.5):
                        break
                    if (exp == exposure_bounds_s[1] and set_fraction > 0.5):
                        break

                # Otherwise, prepare to do so next loop.
                is_railed = True

            self.set_exposure(exp)
            exp = self.get_exposure()
            if exp_prev == exp:
                # If already railed, handle failure cases (TODO).
                if is_railed:
                    break

                # Otherwise, prepare to do so next loop.
                is_railed = True

            self.flush()
            img = self.get_image(hdr=False)

            status = metric(img[sliced])
            err = np.abs(status - set_val) / self.bitresolution

            self.logger.log(
                logging.INFO if verbose else logging.DEBUG,
                "Autoexpose: %.2e s - %s/%s",
                exp, status, self.bitresolution - (self.averaging if self.averaging is not None else 1),
            )

        # The loop targets 50% of resolution.
        # Now set the final exposure if different (TODO, improve).
        if set_fraction != 0.5:
            exp = self.set_exposure(exp * (2 * set_fraction))

        return exp

    @staticmethod
    def _autofocus_metric(img, plot=False):
        """See :meth:`.autofocus()` ``metric=``"""
        dft = np.fft.fftshift(np.fft.fft2(img.astype(float)))
        dft_amp = np.abs(dft)
        dft_norm = dft_amp / np.amax(dft_amp)
        fom = np.sum(dft_norm)

        if plot >= 1:
            _, axs = plt.subplots(1, 2)

            axs[0].imshow(img)
            axs[0].set_title("Image")
            axs[0].set_xticks([])
            axs[0].set_yticks([])

            axs[1].imshow(dft_norm)
            axs[1].set_title(f"FFT\nFoM$ = \\int\\int $|FFT|$ / $max|FFT|$ = {fom}$")
            axs[1].set_xticks([])
            axs[1].set_yticks([])

            _slmsuite_plt_show(name="autofocus_metric")

        return fom

    def autofocus(self, set_z, get_z=0, range_z=2, metric=None, plot=False, verbose=False):
        """
        Finds optimal focus when scanning over some variable ``z``.
        This ``z`` often takes the form of a vertical stage to position a sample precisely
        at the plane of imaging of a lens or objective.
        The default ``metric`` is based on the Fourier contrast of the image,
        and works particularly well when combined with a projected spot array hologram.

        Parameters
        ----------
        set_z : function OR SLM
            Sets the position of the focusing stage to a given ``float``.
            If an SLM is passed, adds a lens phase to the SLM to focus the existing
            pattern. In this case, the units of ``z`` are in Zernike defocus terms
            (wavefronts). The optimal defocus is added to the wavefront calibration
            (``source["phase"]``) of the SLM.
        get_z : function OR float
            Gets the current position of the focusing stage. Should return a ``float``.
            Can also pass a ``float`` representing the center of the search range.
        range_z : array_like OR float OR None
            ``z`` values to sweep over during search relative to the base position ``get_z``.
            If a single ``float`` is passed, sweeps from ``-range_z`` to ``+range_z``
            with 11 steps.
        metric : function OR None
            Function which evaluates the focus quality of an image.
            Should take in an image and return a scalar figure-of-merit (FoM).
            Defaults to :meth:`Camera._autofocus_metric`, which approximates the
            sharpness of the images (implemented as the Fourier contrast of the image,
            the sum of the normalized Fourier amplitudes). The sharper the image, the
            higher the FoM.
        plot : int OR bool
            Whether to provide illustrative plots, at ``1`` and above.
        verbose : bool
            If ``True``, progress is logged at ``INFO``; otherwise at ``DEBUG`` (default).
            Visibility is ultimately governed by :func:`slmsuite.configure_logging`.

        Returns
        -------
        float
            Optimal ``z`` value found.
        """
        # Parse set_z
        if hasattr(set_z, 'set_phase'):
            # SLM passed; create lens phase setter.
            slm = set_z
            base_phase = slm.phase.copy()
            base_correction = slm.source.get('phase', None)
            if base_correction is None:
                base_correction = slm.xp.zeros_like(base_phase)
            base_phase -= base_correction

            def slm_set_z(z_val):
                slm.source['phase'] = (
                    base_correction +
                    as_backend(zernike(slm, index=4, weight=z_val, use_mask=False), slm.xp)
                )
                slm.set_phase(
                    base_phase,
                    settle=True
                )

            set_z = slm_set_z

        if not callable(set_z):
            raise ValueError("set_z must be a function or SLM.")

        # Parse get_z
        z_base = get_z
        if callable(get_z):
            z_base = get_z()

        # Parse range_z. Build a fresh float array so we never mutate a caller's input
        # and so counts can hold NaN sentinels (an integer z_list would not).
        if np.isscalar(range_z):
            z_list = np.linspace(-range_z, range_z, 11, endpoint=True)
        else:
            z_list = np.asarray(range_z, dtype=float)
        z_list = np.sort(z_list + z_base)

        # Parse metric
        if metric is None:
            metric = Camera._autofocus_metric

        # Setup for the sweep
        imlist = []
        counts = np.full(z_list.shape, np.nan)

        self.flush()

        for i, z in enumerate(z_list):
            try:
                self.logger.debug("Moving to z = %.2f...", z)
                set_z(z)

                # Take image and evaluate metric.
                img = self.get_image()
                imlist.append(np.copy(img))
                counts[i] = metric(img)
            except Exception as e:
                self.logger.debug("Autofocus capture at z = %.2f failed: %s", z, e)

        # Handle the case where everything failed.
        if np.all(np.isnan(counts)):
            try:
                set_z(z_base)
            except Exception:
                pass
            raise RuntimeError("Autofocus failed; no valid images captured.")

        # Otherwise, fit a Lorentzian to the data to find the optimum.
        I_max_count = np.nanargmax(counts)

        dz = np.mean(np.diff(z_list))
        popt0 = np.array(
            [z_list[I_max_count], np.nanmax(counts) - np.nanmin(counts), np.nanmin(counts), (z_list[-1]-z_list[0])]
        )
        bounds = np.array(
            [
                [z_list[0], 0, 0, dz],
                [z_list[-1], (np.nanmax(counts) - np.nanmin(counts))*2, np.nanmax(counts), np.inf]
            ]
        )

        try:
            popt, _ = curve_fit(
                lorentzian,
                z_list,
                counts,
                bounds=bounds,
                ftol=1e-5,
                p0=popt0,
            )
            z_opt = popt[0]
            c_opt = popt[1] + popt[2]
        except BaseException:
            self.logger.warning("Autofocus fit failed, using maximum fom as optimum.")
            z_opt = z_list[I_max_count]
            c_opt = counts[I_max_count]

        # Goto the optimal position
        self.logger.log(logging.INFO if verbose else logging.DEBUG, "Moving to optimized value, z = %s", z_opt)
        set_z(z_opt)

        # Show result if desired
        if plot >= 1:
            plt.scatter(z_list, counts, color="k", label="Data")
            plt.xlabel(r"$z$")
            plt.ylabel("Figure of Merit")
            plt.title("Autofocus Sweep")
            plt.scatter(z_opt, c_opt, color="r", label="Result")

            z_list_fine = np.linspace(np.min(z_list), np.max(z_list), 1000)

            lfit = None
            try:
                lfit = lorentzian(z_list_fine, *popt)
            except BaseException:
                lfit = None
            if lfit is not None:
                plt.plot(z_list_fine, lfit, color="r", label="Fit")
            plt.legend()
            _slmsuite_plt_show(name="autofocus")

        return z_opt

