"""
Hardware control for AlliedVision cameras via the Vimba-X :mod:`vmbpy` interface.
This class also supports backwards compatibility with the
`archived <https://github.com/alliedvision/VimbaPython>`_ :mod:`vimba` interface.
Install :mod:`vmbpy` by following the
`provided instructions <https://github.com/alliedvision/VmbPy>`_.
Be sure to include the ``numpy`` flag in the ``pip install`` command,
as the :class:`AlliedVision` class makes use of these features. See especially the
`vimba python manual <https://github.com/alliedvision/VimbaPython/blob/master/Documentation/Vimba%20Python%20Manual.pdf>`_
for reference.

Note
~~~~
Color camera functionality is not currently implemented, and will lead to undefined behavior.
"""

import time
import numpy as np
import warnings

from slmsuite.hardware.cameras.camera import Camera
from slmsuite._logging import make_logger

logger = make_logger(__name__)

try:
    import vmbpy
    vimba_system = vmbpy.VmbSystem
    vimba_name = "vmbpy"

except ImportError:
    try:
        import vimba
        vimba_system = vimba.Vimba
        vimba_name = "vimba"

        logger.warning("vmbpy not installed; falling back to vimba")
    except ImportError:
        vimba_system = None
        vimba_name = ""
        logger.warning("vimba or vmbpy are not installed. Install to use AlliedVision cameras.")


class AlliedVision(Camera):
    r"""
    AlliedVision camera.

    Attributes
    ----------
    sdk : vmbpy.Vimba
        AlliedVision SDK. Shared among instances of :class:`AlliedVision`.
    cam : vmbpy.Camera
        Object to talk with the desired camera.

    Caution
    ~~~~~~~
    The AlliedVision SDK :mod:`vmbpy` includes protections to maintain camera connectivity:
    specifically, the SDK :class:`vmbpy.VmbSystem` and cameras :class:`vmbpy.Camera` are designed
    to be used in concert with ``with`` statements. Unfortunately, this does not mesh with the
    architecture of :mod:`slmsuite`, where notebook-style operation is desired.
    Using ``with`` statements inside :class:`.AlliedVision` methods is likewise not an option,
    as the methods to :meth:`__enter__()` and :meth:`__exit__()` the ``with`` are time-consuming
    due to calls to :meth:`_open()` and :meth:`_close()` the objects, to the point of
    :math:`\mathcal{O}(\text{s})` overhead. :class:`.AlliedVision` disables these protections by
    calling :meth:`__enter__()` and :meth:`__exit__()` directly during :meth:`.__init__()` and
    :meth:`.close()`, instead of inside ``with`` statements.
    """

    sdk = None

    def __init__(self, serial="", pitch_um=None, **kwargs):
        """
        Initialize camera and attributes.

        Caution
        ~~~~~~~
        AlliedVision cameras sometimes have different attribute names depending on the
        model. This constructor tries to set some default properties, but if they fail, it will log a
        warning and continue. The user will need to configure the :mod:`vmbpy` handle
        directly in this case, or in the case where the default configuration is not desired.

        Parameters
        ----------
        serial : str
            Serial number of the camera to open.
            Use :meth:`.info()` to see detected options.
            If empty, defaults to the first camera in the list
            returned by :meth:`vmbpy.get_all_cameras()`.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        if vimba_system is None:
            raise ImportError("vimba or vmbpy are not installed. Install to use AlliedVision cameras.")

        if AlliedVision.sdk is None:
            logger.debug("%s initializing...", vimba_name)
            AlliedVision.sdk = vimba_system.get_instance()
            AlliedVision.sdk.__enter__()

        logger.debug("Looking for cameras...")
        camera_list = AlliedVision.sdk.get_all_cameras()

        serial_list = [cam.get_serial() for cam in camera_list]
        if serial == "":
            if len(camera_list) == 0:
                raise RuntimeError(f"No cameras found by {vimba_name}.")
            if len(camera_list) > 1:
                logger.debug("No serial given; choosing first of %s", serial_list)

            self.cam = camera_list[0]
            serial = self.cam.get_serial()
        else:
            if serial in serial_list:
                self.cam = camera_list[serial_list.index(serial)]
            else:
                raise RuntimeError(
                    f"Serial {serial} not found by {vimba_name}. Available: {serial_list}"
                )

        logger.debug("%s sn '%s' initializing...", vimba_name, serial)
        self.cam.__enter__()

        # Try to set some default properties, with warnings if they fail.
        try:
            self.cam.BinningHorizontal.set(1)
            self.cam.BinningVertical.set(1)
        except:
            logger.warning("Failed to set binning to 1.")

        try:
            self.cam.GainAuto.set("Off")
        except:
            logger.warning("Failed to turn autogain off.")

        try:
            self.cam.ExposureAuto.set("Off")
            self.cam.ExposureMode.set("Timed")
        except:
            logger.warning("Failed to set exposure mode to timed.")


        try:
            self.cam.AcquisitionMode.set("SingleFrame")

            # Future: triggered instead of SingleFrame.
            self.cam.TriggerSelector.set("AcquisitionStart")
            self.cam.TriggerMode.set("Off")
            self.cam.TriggerActivation.set("RisingEdge")
            self.cam.TriggerSource.set("Software")
        except:
            logger.warning("Failed to set acquisition and trigger configuration.")

        # Cache whether the camera has ExposureTimeAbs or ExposureTime, for use
        # in the get/set exposure methods.
        if hasattr(self.cam, "ExposureTime"):
            self._exposure_time_has_abs = False
        elif hasattr(self.cam, "ExposureTimeAbs"):
            self._exposure_time_has_abs = True
        else:
            self._exposure_time_has_abs = None

        # Populate the rest of the class.
        # PixelSize often reports an enum string like "Bpp8"/"Bpp10"/"Bpp12", so extract
        # the digits rather than int(...) directly (which would raise for those models).
        pixel_size = str(self.cam.PixelSize.get())
        super().__init__(
            (self.cam.SensorWidth.get(), self.cam.SensorHeight.get()),
            bitdepth=int("".join(char for char in pixel_size if char.isdigit())),
            pitch_um=pitch_um,
            name=serial,
            **kwargs,
        )

    def close(self, close_sdk=True):
        """
        See :meth:`.Camera.close`

        Parameters
        ----------
        close_sdk : bool
            Whether or not to close the :mod:`vmbpy` instance.
        """
        self.cam.__exit__(None, None, None)

        if close_sdk:
            self.close_sdk()

        del self.cam

    @staticmethod
    def info(verbose=True):
        """
        Discovers all AlliedVision cameras.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of str
            List of AlliedVision serial numbers.
        """
        if vimba_system is None:
            raise ImportError("vimba or vmbpy are not installed. Install to use AlliedVision cameras.")

        if AlliedVision.sdk is None:
            AlliedVision.sdk = vimba_system.get_instance()
            AlliedVision.sdk.__enter__()
            close_sdk = True
        else:
            close_sdk = False

        camera_list = AlliedVision.sdk.get_all_cameras()
        serial_list = [cam.get_serial() for cam in camera_list]

        if verbose:
            print(f"{vimba_name} serials:")
            for serial in serial_list:
                print(f"'{serial}'")

        if close_sdk:
            AlliedVision.close_sdk()

        return serial_list

    @classmethod
    def close_sdk(cls):
        """
        Close the :mod:`vmbpy` instance.
        """
        if cls.sdk is not None:
            cls.sdk.__exit__(None, None, None)
            cls.sdk = None

    ### Property Configuration ###

    def get_properties(self, properties=None):
        """
        Print the list of camera properties.

        Parameters
        ----------
        properties : dict or None
            The target camera's property dictionary. If ``None``, the property
            dictionary is fetched from the camera associated with the calling instance.
        """
        if properties is None:
            properties = self.cam.__dict__.keys()

        for key in properties:
            prop = self.cam.__dict__[key]
            try:
                print(prop.get_name(), end="\t")
            except BaseException as e:
                print(f"Error accessing property dictionary, '{key}':{e}")
                continue

            try:
                print(prop.get(), end="\t")
            except:
                pass

            try:
                print(prop.get_unit(), end="\t")
            except:
                pass

            try:
                print(prop.get_description(), end="\n")
            except:
                print("")

    def set_adc_bitdepth(self, bitdepth):
        """
        Set the digitization bitdepth.

        Parameters
        ----------
        bitdepth : int
            Desired digitization bitdepth.
        """
        bitdepth = int(bitdepth)

        for entry in self.cam.SensorBitDepth.get_all_entries():
            value = entry.as_tuple()  # (name : str, value : int)
            if str(bitdepth) in value[0]:
                self.cam.SensorBitDepth.set(value[1])
                break
        else:
            raise RuntimeError(f"ADC bitdepth {bitdepth} not found.")

    def get_adc_bitdepth(self):
        """
        Get the digitization bitdepth.

        Returns
        -------
        int
            The digitization bitdepth.
        """
        value = str(self.cam.SensorBitDepth.get())
        bitdepth = int("".join(char for char in value if char.isdigit()))
        return bitdepth

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        if self._exposure_time_has_abs is True:
            return float(self.cam.ExposureTimeAbs.get()) / 1e6
        elif self._exposure_time_has_abs is False:
            return float(self.cam.ExposureTime.get()) / 1e6
        else:
            raise RuntimeError("Camera does not have ExposureTime or ExposureTimeAbs property.")

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        if self._exposure_time_has_abs is True:
            self.cam.ExposureTimeAbs.set(float(exposure_s * 1e6))
        elif self._exposure_time_has_abs is False:
            self.cam.ExposureTime.set(float(exposure_s * 1e6))
        else:
            raise RuntimeError("Camera does not have ExposureTime or ExposureTimeAbs property.")

    def _set_woi_hw(self, woi):
        """See :meth:`.Camera._set_woi_hw`."""
        # AlliedVision (GenICam): OffsetX/OffsetY/Width/Height are in
        # binned output pixels when binning is active (same convention as Basler).
        x, w, y, h = [int(v) for v in woi]
        self.cam.OffsetX.set(0)
        self.cam.OffsetY.set(0)
        self.cam.Width.set(w)
        self.cam.Height.set(h)
        self.cam.OffsetX.set(x)
        self.cam.OffsetY.set(y)

    def _get_woi_hw(self):
        """See :meth:`.Camera._get_woi_hw`."""
        return (
            int(self.cam.OffsetX.get()),
            int(self.cam.Width.get()),
            int(self.cam.OffsetY.get()),
            int(self.cam.Height.get()),
        )

    def _set_binning_hw(self, binning):
        """See :meth:`.Camera._set_binning_hw`."""
        binx, biny = int(binning[0]), int(binning[1])
        self.cam.BinningHorizontal.set(binx)
        self.cam.BinningVertical.set(biny)

    def _get_binning_hw(self):
        """See :meth:`.Camera._get_binning_hw`."""
        return (
            int(self.cam.BinningHorizontal.get()),
            int(self.cam.BinningVertical.get()),
        )

    def _get_image_hw(self, timeout_s):
        """See :meth:`.Camera._get_image_hw`."""
        t = time.time()

        # Convert timeout_s to ms
        frame = self.cam.get_frame(timeout_ms=int(1e3 * timeout_s))
        frame = frame.as_numpy_ndarray()

        # We have noticed that sometimes the camera gets into a state where
        # it returns a frame of all zeros apart from one pixel with value of 31.
        # This method is admittedly a hack to try getting a frame a few more times.
        # We welcome contributions to fix this.
        while np.sum(frame) == np.amax(frame) == 31 and time.time() - t < timeout_s:
            frame = self.cam.get_frame(timeout_ms=int(1e3 * timeout_s))
            frame = frame.as_numpy_ndarray()

        return np.squeeze(frame)
