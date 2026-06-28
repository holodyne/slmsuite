"""
**(Untested)** Hardware control for The Imaging Source cameras via :mod:`tisgrabber`.
:mod:`tisgrabber` is one of several different interfaces that The Imaging Source supports.
See
`the tisgrabber source
<https://github.com/TheImagingSource/IC-Imaging-Control-Samples/tree/master/Python/tisgrabber>`_.
This was tested at commit 7846b9e and Python 3.9 with DMK 27BUP031 camera.
The tisgrabber .dll and tisgrabber.py are needed.
Please either install tisgrabber.py or have it in your current working directory.
"""
import warnings
import ctypes
import numpy as np

from slmsuite.hardware.cameras.camera import Camera

try:
    import tisgrabber as tis
except:
    tis = None
    warnings.warn("tisgrabber not installed. Install to use ImagingSource cameras.")

from slmsuite._logging import make_logger

logger = make_logger(__name__)


# Change this DLL path if necessary
DLL_PATH = "./tisgrabber_x64.dll"

class ImagingSource(Camera):
    """
    The Imaging Source camera.

    Attributes
    ----------
    sdk : ctypes.CDLL
        Connects to the Imaging Source SDK. Shared among instances of :class:`ImagingSource`.
    cam : HGRABBER
        Object to talk with the camera. See tisgrabber.h or tisgrabber documentation for more details
    vid_format : str
        Caches the video format currently set by the user if known.
    """
    sdk = None

    @classmethod
    def init_sdk(cls):
        """
        Class method for initializing the sdk. Called when the first instance is instantiated or when the static method info is called.

        Parameters
        ----------
        cls : object
            required parameter for a class method.

        Raises
        ------
        RuntimeError
           If the library fails to initiate. See tisgrabber.h for error codes.
        """
        sdk = ctypes.cdll.LoadLibrary(DLL_PATH)
        tis.declareFunctions(sdk)

        err = sdk.IC_InitLibrary(0)
        if err != 1:
            raise Exception("DLL library failed to initiate. Perhaps check the DLL_PATH in tis_camera.py")

        cls.sdk = sdk

        return err

    @staticmethod
    def safe_call(cb, to_raise, *args, **kwargs):
        """
        Decorator method that automatically error checks the result from callback ``cb``.

        Parameters
        ----------
        cb : function
            Function that is decorated with arguments ``*args`` and ``**kwargs``.
        to_raise : bool
            Whether to raise an exception or simply print out an error.

        Returns
        -------
        err : int
            error code is returned regardless when Exception is raised. Error code information is in tisgrabber.h.
        """
        err = cb(*args, **kwargs)
        if err <= 0:
            err_str = "Error performing operation: " + cb.__name__ + " err code: " + str(err)
            if to_raise:
                raise Exception(err_str)
            else:
                logger.error(err_str)
        return err

    def __init__(
        self,
        serial="",
        vid_format=None,
        pitch_um=None,
        **kwargs
    ):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        serial : str
            This serial is used to open a camera by unique name (see tisgrabber.h).
            It is usually the model name followed by a space and the serial number.
            Use :meth:`.info()` to see detected options.
            If empty, then opens the first camera found.
        vid_format : str
            If None, no format is set and will default to whatever the camera is currently.
            See tisgrabber.h for more information. Example ``"Y800 (2592x1944)"``.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        if tis is None:
            raise ImportError("tisgrabber not installed. Install to use ImagingSource cameras.")

        # Initialize the SDK if needed.
        logger.debug("TIS Camera SDK initializing...")
        if ImagingSource.sdk is None:
            err = ImagingSource.init_sdk()
            if err != 1:
                raise Exception("Error when loading SDK: " + str(err))

        # Then we load the camera from the SDK.
        logger.debug('"%s" initializing...', serial)

        # cam will be the handle that represents the camera.
        self.cam = ImagingSource.sdk.IC_CreateGrabber()
        if serial == "":
            connected_devs = ImagingSource.info()
            if len(connected_devs) == 0:
                raise Exception("No cameras found")
            serial = connected_devs[0] # By default use the first camera that is found
        err = ImagingSource.sdk.IC_OpenDevByUniqueName(self.cam, tis.T(serial))
        if err != 1:
            raise Exception("Error when opening Camera: " + str(err))

        self.vid_format = vid_format

        # Get in prepared mode and then set the video format
        ImagingSource.safe_call(ImagingSource.sdk.IC_PrepareLive, 1, self.cam)
        if vid_format is not None:
            ImagingSource.safe_call(ImagingSource.sdk.IC_SetVideoFormat, 1, self.cam, tis.T(vid_format))

        # Acquire the description of the image.
        width = ctypes.c_long()
        height = ctypes.c_long()
        bpp = ctypes.c_int()
        COLORFORMAT = ctypes.c_int()

        ImagingSource.safe_call(ImagingSource.sdk.IC_GetImageDescription, 1, self.cam, width, height, bpp, COLORFORMAT)

        # Dividing by 3 since it seems like even with format Y800 which is monochrome, it still uses 24 bits per pixel.
        # TODO: fix this to improve read efficiency
        bitdepth = int(bpp.value / 3)

        # Finally, use the superclass constructor to initialize other required variables.
        super().__init__(
            (width.value, height.value),
            bitdepth=bitdepth,
            name=serial,
            pitch_um=pitch_um,
            **kwargs
        )
        self.logger.debug("ImagingSource camera initialized.")

    def close(self):
        """See :meth:`.Camera.close`."""
        ImagingSource.safe_call(ImagingSource.sdk.IC_ReleaseGrabber, self.cam)
        del self.cam

    @staticmethod
    def info(verbose=True):
        """
        Discovers all cameras detected by the SDK.
        Useful for a user to identify the correct serial numbers / etc.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of str
            List of serial numbers or identifiers.
        """
        if tis is None:
            raise ImportError("tisgrabber not installed. Install to use ImagingSource cameras.")

        if ImagingSource.sdk is None:
            err = ImagingSource.init_sdk()
            if err != 1:
                raise Exception("Error when loading SDK: " + str(err))

        # Get device count and then iterate through each device
        devicecount = ImagingSource.sdk.IC_GetDeviceCount()
        serial_list = []
        for i in range(0, devicecount):
            serial_list.append(tis.D(ImagingSource.sdk.IC_GetUniqueNamefromList(i)))

        if verbose: print(serial_list)

        return serial_list

    ### Property Configuration ###

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        exposure = ctypes.c_float()
        ImagingSource.safe_call(ImagingSource.sdk.IC_GetPropertyAbsoluteValue, 1, self.cam, tis.T("Exposure"), tis.T("Value"), exposure)
        return float(exposure.value)

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        # Turn off auto exposure and use the value given.
        ImagingSource.safe_call(ImagingSource.sdk.IC_SetPropertySwitch, 1, self.cam, tis.T("Exposure"), tis.T("Auto"), 0)
        ImagingSource.safe_call(ImagingSource.sdk.IC_SetPropertyAbsoluteValue, 1, self.cam, tis.T("Exposure"), tis.T("Value"), ctypes.c_float(exposure_s))

    def _set_woi_hw(self, woi):
        """See :meth:`.Camera._set_woi_hw`. **(Untested)**"""
        # ImagingSource: width/height in the video format string are output (binned) pixels.
        # Partial scan X/Y offsets are in physical (unbinned) sensor pixels.
        # Ref: https://www.theimagingsource.com/en-us/documentation/icpython/properties.html
        binx, biny = self._binning
        x, w, y, h = [int(v) for v in woi]
        x_phys, y_phys = x * binx, y * biny
        idx = self.vid_format.find("(")
        this_vid_format = self.vid_format[:idx]
        tot_format = this_vid_format + "(" + str(w) + "x" + str(h) + ")"
        ImagingSource.safe_call(ImagingSource.sdk.IC_SetVideoFormat, 1, self.cam, tis.T(tot_format))
        ImagingSource.safe_call(ImagingSource.sdk.IC_SetPropertySwitch, 1, self.cam, tis.T("Partial scan"), tis.T("Auto-center"), 0)
        ImagingSource.safe_call(ImagingSource.sdk.IC_SetPropertyValue, 1, self.cam, tis.T("Partial scan"), tis.T("X Offset"), x_phys)
        ImagingSource.safe_call(ImagingSource.sdk.IC_SetPropertyValue, 1, self.cam, tis.T("Partial scan"), tis.T("Y Offset"), y_phys)

    def _get_woi_hw(self):
        """See :meth:`.Camera._get_woi_hw`. **(Untested)**"""
        # width/height from IC_GetImageDescription are output (binned) pixels.
        # X/Y offsets from IC_GetPropertyValue are physical pixels; divide by binning.
        binx, biny = self._binning
        width = ctypes.c_long()
        height = ctypes.c_long()
        bpp = ctypes.c_int()
        COLORFORMAT = ctypes.c_int()
        ImagingSource.safe_call(ImagingSource.sdk.IC_GetImageDescription, 1, self.cam, width, height, bpp, COLORFORMAT)
        x_offset = ctypes.c_long()
        y_offset = ctypes.c_long()
        ImagingSource.sdk.IC_GetPropertyValue(self.cam, tis.T("Partial scan"), tis.T("X Offset"), x_offset)
        ImagingSource.sdk.IC_GetPropertyValue(self.cam, tis.T("Partial scan"), tis.T("Y Offset"), y_offset)
        return (int(x_offset.value) // binx, int(width.value), int(y_offset.value) // biny, int(height.value))

    def _set_binning_hw(self, binning):
        """See :meth:`.Camera._set_binning_hw`. **(Untested)**"""
        binx, biny = int(binning[0]), int(binning[1])
        if biny != binx:
            raise NotImplementedError("ImagingSource requires symmetric binning.")
        buf = tis.T(str(biny))
        err = ImagingSource.sdk.IC_SetPropertyMapStrings(
            self.cam, tis.T("Binning factor"), tis.T("Value"), buf
        )
        if err <= 0:
            raise NotImplementedError(f"Camera {self.name} does not support binning.")

    def _get_binning_hw(self):
        """See :meth:`.Camera._get_binning_hw`."""
        buf = (ctypes.c_char * 128)()
        err = ImagingSource.sdk.IC_GetPropertyMapStrings(
            self.cam, tis.T("Binning factor"), tis.T("Value"), buf, ctypes.sizeof(buf)
        )
        if err > 0:
            try:
                return (int(buf.value.decode()), int(buf.value.decode()))
            except (ValueError, UnicodeDecodeError):
                pass
        return (1, 1)

    def _get_image_hw(self, timeout_s):
        """See :meth:`.Camera._get_image_hw`."""
        # Raw, untransformed frame shape that hardware delivers (WOI/binning applied in hardware).
        H, W = self._hw_image_shape
        # 8-bit RGB: 3 bytes per pixel (even Y800 mono is delivered as RGB).
        buffer_size = 3 * H * W
        # Starts the image acquisition
        ImagingSource.safe_call(ImagingSource.sdk.IC_StartLive, 0, self.cam, 0)
        # Snap image. IC_SnapImage expects an integer millisecond timeout.
        timeout_ms = int(1000 * timeout_s)
        err = ImagingSource.safe_call(ImagingSource.sdk.IC_SnapImage, 0, self.cam, timeout_ms)
        # If there is an error, then snap image again (bounded so a persistently
        # failing camera cannot loop forever).
        attempts = 1
        while err <= 0 and attempts < self.capture_attempts:
            err = ImagingSource.safe_call(ImagingSource.sdk.IC_SnapImage, 0, self.cam, timeout_ms)
            attempts += 1
        # Get image
        ptr = ImagingSource.safe_call(ImagingSource.sdk.IC_GetImagePtr, 0, self.cam)
        img_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte * buffer_size))
        # Reshape the image according to the width and height.
        # TODO: there are more efficient ways to reshape the array only considering the R component.
        img = np.ndarray(buffer=img_ptr.contents, dtype=np.uint8, shape=(H, W, 3)) # 3 for RGB
        ImagingSource.safe_call(ImagingSource.sdk.IC_StopLive, 0, self.cam)
        # We take only the 1st component, assuming that the image is monochromatic.
        # Return the raw untransformed frame; the base class applies self.transform.
        return np.copy(img[:, :, 0])