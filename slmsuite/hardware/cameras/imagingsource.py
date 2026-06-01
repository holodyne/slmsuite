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

import ctypes
import warnings

import numpy as np

from slmsuite.hardware.cameras.camera import Camera

try:
    import tisgrabber as tis
except Exception:
    tis = None
    warnings.warn("tisgrabber not installed. Install to use ImagingSource cameras.")


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
            raise Exception(
                "DLL library failed to initiate. Perhaps check the DLL_PATH in tis_camera.py"
            )

        cls.sdk = sdk

        return err

    @staticmethod
    def safe_call(cb, to_raise, *args, **kwargs):
        """
        Decorator method that automatically error checks the result from callback `cb`.

        Parameters
        ----------
        cb : function
            Function that is decorated with arguments `*args` and `**kwargs`.
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
                print(err_str)
        return err

    def __init__(self, serial="", vid_format=None, pitch_um=None, verbose=True, **kwargs):
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
        verbose : bool
            Whether or not to print extra information.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        if tis is None:
            raise ImportError("tisgrabber not installed. Install to use ImagingSource cameras.")

        # Initialize the SDK if needed.
        if verbose:
            print("TIS Camera SDK initializing... ", end="")
        if ImagingSource.sdk is None:
            err = ImagingSource.init_sdk()
            if err != 1:
                raise Exception("Error when loading SDK: " + str(err))
        if verbose:
            print("success")

        # Then we load the camera from the SDK.
        if verbose:
            print(f'"{serial}" initializing... ', end="")

        # cam will be the handle that represents the camera.
        self.cam = ImagingSource.sdk.IC_CreateGrabber()
        if serial == "":
            connected_devs = ImagingSource.info()
            if len(connected_devs) == 0:
                raise Exception("No cameras found")
            serial = connected_devs[0]  # By default use the first camera that is found
        err = ImagingSource.sdk.IC_OpenDevByUniqueName(self.cam, tis.T(serial))
        if err != 1:
            raise Exception("Error when opening Camera: " + str(err))

        self.vid_format = vid_format

        # Get in prepared mode and then set the video format
        ImagingSource.safe_call(ImagingSource.sdk.IC_PrepareLive, 1, self.cam)
        if vid_format is not None:
            ImagingSource.safe_call(
                ImagingSource.sdk.IC_SetVideoFormat, 1, self.cam, tis.T(vid_format)
            )

        # Acquire the description of the image.
        width = ctypes.c_long()
        height = ctypes.c_long()
        bpp = ctypes.c_int()
        COLORFORMAT = ctypes.c_int()

        ImagingSource.safe_call(
            ImagingSource.sdk.IC_GetImageDescription, 1, self.cam, width, height, bpp, COLORFORMAT
        )

        # Dividing by 3 since it seems like even with format Y800 which is monochrome, it still uses 24 bits per pixel.
        # TODO: fix this to improve read efficiency
        bitdepth = int(bpp.value / 3)

        # Finally, use the superclass constructor to initialize other required variables.
        super().__init__(
            (width.value, height.value), bitdepth=bitdepth, name=serial, pitch_um=pitch_um, **kwargs
        )
        if verbose:
            print("success")

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
        -------
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
        serial_list = [
            tis.D(ImagingSource.sdk.IC_GetUniqueNamefromList(i)) for i in range(devicecount)
        ]

        if verbose:
            print(serial_list)

        return serial_list

    ### Property Configuration ###

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        exposure = ctypes.c_float()
        ImagingSource.safe_call(
            ImagingSource.sdk.IC_GetPropertyAbsoluteValue,
            1,
            self.cam,
            tis.T("Exposure"),
            tis.T("Value"),
            exposure,
        )
        return float(exposure.value)

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        # Turn off auto exposure and use the value given.
        ImagingSource.safe_call(
            ImagingSource.sdk.IC_SetPropertySwitch, 1, self.cam, tis.T("Exposure"), tis.T("Auto"), 0
        )
        ImagingSource.safe_call(
            ImagingSource.sdk.IC_SetPropertyAbsoluteValue,
            1,
            self.cam,
            tis.T("Exposure"),
            tis.T("Value"),
            ctypes.c_float(exposure_s),
        )

    def set_woi(self, woi=None):
        """See :meth:`.Camera.set_woi`."""
        if woi is None:
            width = ctypes.c_long()
            height = ctypes.c_long()
            bpp = ctypes.c_int()  # Bits per pixel
            COLORFORMAT = ctypes.c_int()

            ImagingSource.safe_call(
                ImagingSource.sdk.IC_GetImageDescription,
                1,
                self.cam,
                width,
                height,
                bpp,
                COLORFORMAT,
            )

            width = width.value
            height = height.value
            self.woi = (0, width, 0, height)
        else:
            # Format for woi is specified in :meth:`.Camera.set_woi`.
            width = int(woi[1])
            height = int(woi[3])
            xpos = int(woi[0])
            ypos = int(woi[2])
            # This keeps the original format
            idx = self.vid_format.find("(")  # TODO: is this general?
            this_vid_format = self.vid_format[:idx]
            # We bring in the new width and height specified in the video format
            tot_format = this_vid_format + "(" + str(width) + "x" + str(height) + ")"
            ImagingSource.safe_call(
                ImagingSource.sdk.IC_SetVideoFormat, 1, self.cam, tis.T(tot_format)
            )
            # Now offset
            ImagingSource.safe_call(
                ImagingSource.sdk.IC_SetPropertySwitch,
                1,
                self.cam,
                tis.T("Partial scan"),
                tis.T("Auto-center"),
                0,
            )
            ImagingSource.safe_call(
                ImagingSource.sdk.IC_SetPropertyValue,
                1,
                self.cam,
                tis.T("Partial scan"),
                tis.T("X Offset"),
                xpos,
            )
            ImagingSource.safe_call(
                ImagingSource.sdk.IC_SetPropertyValue,
                1,
                self.cam,
                tis.T("Partial scan"),
                tis.T("Y Offset"),
                ypos,
            )

        self.shape = (height, width)

    def _get_image_hw(self, timeout_s):
        """See :meth:`.Camera.get_image`."""
        buffer_size = (
            3 * self.bitdepth * self.shape[0] * self.shape[1]
        )  # times 3 is because even Y800 is RGB
        # Starts the image acquisition
        ImagingSource.safe_call(ImagingSource.sdk.IC_StartLive, 0, self.cam, 0)
        # Snap image
        err = ImagingSource.safe_call(ImagingSource.sdk.IC_SnapImage, 0, self.cam, 1000 * timeout_s)
        # If there is an error, then snap image again
        while err <= 0:
            err = ImagingSource.safe_call(
                ImagingSource.sdk.IC_SnapImage, 0, self.cam, 1000 * timeout_s
            )
        # Get image
        ptr = ImagingSource.safe_call(ImagingSource.sdk.IC_GetImagePtr, 0, self.cam)
        img_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte * buffer_size))
        # Reshape the image according to the width and height.
        # TODO: there are more efficient ways to reshape the array only considering the R component.
        img = np.ndarray(
            buffer=img_ptr.contents, dtype=np.uint8, shape=(self.shape[0], self.shape[1], 3)
        )  # 3 for RGB
        ImagingSource.safe_call(ImagingSource.sdk.IC_StopLive, 0, self.cam)
        # We take only the 1st component, assuming that the image is monochromatic.
        return self.transform(img[:, :, 0])
