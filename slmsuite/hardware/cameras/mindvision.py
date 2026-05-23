"""
Hardware control for MindVision cameras via :mod:`mvsdk`.
This requires the :mod:`mvsdk` library and python header to be copied
into the local directory and added to path.
These files can be downloaded from `MindVision's download page
<https://www.mindvision.com.cn/category/software/demo-development-routine/>`_.
The python header can also be found in the `dddomodossola/nastroprint
<https://github.com/dddomodossola/nastroprint/blob/master/mvsdk.py>`_
GitHub package.
"""

import time
import warnings

import numpy as np

from slmsuite.hardware.cameras.camera import Camera

try:
    import mvsdk as _mvsdk
except:
    _mvsdk = None
    warnings.warn("mvsdk not installed.")


class MindVision(Camera):
    """
    MindVision camera subclass for interfacing with the :mod:`mvsdk`.
    """

    # Class variable (same for all instances of MindVision) pointing to a singleton SDK.
    sdk = None

    def __init__(self, serial="", pitch_um=None, verbose=True, **kwargs):
        """
        Initialize the MindVision camera and attributes.

        Parameters
        ----------
        serial : str
            Serial number of the camera to open.
            Use :meth:`.info()` to see detected options.
            If empty, defaults to the first camera in the list
            returned by :meth:`.info()`.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        verbose : bool
            Whether or not to print extra information.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        if _mvsdk is None:
            raise ImportError("mvsdk not installed.")

        if MindVision.sdk is None:
            if verbose:
                print("mvsdk initializing... ", end="")
            _mvsdk._Init()
            if verbose:
                print("success")

        # Grab the list of cameras.
        if verbose:
            print("Looking for cameras... ", end="")
        camera_list = _mvsdk.CameraEnumerateDevice()
        if not camera_list:
            raise RuntimeError("No cameras found by mvsdk.")
        serial_list = [cam.GetSn() for cam in camera_list]

        # Find the camera by serial number or use the first available camera.
        if serial:
            self.cam = next((cam for cam in camera_list if cam.GetSn() == serial), None)
            if self.cam is None:
                raise RuntimeError(f"Serial {serial} not found.\nAvailable: {serial_list}")
        else:
            self.cam = camera_list[0]
            if len(camera_list) > 1 and verbose:
                print(f"No serial given... Choosing first of {serial_list}...")
        if verbose:
            print("success")
        serial = self.cam.GetSn()

        # Turn the camera on.
        if verbose:
            print(f"Initializing sn '{serial}'... ", end="")
        self.handle = 0
        try:
            self.handle = _mvsdk.CameraInit(self.cam, -1, -1)
        except _mvsdk.CameraException as e:
            print(f"CameraInit Failed ({e.error_code}):\n{e.message}")
            raise e

        # Fill in parameters from the capability class.
        self.capability = _mvsdk.CameraGetCapability(self.handle)
        self.mono = self.capability.sIspCapacity.bMonoSensor != 0
        if self.mono:
            _mvsdk.CameraSetIspOutFormat(self.handle, _mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            _mvsdk.CameraSetIspOutFormat(self.handle, _mvsdk.CAMERA_MEDIA_TYPE_BGR8)
            warnings.warn("Camera is not grayscale. Color cameras may cause issues in slmsuite.")
        _mvsdk.CameraSetTriggerMode(self.handle, 1)
        _mvsdk.CameraSetAeState(self.handle, 0)
        _mvsdk.CameraSetExposureTime(self.handle, 30 * 1000)
        _mvsdk.CameraPlay(self.handle)

        # Calculate the size required for the RGB buffer, which is allocated directly according to the maximum resolution of the camera.
        buffer_size = (
            self.capability.sResolutionRange.iWidthMax
            * self.capability.sResolutionRange.iHeightMax
            * (1 if self.mono else 3)
        )

        # Allocate RGB buffer to store images output by ISP
        # Note: RAW data is transmitted from the camera to the PC,
        # and converted to RGB data through the software ISP on the PC
        # (if it is a black and white camera, there is no need to convert the format,
        # but the ISP has other processing, so this buffer also needs to be allocated)
        self.buffer = _mvsdk.CameraAlignMalloc(buffer_size, 16)

        # Fill in superclass parameters from the capability class.
        super().__init__(
            (
                self.capability.sResolutionRange.iWidthMax,
                self.capability.sResolutionRange.iHeightMax,
            ),
            bitdepth=8,
            pitch_um=pitch_um,
            name=serial,
            **kwargs,
        )
        if verbose:
            print("success")

    def close(self):
        """
        Close the camera and release resources.
        """
        if self.handle:
            _mvsdk.CameraAlignFree(self.buffer)
            _mvsdk.CameraUnInit(self.handle)
            self.handle = None

    @staticmethod
    def info(verbose=True):
        """
        Discovers all cameras detected by the :mod:`mvsdk`.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        -------
        list of str
            List of :mod:`mvsdk` serial numbers.
        """
        if _mvsdk is None:
            raise ImportError(
                "mvsdk not installed. Copy mvsdk.py from dddomodossola/nastroprint to use Mindvision cameras."
            )

        camera_list = _mvsdk.CameraEnumerateDevice()
        serial_list = [cam.GetSn() for cam in camera_list]

        if verbose:
            for cam in camera_list:
                print(f"'{cam.GetSn()}': {cam.GetFriendlyName()} ({cam.GetPortType()})")

        return serial_list

    def print_capability(self):
        """
        MindVision specific method (copied from MindVision examples) to print the
        capability of the camera.
        """
        cap = self.capability

        for i in range(cap.iTriggerDesc):
            desc = cap.pTriggerDesc[i]
            print(f"{desc.iIndex}: {desc.GetDescription()}")
        for i in range(cap.iImageSizeDesc):
            desc = cap.pImageSizeDesc[i]
            print(f"{desc.iIndex}: {desc.GetDescription()}")
        for i in range(cap.iClrTempDesc):
            desc = cap.pClrTempDesc[i]
            print(f"{desc.iIndex}: {desc.GetDescription()}")
        for i in range(cap.iMediaTypeDesc):
            desc = cap.pMediaTypeDesc[i]
            print(f"{desc.iIndex}: {desc.GetDescription()}")
        for i in range(cap.iFrameSpeedDesc):
            desc = cap.pFrameSpeedDesc[i]
            print(f"{desc.iIndex}: {desc.GetDescription()}")
        for i in range(cap.iPackLenDesc):
            desc = cap.pPackLenDesc[i]
            print(f"{desc.iIndex}: {desc.GetDescription()}")
        for i in range(cap.iPresetLut):
            desc = cap.pPresetLutDesc[i]
            print(f"{desc.iIndex}: {desc.GetDescription()}")
        for i in range(cap.iAeAlmSwDesc):
            desc = cap.pAeAlmSwDesc[i]
            print(f"{desc.iIndex}: {desc.GetDescription()}")
        for i in range(cap.iAeAlmHdDesc):
            desc = cap.pAeAlmHdDesc[i]
            print(f"{desc.iIndex}: {desc.GetDescription()}")
        for i in range(cap.iBayerDecAlmSwDesc):
            desc = cap.pBayerDecAlmSwDesc[i]
            print(f"{desc.iIndex}: {desc.GetDescription()}")
        for i in range(cap.iBayerDecAlmHdDesc):
            desc = cap.pBayerDecAlmHdDesc[i]
            print(f"{desc.iIndex}: {desc.GetDescription()}")

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        return _mvsdk.CameraGetExposureTime(self.handle) / 1e6

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        _mvsdk.CameraSetExposureTime(self.handle, exposure_s * 1e6)

    def set_woi(self, woi=None):
        """See :meth:`.Camera.set_woi`."""
        return

    def _get_image_hw(self, timeout_s):
        # TODO: are the following two commands necessary for every call?

        # Switch camera mode to continuous acquisition.
        # _mvsdk.CameraSetTriggerMode(self.handle, 0)
        _mvsdk.CameraSoftTrigger(self.handle)

        # Let the SDK internal image taking thread start working.
        # _mvsdk.CameraPlay(self.handle)

        # Get a frame from the camera
        try:
            #
            raw_data, frame_head = _mvsdk.CameraGetImageBuffer(self.handle, int(timeout_s * 1000))

            # FUTURE: Go directly from the raw_data to numpy instead of through self.buffer?
            _mvsdk.CameraImageProcess(self.handle, raw_data, self.buffer, frame_head)
            _mvsdk.CameraReleaseImageBuffer(self.handle, raw_data)

            # Reshape the buffer as a numpy array, and return a copy.
            frame_data = (_mvsdk.c_ubyte * frame_head.uBytes).from_address(self.buffer)

            if self.mono:
                return np.copy(np.frombuffer(frame_data, dtype=np.uint8).reshape(self.shape))
            else:
                rgb_shape = (self.shape[0], self.shape[1], 3)
                return np.copy(np.frombuffer(frame_data, dtype=np.uint8).reshape(rgb_shape))

        except _mvsdk.CameraException as e:
            print(f"CameraGetImageBuffer failed ({e.error_code}):\n{e.message}")
