"""
Wraps OpenCV's :mod:`cv2` ``VideoCapture`` class, which supports many webcams and videostreams.
"""
import numpy as np
import cv2
import time

from slmsuite.hardware.cameras.camera import Camera

class Webcam(Camera):
    """
    Wraps OpenCV's :mod:`cv2` ``VideoCapture`` class,
    which supports many webcams and videostreams.

    Warning
    -------
    This class does not properly handle color images
    and does not properly populate datatype information.
    Webcams usually support different codecs, and only the default is enabled here.
    Exposure may not be handled correctly for some cameras.

    See Also
    --------
    `OpenCV documentation <https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html>`_.

    Attributes
    ----------
    cam : cv2.VideoCapture
        Most cameras will wrap some handle which connects to the hardware.
    """

    def __init__(
        self,
        identifier=0,
        resolution=None,
        capture_api=cv2.CAP_ANY,
        pitch_um=None,
        verbose=True,
        **kwargs
    ):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        identifier : int OR str
            Identity of the camera to open. This can be either an integer index
            (numbered by the OS) or a string URL of a videostream
            (e.g. ``protocol://host:port/script_name?script_params|auth``).
            The OS's default camera (index of ``0``) is used as the default.
        resolution : (int, int) OR None
            Desired ``(width, height)`` in pixels. If ``None``, uses the camera's
            current resolution. The camera may round to the nearest supported
            mode; the actual resolution is read back after setting.

            Note
            ~~~~
            Webcam resolution changes the scale of the full sensor output; this is not a
            WOI crop. All supported resolutions cover the same field of view.
        capture_api : int
            The ``cv2.VideoCaptureAPI`` to use for capturing.
            Essentially, this is the driver that should be used.
            Defaults to ``cv2.CAP_ANY`` (choose OS default).
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        verbose : bool
            Whether or not to print extra information.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        id = f'{identifier}' if isinstance(identifier, str) else identifier
        if verbose: print(f"Webcam {id} initializing... ", end="")
        self.cam = cv2.VideoCapture(identifier, capture_api)
        time.sleep(.5)
        if not self.cam.isOpened():
            raise RuntimeError(f"Failed to initialize webcam {id}")

        time.sleep(.5)

        # Request the desired resolution before reading back the actual values.
        if resolution is not None:
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution[0]))
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution[1]))
            time.sleep(.5)

        # Read back the actual resolution the camera settled on.
        super().__init__(
            (
                int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            bitdepth=8,
            pitch_um=pitch_um,
            name=str(identifier),
            **kwargs
        )

        time.sleep(.5)
        self.backend = self.cam.getBackendName()
        self.set_auto_exposure(False)
        time.sleep(.5)
        self.set_exposure(self.get_exposure())
        time.sleep(.5)
        if verbose: print("success")

    def close(self):
        """See :meth:`.Camera.close`."""
        self.cam.release()
        del self.cam

    @staticmethod
    def info(verbose=True):
        """Not supported by :class:`Webcam`."""
        raise NotImplementedError()

    def get_auto_exposure(self):
        return self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE)

    def set_auto_exposure(self, tf):
        self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if tf else 1)

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        return 2**float(self.cam.get(cv2.CAP_PROP_EXPOSURE))

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        self.cam.set(cv2.CAP_PROP_EXPOSURE, np.log2(exposure_s))

    def _get_image_hw(self, timeout_s):
        """See :meth:`.Camera._get_image_hw`."""
        (success, img) = self.cam.read()
        if not success: raise RuntimeError("Could not grab frame.")
        img = np.array(img)
        if len(img.shape) == 3:
            return img[:,:,::-1]    # Flip BGR to RGB; FUTURE: Make more general.
        else:
            return img