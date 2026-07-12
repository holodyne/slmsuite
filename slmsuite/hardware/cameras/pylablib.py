"""
Light wrapper for the :mod:`pylablib` package.
See the supported `cameras
<https://pylablib.readthedocs.io/en/stable/devices/cameras_root.html>`_.
:mod:`pylablib` must be installed ``pip install pylablib``.
For example, the following code loads a UC480 camera:

.. highlight:: python
.. code-block:: python

    # Load a legacy Thorlabs camera using the UC480 driver.
    import pylablib as pll
    pll.par["devices/dlls/uc480"] = "path/to/uc480/dlls"
    from pylablib.devices.uc480 import UC480Camera
    pll_cam = UC480Camera()

    # Wrap the camera with the slmsuite-compatible class.
    from slmsuite.hardware.cameras.pylablib import PyLabLib
    cam = PyLabLib(pll_cam)

Note
~~~~
Color cameras reduce each frame to a single channel selected by the base-class
:attr:`~slmsuite.hardware.cameras.camera.Camera.color_channel` setting, for both
single-frame and batch/averaging acquisition.
"""
import numpy as np
import warnings
from slmsuite.hardware.cameras.camera import Camera

try:
    from pylablib.devices.interface.camera import ICamera
except:
    ICamera = None
    warnings.warn("pylablib not installed. Install to use PyLabLib cameras.")

from slmsuite._logging import make_logger

logger = make_logger(__name__)

class PyLabLib(Camera):
    """
    A wrapped :mod:`pylablib` camera.

    Attributes
    ----------
    cam : pylablib.devices.interface.camera.ICamera
        Object to talk with the desired camera.
    """

    ### Initialization and termination ###

    def __init__(self, cam=None, pitch_um=None, **kwargs):
        """
        Initialize camera and attributes. Initial profile is ``"single"``.

        Parameters
        ----------
        cam : pylablib.devices.interface.camera.Camera
            This class is just a wrapper for :mod:`pylablib`, so the user must pass a
            constructed :mod:`pylablib` camera. For example:

            .. highlight:: python
            .. code-block:: python

                # Load a legacy Thorlabs camera using the UC480 driver.
                import pylablib as pll
                pll.par["devices/dlls/uc480"] = "path/to/uc480/dlls"
                from pylablib.devices.uc480 import UC480Camera
                pll_cam = UC480Camera()

                # Wrap the camera with the slmsuite-compatible class.
                from slmsuite.hardware.cameras.pylablib import PyLabLib
                cam = PyLabLib(pll_cam)

        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        kwargs
            See :meth:`.Camera.__init__` for permissible options.

        Raises
        ------
        RuntimeError
           If the camera can not be reached.
        """
        if ICamera is None:
            raise ImportError("pylablib not installed. Install to use PyLabLib cameras.")

        if not isinstance(cam, ICamera):
            raise ValueError(
                "A subclass of pylablib.devices.interface.camera.Camera must be passed as cam."
            )

        # Create a name for the camera, defaulting to kwargs.
        name = ""
        di = cam.get_device_info()
        info_counter = 1
        for info in di:
            if isinstance(info, str):   # This will usually catch the mode name and serial number.
                name += info + "_"
                info_counter += 1

            if info_counter > 3:
                break
        name = name.strip("_")
        if len(name) == 0:
            name = "pylablibcamera"
        name = kwargs.pop("name", name)

        logger.debug("Cam %s parsing...", name)
        height, width = cam.get_data_dimensions()
        self.cam = cam

        super().__init__(
            (width, height),
            bitdepth=kwargs.pop("bitdepth", 8),     # Currently defaults to 8 because pylablib doesn't cache this for most cameras. Update in the future, maybe.
            pitch_um=pitch_um,                      # Currently unset because pylablib doesn't cache this. Update in the future, maybe.
            name=name,
            **kwargs
        )
        self.logger.debug("PyLabLib camera initialized.")

    def close(self):
        """
        See :meth:`.Camera.close`.
        """
        try:
            self.cam.close()
        except Exception as e:
            raise RuntimeError(
                "This pylablib camera failed to close:\n{}".format(e)
            ) from e

    @staticmethod
    def info(verbose=True):
        """
        Method to load display information.

        Returns
        -------
        list
            Always raises :exc:`RuntimeError`.
        """
        raise RuntimeError(
            ".info() is not applicable to pylablib cameras, which must be "
            "constructed outside this wrapper."
        )

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        return self.cam.get_exposure()

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        self.cam.set_exposure(float(exposure_s))

    def _set_woi_hw(self, woi):
        """See :meth:`.Camera._set_woi_hw`. **(Untested)**"""
        # pylablib ROI coordinates are physical (unbinned) sensor pixels, exclusive end.
        # https://pylablib.readthedocs.io/en/stable/_modules/pylablib/devices/Thorlabs/TLCamera.html
        binx, biny = self._binning
        x, w, y, h = (int(v) for v in woi)
        roi = dict(hstart=x * binx, hend=(x + w) * binx, vstart=y * biny, vend=(y + h) * biny)
        try:
            self.cam.set_roi(**roi, hbin=binx, vbin=biny)
        except:
            # Some pylablib cameras don't support setting binning alongside the ROI.
            self.cam.set_roi(**roi)

    def _get_woi_hw(self):
        """See :meth:`.Camera._get_woi_hw`. **(Untested)**"""
        # pylablib get_roi() returns (hstart, hend, vstart, vend[, hbin, vbin]) in physical pixels.
        binx, biny = self._binning
        roi = self.cam.get_roi()
        x_p = int(roi[0])
        w_p = int(roi[1]) - x_p
        y_p = int(roi[2])
        h_p = int(roi[3]) - y_p
        return (x_p // binx, w_p // binx, y_p // biny, h_p // biny)

    def _set_binning_hw(self, binning):
        """See :meth:`.Camera._set_binning_hw`."""
        # self._woi is already in physical (unbinned) pixels, so send it directly with the
        # new binning (the base re-applies the WOI afterward).
        binx, biny = binning
        x, w, y, h = (int(v) for v in self._woi)
        self.cam.set_roi(hstart=x, hend=x + w, vstart=y, vend=y + h, hbin=binx, vbin=biny)

    def _get_binning_hw(self):
        """See :meth:`.Camera._get_binning_hw`."""
        # get_roi() includes (hbin, vbin) as elements 4 and 5 when binning is supported.
        roi = self.cam.get_roi()
        if len(roi) >= 6:
            return (int(roi[4]), int(roi[5]))
        return (1, 1)

    def _get_image_hw(self, timeout_s):
        """
        Method to pull an image from the camera and return.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait for the frame to be fetched (currently unused).

        Returns
        -------
        numpy.ndarray
            Array of shape :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`.
        """
        return self.cam.snap(timeout=timeout_s)

    def _get_images_hw(self, image_count, timeout_s, out=None):
        """See :meth:`.Camera._get_images_hw`."""
        imgs = self.cam.grab(nframes=image_count, frame_timeout=timeout_s)
        if out is not None:
            out[...] = imgs
            return out
        else:
            return np.array(imgs)
