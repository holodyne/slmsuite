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
Color camera functionality is not currently implemented, and will lead to undefined behavior.
"""
import warnings
from slmsuite.hardware.cameras.camera import Camera

try:
    from pylablib.devices.interface.camera import ICamera
except:
    ICamera = None
    warnings.warn("pylablib not installed. Install to use PyLabLib cameras.")

class PyLabLib(Camera):
    """
    A wrapped :mod:`pylablib` camera.

    Attributes
    ----------
    cam : pylablib.devices.interface.camera.ICamera
        Object to talk with the desired camera.
    """

    ### Initialization and termination ###

    def __init__(self, cam=None, pitch_um=None, verbose=True, **kwargs):
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
        verbose : bool
            Whether or not to print extra information.
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

        if verbose: print(f"Cam {name} parsing... ", end="")
        height, width = cam.get_data_dimensions()
        self.cam = cam

        super().__init__(
            (width, height),
            bitdepth=8,         # Currently defaults to 8 because pylablib doesn't cache this. Update in the future, maybe.
            pitch_um=pitch_um,  # Currently unset because pylablib doesn't cache this. Update in the future, maybe.
            name=name,
            **kwargs
        )
        if verbose: print("success")

    def close(self):
        """
        See :meth:`.Camera.close`.
        """
        try:
            self.cam.close()
        except:
            raise RuntimeError("This instrumental camera does not support .close().")

    @staticmethod
    def info(verbose=True):
        """
        Method to load display information.

        Returns
        -------
        list
            An empty list.
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

    def _get_roi(self, woi=None, binning=None):
        if woi is None:
            woi = self._woi
        if binning is None:
            biny, binx = self._binning
        else:
            biny, binx = binning

        x, w, y, h = [int(v) for v in woi]
        x_p, w_p, y_p, h_p = x * binx, w * binx, y * biny, h * biny

        return (x_p, x_p + w_p, y_p, y_p + h_p, binx, biny)

    def _set_woi_hw(self, woi):
        """See :meth:`.Camera._set_woi_hw`."""
        # pylablib ROI coordinates are physical (unbinned) sensor pixels, exclusive end.
        # https://pylablib.readthedocs.io/en/stable/_modules/pylablib/devices/Thorlabs/TLCamera.html
        hstart, hend, vstart, vend, binx, biny = self._get_roi(woi=woi, binning=None)
        try:
            self.cam.set_roi(
                hstart=hstart, hend=hend,
                vstart=vstart, vend=vend,
                hbin=binx, vbin=biny
            )
        except:
            # Some pylablib cameras don't support setting binning alongside ROI. Try setting ROI without binning.
            self.cam.set_roi(
                hstart=hstart, hend=hend,
                vstart=vstart, vend=vend
            )

    def _get_woi_hw(self):
        """See :meth:`.Camera._get_woi_hw`."""
        # pylablib get_roi() returns (hstart, hend, vstart, vend[, hbin, vbin]) in physical pixels.
        biny, binx = self._binning
        roi = self.cam.get_roi()
        x_p = int(roi[0])
        w_p = int(roi[1]) - x_p
        y_p = int(roi[2])
        h_p = int(roi[3]) - y_p
        return (x_p // binx, w_p // binx, y_p // biny, h_p // biny)

    def _set_binning_hw(self, binning):
        """See :meth:`.Camera._set_binning_hw`."""
        # pylablib set_roi accepts hbin/vbin to set binning alongside ROI.
        hstart, hend, vstart, vend, binx, biny = self._get_roi(woi=None, binning=binning)

        self.cam.set_roi(hstart=hstart, hend=hend, vstart=vstart, vend=vend, hbin=binx, vbin=biny)

    def _get_binning_hw(self):
        """See :meth:`.Camera._get_binning_hw`."""
        # get_roi() includes (hbin, vbin) as elements 4 and 5 when binning is supported.
        roi = self.cam.get_roi()
        if len(roi) >= 6:
            return (int(roi[5]), int(roi[4]))
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
        return self.cam.grab(nframes=image_count, frame_timeout=timeout_s)
