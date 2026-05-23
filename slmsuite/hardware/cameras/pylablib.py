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
    A wrapped :mod:`instrumental` camera.

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
            if isinstance(info, str):  # This will usually catch the mode name and serial number.
                name += info + "_"
                info_counter += 1

            if info_counter > 3:
                break
        name = name.strip("_")
        if len(name) == 0:
            name = "pylablibcamera"
        name = kwargs.pop("name", name)

        if verbose:
            print(f"Cam {name} parsing... ", end="")
        height, width = cam.get_data_dimensions()
        self.cam = cam

        super().__init__(
            (width, height),
            bitdepth=8,  # Currently defaults to 8 because pylablib doesn't cache this. Update in the future, maybe.
            pitch_um=pitch_um,  # Currently unset because pylablib doesn't cache this. Update in the future, maybe.
            name=name,
            **kwargs,
        )
        if verbose:
            print("success")

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

    def set_woi(self, woi=None):
        """
        Method to narrow the imaging region to a 'window of interest'
        for faster framerates.

        Parameters
        ----------
        woi : list, None
            See :attr:`~slmsuite.hardware.cameras.camera.Camera.woi`.
            If ``None``, defaults to largest possible.

        Returns
        -------
        woi : list
            :attr:`~slmsuite.hardware.cameras.camera.Camera.woi`.
        """
        raise NotImplementedError()

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
