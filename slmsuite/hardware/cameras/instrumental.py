"""
Light wrapper for the :mod:`instrumental-lib` package.
The :mod:`instrumental` module must be
`installed <https://instrumental-lib.readthedocs.io/en/stable/install.html>`_.
``pip install instrumental-lib``.
For example, the following code loads a UC480 camera:

.. highlight:: python
.. code-block:: python

    # Load a legacy Thorlabs camera using the UC480 driver.
    from instrumental.drivers.cameras.uc480 import UC480Camera

    i_cam = UC480Camera()

    # Wrap the camera with the slmsuite-compatible class.
    from slmsuite.hardware.cameras.instrumental import Instrumental

    cam = Instrumental(i_cam)

Note
~~~~
Color camera functionality is not currently implemented, and will lead to undefined behavior.
"""

import warnings

from slmsuite.hardware.cameras.camera import Camera

try:
    from instrumental import instrument, list_instruments
    from instrumental.drivers import ParamSet
    import instrumental.drivers.cameras as instrumental_cameras
except ImportError:
    instrument = None
    warnings.warn("instrumental-lib not installed. Install to use Instrumental cameras.")


class Instrumental(Camera):
    """
    A wrapped :mod:`instrumental` camera.

    Attributes
    ----------
    cam : instrumental.drivers.cameras.Camera
        Object to talk with the desired camera.
    exposure_s : float
        Instrumental doesn't save exposure. It sets the exposure at each
        :meth:`get_image`. This variable stores the desired exposure.
        Defaults to .001 (1 ms).
    """

    ### Initialization and termination ###

    def __init__(self, cam=None, pitch_um=None, verbose=True, **kwargs):
        """
        Initialize camera and attributes. Initial profile is ``"single"``.

        Parameters
        ----------
        cam : instrumental.drivers.cameras.Camera OR instrumental.drivers.ParamSet OR None
            This class is just a wrapper for :mod:`instrumental`, so the user must pass a
            constructed :mod:`instrumental` camera or equivalent.
            If a ``ParamSet`` is provided, the camera is constructed.
            If ``None``, the first instrument in ``instrumental.list_instruments()`` is used,
            with errors or warnings if there are many or no instruments available.
            For example, the following code loads constructed camera:

            .. highlight:: python
            .. code-block:: python

                # Load a legacy Thorlabs camera using the UC480 driver.
                from instrumental.drivers.cameras.uc480 import UC480Camera

                i_cam = UC480Camera()

                # Wrap the camera with the slmsuite-compatible class.
                from slmsuite.hardware.cameras.instrumental import Instrumental

                cam = Instrumental(i_cam)

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
        if instrument is None:
            raise ImportError(
                "instrumental-lib not installed. Install to use Instrumental cameras."
            )

        if cam is None:
            instruments = list_instruments()

            if len(instruments) == 0:
                raise RuntimeError("No instrumental cameras detected.")
            else:
                if len(instruments) > 1:
                    warnings.warn(f"Multiple instruments detected; using first of {instruments}")
                cam = list_instruments()[0]

        if isinstance(cam, ParamSet):
            cam = instrument(cam, reopen_policy="reuse")

        if not isinstance(cam, instrumental_cameras.Camera):
            raise ValueError(
                "A subclass of instrumental.drivers.cameras.Camera must be passed as cam."
            )

        name = kwargs.pop("name", cam.model.decode("utf-8") + "_" + cam.serial.decode("utf-8"))
        if verbose:
            print(f"Cam {name} parsing... ", end="")
        self.cam = cam

        super().__init__(
            (self.cam.width, self.cam.height),
            bitdepth=8,  # Currently defaults to 8 because instrumental doesn't cache this. Update in the future, maybe.
            pitch_um=pitch_um,  # Currently unset because instrumental doesn't cache this. Update in the future, maybe.
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
        except Exception:
            raise RuntimeError("This instrumental camera does not support .close().") from None

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
            ".info() is not applicable to instrumental cameras, which must be "
            "constructed outside this wrapper."
        )

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        return float(self.cam.exposure._magnitude) / 1000

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        self.cam.exposure = 1000.0 * float(exposure_s)

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
            The time in seconds to wait for the frame to be fetched.

        Returns
        -------
        numpy.ndarray
            Array of shape :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`.
        """
        return self.cam.grab_image(timeout=str(timeout_s) + "s", copy=True)
