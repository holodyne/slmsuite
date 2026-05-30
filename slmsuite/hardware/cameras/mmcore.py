"""
Hardware control for Micro-Manager cameras via the :mod:`pymmcore` interface.
Micro-Manager supports a wide variety of scientific cameras.
See :mod:`pymmcore` `documentation <https://github.com/micro-manager/pymmcore>`_
for installation instructions.
"""

import os
import warnings

from slmsuite.hardware.cameras.camera import Camera

try:
    import pymmcore
except ImportError:
    pymmcore = None
    warnings.warn("pymmcore not installed. Install to use Micro-Manager cameras.")


class MMCore(Camera):
    """
    Micro-Manager camera.

    Attributes
    ----------
    cam : pymmcore.CMMCore
        Object to talk with the desired camera.
    """

    def __init__(
        self,
        config,
        path="C:\\Program Files\\Micro-Manager-2.0",
        pitch_um=None,
        verbose=True,
        **kwargs,
    ):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        config : str
            Name of the config file corresponding to the desired camera. This is assumed to be
            a ``.cfg`` file stored in the Micro-Manager ``path`` (see below), unless an
            absolute path is given.
            The filetype ``.cfg`` may be included or omitted,
            but the :attr:`name` of the camera will be without it.
        path : str
            Directory of the Micro-Manager installation. Defaults to the default Windows
            directory of a Micro-Manager 2.0 installation.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        verbose : bool
            Whether or not to print extra information.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        if pymmcore is None:
            raise ImportError("pymmcore not installed. Install to use Micro-Manager cameras.")

        # Parse config.
        if len(config) > 4 and config[-4:] == ".cfg":
            config = config[:-4]
        config_path, config = os.path.split(config)
        if not os.path.isabs(config_path):
            config_path = os.path.join(path, config_path)

        # Load mmcore.
        if verbose:
            print("CMMCore initializing... ", end="")
        self.cam = pymmcore.CMMCore()
        self.cam.setDeviceAdapterSearchPaths([path])
        if verbose:
            print("success")

        # Load the camera using the config.
        if verbose:
            print(f"'{config}' initializing... ", end="")
        self.cam.loadSystemConfiguration(os.path.join(config_path, config + ".cfg"))

        # Fill in slmsuite variables.
        super().__init__(
            (self.cam.getImageWidth(), self.cam.getImageHeight()),
            bitdepth=self.cam.getImageBitDepth(),
            pitch_um=pitch_um,
            name=config,
            **kwargs,
        )
        if verbose:
            print("success")

    @staticmethod
    def info(path="C:\\Program Files\\Micro-Manager-2.0"):
        """
        Detects ``.cfg`` files present in the given path.

        Parameters
        ----------
        path : str
            Path to search. This path defaults to the default Micro-Manager install location,
            as ``.cfg`` files save there by default.

        Returns
        -------
        list of str
            Names of the detected ``.cfg`` files.
        """
        if pymmcore is None:
            raise ImportError("pymmcore not installed. Install to use Micro-Manager cameras.")


        # Check if the provided path exists and is a directory.
        if os.path.isdir(path):
            cfg_files = [
                file_name
                for file_name in os.listdir(path)
                if file_name.endswith(".cfg")
            ]
        else:
            raise ValueError(f"The provided path '{path}' is not a valid directory.")

        return cfg_files

    def close(self):
        """See :meth:`.Camera.close`."""
        del self.cam

    ### Property Configuration ###

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        return float(self.cam.getExposure()) / 1e3

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        self.cam.setExposure(1e3 * exposure_s)

    def set_woi(self, woi=None):
        """See :meth:`.Camera.set_woi`."""
        return

    def _get_image_hw(self, timeout_s):
        """See :meth:`.Camera._get_image_hw`."""
        self.cam.snapImage()
        return self.cam.getImage()
