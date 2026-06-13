"""
Template for writing a subclass for camera hardware control in :mod:`slmsuite`.
Outlines which camera superclass functions must be implemented.
"""

from slmsuite.hardware.cameras.camera import Camera

class Template(Camera):
    """
    Template for adding a new camera to :mod:`slmsuite`. Replace :class:`Template`
    with the desired subclass name. :class:`~slmsuite.hardware.cameras.camera.Camera` is the
    superclass that sets the requirements for :class:`Template`.

    Attributes
    ----------
    sdk : object
        Many cameras have a singleton SDK class which handles all the connected cameras
        of a certain brand. This is generally implemented as a class variable.
    cam : object
        Most cameras will wrap some handle which connects to the hardware.
    """

    # Class variable (same for all instances of Template) pointing to a singleton SDK.
    sdk = None

    def __init__(
        self,
        serial="",
        pitch_um=None,
        verbose=True,
        **kwargs
    ):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        serial : str
            TODO: Most SDKs identify different cameras by some serial number or string.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
            TODO: See if the SDK can pull this information directly from the camera.
        verbose : bool
            Whether or not to print extra information.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        # TODO: Insert code here to initialize the camera hardware, load properties, etc.

        # Mandatory functions:
        # - Opening a connection to the device

        # Other possibilities to consider:
        # - Loading a connection to the SDK, if applicable.
        # - Gathering parameters such a width, height, and bitdepth.

        # Most cameras have an SDK that needs to be loaded before the camera
        if verbose: print("Template SDK initializing... ", end="")
        raise NotImplementedError()
        Template.sdk = something()                      # TODO: Fill in proper function.
        if verbose: print("success")

        # Then we load the camera from the SDK
        if verbose: print(f"'{serial}' initializing... ", end="")
        raise NotImplementedError()
        self.cam = sdk.something(serial)                # TODO: Fill in proper function.

        # ... Other setup.

        # Finally, use the superclass constructor to initialize other required variables.
        super().__init__(
            (self.cam.get_width(), self.cam.get_height()),  # TODO: Fill in proper functions.
            bitdepth=self.cam.get_depth(),
            pitch_um=pitch_um,
            name=serial,
            **kwargs
        )
        if verbose: print("success")

    def close(self):
        """See :meth:`.Camera.close`."""
        raise NotImplementedError()
        self.cam.close()                                # TODO: Fill in proper function.
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
        raise NotImplementedError()
        serial_list = Template.sdk.get_serial_list()    # TODO: Fill in proper function.
        return serial_list

    ### Required Methods ###

    ## Exposure (used in autoexpose and other functions)
    # User-facing is in seconds. SDK might require different units (e.g. milliseconds).

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        raise NotImplementedError()
        return float(self.cam.get_exposure()) / 1e3     # TODO: Fill in proper function.

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        raise NotImplementedError()
        self.cam.set_exposure(1e3 * exposure_s)         # TODO: Fill in proper function.

    ## Core imaging

    def _get_image_hw(self, timeout_s):
        """See :meth:`.Camera._get_image_hw`."""
        raise NotImplementedError()
        # The core method: grabs an image from the camera.
        # Note: the camera superclass' get_image function performs follow-on processing
        # (similar to how the SLM superclass' set_phase method pairs with _set_phase_hw methods
        # for each subclass) -- frame averaging, transformations, and so on -- so this
        # method should be limited to camera-interface specific functions.
        return self.cam.get_image_function()     # TODO: Fill in proper function.

    ### Optional Methods ###

    ### Secondary imaging (some cameras offer batch image capture)

    # def _get_images_hw(self, image_count, timeout_s, out=None):
    #     """See :meth:`.Camera._get_images_hw`."""
    #     # Similar to the core method but for a batch of images.
    #     # This should be used if the camera has a hardware-specific method of grabbing
    #     # frame batches. If not defined, the superclass captures and averages sequential
    #     # _get_image_hw images.
    #     return self.cam.get_images_function()     # TODO: Fill in proper function.

    ## Setting window of interest (WOI)

    # def _set_woi_hw(self, woi):
    #     """See :meth:`.Camera._set_woi_hw`."""
    #     pass # TODO: Fill in proper function. Do not fill in if the camera does not support WOI.

    # def _get_woi_hw(self):
    #     """See :meth:`.Camera._get_woi_hw`."""
    #     pass # TODO: Fill in proper function. Do not fill in if the camera does not support WOI.

    ## Setting binning

    # def _set_binning_hw(self, binning):
    #     """See :meth:`.Camera._set_binning_hw`."""
    #     pass # TODO: Fill in proper function. Do not fill in if the camera does not support binning.

    # def _get_binning_hw(self):
    #     """See :meth:`.Camera._get_binning_hw`."""
    #     pass # TODO: Fill in proper function. Do not fill in if the camera does not support binning.

    ## Clear the image buffer (superclass default method calls get_image successively)

    # def flush(self, timeout_s=1):
    #     """See :meth:`.Camera.flush`."""
    #     raise NotImplementedError()
    #     # Clears ungrabbed images from the queue; the abstract default calls .get_image twice.