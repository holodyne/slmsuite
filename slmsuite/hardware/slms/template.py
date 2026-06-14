"""
Template for writing a subclass for SLM hardware control in :mod:`slmsuite`.
Outlines which SLM superclass functions must be implemented.
"""
from slmsuite.hardware.slms.slm import SLM
from slmsuite._logging import make_logger

logger = make_logger(__name__)

class Template(SLM):
    """
    Template for implementing a new SLM subclass. Replace :class:`Template`
    with the desired subclass name. :class:`~slmsuite.hardware.slms.slm.SLM` is the
    superclass that sets the requirements for :class:`Template`.
    """

    def __init__(
        self,
        bitdepth=8,         # TODO: Remove these arguments if the SLM SDK
        wav_um=1,           #       has some function to read them from the SLM.
        pitch_um=(8,8),     #       Otherwise, the user must supply them as they
        **kwargs            #       are critical for transformations and calibrations.
    ):
        r"""
        Initialize SLM and attributes.

        Parameters
        ----------
        bitdepth : int
            Depth of SLM pixel well in bits. Defaults to 8.
        wav_um : float
            Wavelength of operation in microns. Defaults to 1 μm.
        pitch_um : (float, float)
            Pixel pitch in microns. Defaults to 8 micron square pixels.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.

        Note
        ~~~~
        These arguments, which ultimately are used to instantiate the :class:`.SLM` superclass,
        may be more accurately filled by calling the SLM's SDK functions.
        See the other implemented SLM subclasses for examples.
        """

        # TODO: Insert code here to initialize the SLM hardware, load properties, etc.

        # Mandatory functions:
        # - Opening a connection to the device.

        sdk = TODO()

        # Other possibilities to consider:
        # - Setting the SLM's operating wavelength (wav_um).
        # - Updating the SLM's phase table if necessary, and/or setting the design
        #   wavelength (wav_design_um).
        # - Setting the SLM's default settle time (abstract class SLM uses
        #   settle_time_s=0.3 seconds). This is important for experimental feedback to
        #   allow the SLM to settle before viewing the result on a camera.
        # - Checking for and saving the SLM parameters (height, width, etc).

        # Instantiate the superclass
        super().__init__(
            (sdk.width(), sdk.height()),
            bitdepth=bitdepth,
            wav_um=wav_um,
            pitch_um=pitch_um,
            **kwargs
        )

        # Zero the display using the superclass `set_phase()` function.
        self.set_phase(None)

    def close(self):
        """Close the SLM and delete related objects."""
        raise NotImplementedError()

    @staticmethod
    def info(verbose=True):
        """
        Discovers all SLMs detected by an SDK.
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
        serial_list = get_serial_list()     # TODO: Fill in proper function.
        return serial_list

    def _set_phase_hw(
            self,
            display,
            # execute: bool = True,    # TODO: Implement if supported.
            # block: bool = True,      # TODO: Implement if supported.
            # Add other keyword arguments if needed;
            # these are passed directly from set_phase(**kwargs)
        ):
        """
        Hardware-specific implementation.

        See :meth:`SLM._set_phase_hw` for the base class documentation.

        Parameters
        ----------
        display
            Integer data to display on the SLM. See :meth:`.SLM._set_phase_hw`.
        """
        # TODO: Insert code here to write raw phase data to the SLM.
        raise NotImplementedError()

    # def _format_phase_hw(self, phase):
    #     """
    #     Optional override to format phase data for the SLM. The base class default
    #     performs grayscale conversion via :meth:`._phase2gray`. Only override if
    #     your SLM requires custom formatting (e.g. electrode bitmaps with a PLM).
    #     See :meth:`.SLM._format_phase_hw` for further detail.
    #
    #     Parameters
    #     ----------
    #     phase : numpy.ndarray
    #         Array containing phase data.
    #
    #     Returns
    #     -------
    #     numpy.ndarray
    #         Processed bitmap data for the SLM device.
    #     """
    #     # TODO (If required for SLM): Insert code here to convert phase data to SLM data.

    # Triggering (implement if supported):

    # def set_input_trigger(self, on : bool = False):
    #     r"""
    #     Configures the input trigger of the SLM, where an external electronic signal can
    #     synchronize the time at which the SLM updates its display.

    #     Parameters
    #     ----------
    #     on : bool
    #         Subclasses *must* support a boolean configuration argument, but can
    #         also accept other datatypes or parameters as needed.
    #     """
    #     raise NotImplementedError("This SLM does not support input triggering.")

    # def set_output_trigger(self, on : bool = False):
    #     r"""
    #     Configures the output trigger of the SLM, where the SLM can send an electronic
    #     signal upon updating its display.

    #     Parameters
    #     ----------
    #     on : bool
    #         Subclasses *must* support a boolean configuration argument, but can
    #         also accept other datatypes or parameters as needed.
    #     """
    #     raise NotImplementedError("This SLM does not support output triggering.")
