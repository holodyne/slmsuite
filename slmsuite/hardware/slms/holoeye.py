"""
Hardware control for Holoeye SLMs.
Created for SLM Display SDK (Python) v4.0.0.
Tested with Holoeye SLM ERIS-NIR-153.

Important
~~~~~~~~~
Check that the SLM Display SDK is in the default folder
``C:\\Program Files\\HOLOEYE Photonics\\SLM Display SDK`` (Python) v4.0.0
or otherwise add the installation folder to your python path.
"""
import warnings
from slmsuite.hardware.slms.slm import SLM
from slmsuite.misc.xp import as_numpy
from slmsuite._logging import make_logger

logger = make_logger(__name__)

# Set the path for the SLM Display SDK
import os
import sys
try:
    env_path = os.getenv("HEDS_4_0_PYTHON")
    if env_path is None or not os.path.isdir(env_path):
        env_path = os.path.abspath("../..")
    importpath_api =  os.path.join(env_path, "api", "python")
    importpath_HEDS =  os.path.join(env_path, "examples")
    sys.path.append(importpath_api)
    sys.path.append(importpath_HEDS)
except Exception:
    pass

# Load Holoeye's SDK module.
try:
    import HEDS
    # from hedslib.heds_types import *
    from hedslib import heds_types
except ImportError:
    HEDS = None
    warnings.warn("Holoeye SDK HEDS not installed. Install to use Holoeye SLMs.")

class Holoeye(SLM):
    """
    Interfaces with Holoeye SLMs via the ``HEDS`` library.

    Attributes
    ----------
    slm_lib : HEDS.SLM
        Handle to the SLM.
    preselect : str
        Preselect string for the SLM. Used to identify the SLM.
    """

    def __init__(
        self,
        preselect=None,
        wav_um=1,
        verbose=True,
        **kwargs
    ):
        r"""
        Initializes an instance of a Holoeye SLM.

        Parameters
        ----------
        preselect : str
            Preselect string for the SLM. Examples:

            - ``"index:0"`` select first SLM available in the system.
            - ``"name:pluto;serial:0001"`` select a PLUTO SLM with the serial number 0001.
            - ``"name:luna"`` select a LUNA SLM.
            - ``"serial:2220-0011"`` select a LUNA/2220 SLM just by passing the serial number.
            - ``"connect://ipv4:port"``, e.g. ``"connect://127.0.0.1:6230"``, connect to
              a manually started process.

        wav_um : float
            Wavelength of operation in microns. Defaults to 1 μm.
        verbose : bool
            Whether to print extra information.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.

        Important
        ~~~~~~~~~
        The Holoeye SLM Display SDK must be installed and the path to the SDK must be added to the python path.
        See :mod:`~slmsuite.hardware.slms.holoeye` for instructions on how to do this.
        """
        if HEDS is None:
            raise ImportError("SDK HEDS not installed. Install to use Holoeye SLMs.")

        # Initialize the SDK and check that version 4.0 of the SDK is being used.
        error = HEDS.SDK.Init(4,0)
        self._handle_error(error)

        # Connect and open the SLM
        if verbose: print("Opening SLM ...", end="")
        self.preselect = preselect
        self.slm_lib = HEDS.SLM.Init(preselect=self.preselect)
        self._handle_error(self.slm_lib.errorCode())
        if verbose: print("success")

        # Set the SLM's operating wavelength (wav_um) in nm.
        error = self.slm_lib.setWavelength(wav_um * 1000)
        self._handle_error(error)

        # Check the SLM's parameters.
        pitch_um = (self.slm_lib.pixelsize_um(), self.slm_lib.pixelsize_um())
        width = self.slm_lib.width_px()
        height = self.slm_lib.height_px()

        # Instantiate the superclass
        super().__init__(
            (width, height),
            bitdepth=8,
            wav_um=wav_um,
            pitch_um=pitch_um,
            **kwargs
        )

        # Zero the display using the superclass `set_phase()` function.
        self.set_phase(None)

    def _handle_error(self, error):
        """
        Handles errors from the Holoeye SDK.
        Raises an exception if the error code is not HEDSERR_NoError.

        Parameters
        ----------
        error : int
            Error code returned by the Holoeye SDK.

        Raises
        ------
        RuntimeError
            If the error code is not HEDSERR_NoError.
        """
        if error != heds_types.HEDSERR_NoError:
            raise RuntimeError(HEDS.SDK.ErrorString(error))

    @staticmethod
    def info(verbose=True):
        """
        Discovers all SLMs detected by an SDK.
        Useful for a user to identify the correct serial numbers / etc.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("This functionality is not supported by Holoeye. Use the EDID device detection GUI instead.")

    def close(self):
        """
        See :meth:`.SLM.close`.
        """
        error = self.slm_lib.window().close()
        self._handle_error(error)

    def _set_phase_hw(self, display):
        """
        Hardware-specific implementation for Holoeye SLM devices.

        See :meth:`SLM._set_phase_hw` for the base class documentation.

        Parameters
        ----------
        display
            Integer data to display on the SLM. See :meth:`.SLM._set_phase_hw`.
        """
        display = as_numpy(display)   # The driver needs host memory.

        error = self.slm_lib.showPhaseData(display, phase_unit=256)
        self._handle_error(error)

    def load_vendor_phase_correction(self, file_path):
        """
        Load phase correction provided by Holoeye from file,
        directly into the SLM SDK.
        Note that this bypasses the slmsuite standard of setting
        ``"phase"`` in :attr:`~slmsuite.hardware.slms.slm.SLM.source`.

        Parameters
        ----------
        file_path : str
            File path for the vendor-provided phase correction.
        """
        # Enable wavefront compensation visualization in SLM preview window and stay with SLM preview scale "Fit"
        error = self.slm_lib.preview().setSettings(
            flags=heds_types.HEDSSLMPF_ShowWavefrontCompensation,
            zoom=0.0
        )
        self._handle_error(error)

        # Load the wavefront compensation file
        error = self.slm_lib.window().loadWavefrontCompensationFile(str(file_path))
        self._handle_error(error)
