"""
Hardware control for Santec SLMs via USB (cross-platform, no vendor DLLs required).

Provides :class:`SantecUSB`, a drop-in replacement for :class:`.Santec` that
communicates with Santec LCOS SLMs directly over USB using :class:`._santec_ftdi.SantecFTDI`
and PyD3XX. No ``SLMFunc.dll`` or Windows-only dependencies are required.

Note
~~~~
Install PyD3XX before use::

    pip install PyD3XX

PyD3XX bundles the FTDI D3XX shared libraries for Windows, Linux, and macOS.
No separate OS-level driver installation is required (Linux: add a udev rule for
the FT601 USB PID ``0x601f`` to allow non-root access).

Note
~~~~
:class:`SantecUSB` operates in USB/Memory mode only (``VI=0``). The SLM must
not be connected via DVI-D when using this driver. Phase patterns are
double-buffered across slots 1 and 2: each write uploads to the inactive slot
and then switches the display, so the active slot is never written while shown.

Note
~~~~
Santec provides base wavefront correction files accounting for the curvature of
the SLM surface. Load these via :meth:`.load_vendor_phase_correction`.
"""

import time
import warnings

import numpy as np

from ..slm import SLM
from ._santec_ftdi import SantecFTDI

try:
    import cv2
except ImportError:
    cv2 = None

# --- error bit definitions (from programmer's guide p.63 and _slm_win.py) ---

_DRIVEBOARD_ERROR: dict[int, str] = {
    0x01: "Startup error 1 (Drive board)",
    0x02: "Startup error 2 (Drive board)",
    0x04: "Video signal error (No signal)",
    0x08: "Drive board temperature error (70 deg C or higher)",
}

_OPTIONBOARD_ERROR: dict[int, str] = {
    0x01: "Startup error 1 (Option board)",
    0x02: "Startup error 2 (Option board)",
    0x04: "Voltage level error (DC 5.0V)",
    0x08: "Option board temperature error (70 deg C or higher)",
}

# maps firmware status strings to (code, name, note) tuples matching Santec._parse_status
_STATUS_MAP: dict[str, tuple[int, str, str]] = {
    "OK": (0, "SLM_OK", "All good!"),
    "BS": (2, "SLM_BS", "SLM is busy."),
    "NG": (-1, "SLM_NG", "Command not supported or error."),
    "NO RESPONSE": (-2, "SLM_NORESPONSE", "No response from firmware."),
}


class SantecUSB(SLM):
    """
    Interfaces with Santec SLMs via USB, without vendor DLLs.

    Drop-in replacement for :class:`.Santec`. Uses :class:`._santec_ftdi.SantecFTDI`
    for all device communication. Operates in USB/Memory mode (``VI=0``); phase
    patterns are double-buffered across slots 1 and 2 to avoid writing to the
    slot currently being displayed.

    Attributes
    ----------
    serial_number : str
        FTDI chip serial number used to identify the device.
    driveboard_id : str
        Drive board serial number string.
    optionboard_id : str
        Option board serial number string.
    """

    def __init__(
        self,
        serial_number: str,
        resolution: tuple[int, int] = (1920, 1200),
        bitdepth: int = 10,
        wav_um: float = 1,
        pitch_um: tuple[float, float] = (8, 8),
        verbose: bool = True,
        **kwargs,
    ) -> None:
        r"""
        Open a Santec SLM over USB and initialize phase calibration.

        Arguments
        ---------
        serial_number : str
            FTDI chip serial number. Use :meth:`info` to discover connected devices.
        resolution : (int, int)
            ``(width, height)`` of the SLM in pixels. Defaults to ``(1920, 1200)``,
            which is correct for SLM-200, SLM-210, and SLM-300.
        bitdepth : int
            Depth of SLM pixel well in bits. Defaults to 10.
        wav_um : float
            Wavelength of operation in microns. Defaults to 1 um.
        pitch_um : (float, float)
            Pixel pitch in microns. Defaults to 8 micron square pixels.
        verbose : bool
            Whether to print initialization progress.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.

        Note
        ----
        The phase table is reconfigured based on ``wav_design_um`` (defaults to
        ``wav_um``). This process takes roughly 40 seconds when a change is needed.
        If the resulting maximum phase deviates from 2pi by more than 2%, ``wav_design_um``
        is corrected automatically and a warning is printed.
        """
        self.serial_number = serial_number

        wav_design_um: float = kwargs.pop("wav_design_um", None)
        if wav_design_um is None:
            wav_design_um = wav_um

        if verbose:
            print("SantecUSB serial={} initializing... ".format(serial_number), end="")

        ftdi = SantecFTDI(serial_number)
        ftdi.open()
        self._ftdi = ftdi

        try:
            # wait for device to finish booting; any non-BS response means ready
            while ftdi.get_status() == "BS":
                pass

            self.get_error(raise_error=True)

            ftdi.set_video_mode(0)

            # prime slot 2 with zeros so the first _set_phase_hw can safely
            # write to the inactive slot 1 without touching the displayed slot
            width, height = resolution
            ftdi.upload_image(2, np.zeros((height, width), dtype=np.uint16))
            ftdi.display_slot(2)
            self._active_slot = 2

            if verbose:
                print("success")

            # configure phase table if needed
            current_nm, current_phase_pi = ftdi.get_wavelength()
            wav_desired_nm = int(wav_design_um * 1e3)

            attempt = 1
            while current_nm != wav_desired_nm and attempt <= 5:
                if verbose:
                    if attempt == 1:
                        print(
                            "Current phase table: wav={} nm, maxphase={:.2f}pi".format(
                                current_nm, current_phase_pi
                            )
                        )
                        print(
                            "Desired phase table: wav={} nm, maxphase=2.00pi".format(
                                wav_desired_nm
                            )
                        )
                    else:
                        print("(attempt {})".format(attempt))
                    print("     ...Updating phase table (this may take 40 seconds)...")

                ftdi.set_wavelength(wav_desired_nm, 2)
                ftdi.save_wavelength()
                # the official Santec sequence (manual p.7, p.9) does not include
                # ReadWL immediately after WriteWL + WriteAW; the FPGA does not
                # respond to ReadWL during the post-calibration settling period.
                # retry with fresh commands every 10s for up to 300s.
                _wl_read_ok = False
                for _retry in range(30):
                    try:
                        current_nm, current_phase_pi = ftdi.get_wavelength()
                        _wl_read_ok = True
                        break
                    except RuntimeError:
                        time.sleep(10)
                if not _wl_read_ok:
                    # calibration succeeded but ReadWL is still not ready;
                    # use nominal values and warn -- exact deviation on next init
                    current_nm = wav_desired_nm
                    current_phase_pi = 2.0
                    warnings.warn(
                        "Wavelength set to {} nm but ReadWL unavailable (FPGA settling "
                        "after calibration). Restart SantecUSB to obtain exact phase "
                        "deviation and correct wav_design_um.".format(wav_desired_nm)
                    )

                if verbose and _wl_read_ok:
                    print(
                        "Updated phase table: wav={} nm, maxphase={:.2f}pi".format(
                            current_nm, current_phase_pi
                        )
                    )

                # stop retrying if the wavelength did not change after the first attempt;
                # the firmware cannot recalibrate to this wavelength (hardware limitation)
                if attempt == 1 and current_nm != wav_desired_nm:
                    break

                attempt += 1

            if current_nm != wav_desired_nm or abs(current_phase_pi - 2.0) > 1.0:
                raise RuntimeError(
                    "Failed to update Santec phase table to {} nm "
                    "(current: {} nm, {:.2f}pi). "
                    "Check that wav_design_um matches the SLM's supported wavelength range.".format(
                        wav_desired_nm, current_nm, current_phase_pi
                    )
                )

            if verbose and abs(current_phase_pi - 2.0) > 0.04:
                wav_design_fixed_um = wav_design_um * (current_phase_pi / 2.0)
                print(
                    "  Warning: phase table maximum deviates >2% from 2pi ({:.2f}pi).".format(
                        current_phase_pi
                    )
                )
                print(
                    "    wav_design_um adjusted to {:.4f} um (was {:.4f} um).".format(
                        wav_design_fixed_um, wav_design_um
                    )
                )
                if wav_um / wav_design_fixed_um != 1:
                    print(
                        "    phase_scaling={:.4f} != 1; speed implications apply (see set_phase()).".format(
                            wav_um / wav_design_fixed_um
                        )
                    )
                wav_design_um = wav_design_fixed_um

            self.driveboard_id = ftdi.get_board_serial()
            self.optionboard_id = ftdi.get_option_board_serial()

            width, height = resolution
            super().__init__(
                (width, height),
                bitdepth=bitdepth,
                name=kwargs.pop("name", ftdi.get_firmware_serial()),
                wav_um=wav_um,
                wav_design_um=wav_design_um,
                pitch_um=pitch_um,
                **kwargs,
            )

            self.set_phase(None)

        except Exception as init_error:
            try:
                ftdi.close()
            except Exception as close_error:
                print(
                    "Could not close SantecUSB serial={} after init failure: {}".format(
                        serial_number, close_error
                    )
                )
            raise init_error

    # -------------------------------------------------------------------------
    # lifecycle
    # -------------------------------------------------------------------------

    @staticmethod
    def info(verbose: bool = True) -> list[str]:
        """
        List connected Santec SLM serial numbers.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered serials.

        Returns
        -------
        list of str
            FTDI serial number strings for each connected device.
        """
        serials = SantecFTDI.list_devices()
        if verbose:
            print("SantecUSB devices detected:")
            for s in serials:
                print("  {}".format(s))
        return serials

    def close(self) -> None:
        """Close the USB connection. See :meth:`.SLM.close`."""
        self._ftdi.close()

    # -------------------------------------------------------------------------
    # hardware write (abstract method implementation)
    # -------------------------------------------------------------------------

    def _set_phase_hw(self, display: np.ndarray) -> None:
        """
        Upload integer display data to the inactive slot and switch the display.

        Alternates between slots 1 and 2 so the active slot is never written
        while it is being displayed by the FPGA.

        See :meth:`.SLM._set_phase_hw` for the base class documentation.

        Parameters
        ----------
        display : numpy.ndarray
            Integer array of shape ``(height, width)`` to display on the SLM.
        """
        write_slot = 3 - self._active_slot
        self._ftdi.upload_image(write_slot, display)
        self._ftdi.display_slot(write_slot)
        self._active_slot = write_slot

    # -------------------------------------------------------------------------
    # diagnostics (matching Santec interface)
    # -------------------------------------------------------------------------

    def get_temperature(self) -> tuple[float, float]:
        """
        Read drive board and option board temperatures.

        Returns
        -------
        (float, float)
            Temperature in Celsius of the drive board and option board.
        """
        return self._ftdi.get_temperature()

    def get_error(self, raise_error: bool = True, return_codes: bool = False):
        """
        Read drive board and option board error flags.

        Parameters
        ----------
        raise_error : bool
            Raise a RuntimeError if any errors are present.
        return_codes : bool
            Return raw ``(drive_bits, option_bits)`` integers instead of a list of strings.

        Returns
        -------
        list of str or (int, int)
            Error message strings, or raw error code integers when ``return_codes=True``.
        """
        drive_bits, option_bits = self._ftdi.get_errors()
        errors = []
        for bit, msg in _DRIVEBOARD_ERROR.items():
            if drive_bits & bit:
                errors.append(msg)
        for bit, msg in _OPTIONBOARD_ERROR.items():
            if option_bits & bit:
                errors.append(msg)
        if errors:
            error_str = "Santec error: " + ", ".join("'" + e + "'" for e in errors)
            if raise_error:
                raise RuntimeError(error_str)
            warnings.warn(error_str)
        if return_codes:
            return (drive_bits, option_bits)
        return errors

    def get_status(self, raise_error: bool = True) -> tuple[int, str, str]:
        """
        Read and parse the FPGA status string.

        Parameters
        ----------
        raise_error : bool
            Raise a RuntimeError when status is not OK.

        Returns
        -------
        (int, str, str)
            Status in ``(code, name, note)`` form.
        """
        return SantecUSB._parse_status(self._ftdi.get_status(), raise_error)

    @staticmethod
    def _parse_status(response: str, raise_error: bool = True) -> tuple[int, str, str]:
        """
        Map a USB firmware status string to a ``(code, name, note)`` tuple.

        Matches the return signature of :meth:`.Santec._parse_status`.

        Parameters
        ----------
        response : str
            Firmware response: ``"OK"``, ``"BS"``, ``"NG"``, or ``"NO RESPONSE"``.
        raise_error : bool
            Raise a RuntimeError when status is not OK.

        Returns
        -------
        (int, str, str)
            Status tuple.
        """
        entry = _STATUS_MAP.get(response, (-99, "SLM_UNKNOWN", response))
        code, name, note = entry
        if code != 0:
            msg = "Santec status {}; '{}'".format(name, note)
            if raise_error:
                raise RuntimeError(msg)
            warnings.warn(msg)
        return (code, name, note)

    # -------------------------------------------------------------------------
    # vendor phase correction (identical to Santec.load_vendor_phase_correction)
    # -------------------------------------------------------------------------

    def load_vendor_phase_correction(
        self,
        file_path: str,
        smooth: bool = False,
        overwrite: bool = True,
    ) -> np.ndarray:
        """
        Load phase correction provided by Santec from a CSV file.

        Sets ``"phase"`` in :attr:`~slmsuite.hardware.slms.slm.SLM.source`.

        Parameters
        ----------
        file_path : str
            Path to the Santec-provided CSV correction file.
        smooth : bool
            Apply Gaussian blur to smooth the correction map.
            Requires ``cv2``; ignored with a warning if ``cv2`` is not installed.
        overwrite : bool
            Overwrite the existing ``source["phase"]``.

        Note
        ~~~~
        This correction is only fully valid at the wavelength at which it was collected.

        Returns
        -------
        numpy.ndarray
            Phase correction array in radians.
        """
        try:
            # skip first row (header) and first column (Y coordinates)
            phase_map = np.loadtxt(file_path, skiprows=1, dtype=int, delimiter=",")[
                :, 1:
            ]
            phase = (-2 * np.pi / self.bitresolution) * phase_map.astype(float)

            if smooth:
                if cv2 is None:
                    warnings.warn(
                        "cv2 not installed; skipping smoothing. "
                        "Install opencv-python to enable smooth=True."
                    )
                else:
                    size_blur = 15
                    real = cv2.GaussianBlur(np.cos(phase), (size_blur, size_blur), 0)
                    imag = cv2.GaussianBlur(np.sin(phase), (size_blur, size_blur), 0)
                    phase = np.arctan2(imag, real) + np.pi

            if overwrite:
                self.source["phase"] = phase

            return phase
        except Exception as e:
            warnings.warn("Error while loading phase correction.\n{}".format(e))
            return self.source["phase"]

    # -------------------------------------------------------------------------
    # trigger overrides (SLM base stubs → SantecFTDI)
    # -------------------------------------------------------------------------

    def set_input_trigger(self, on: bool = False) -> None:
        """
        Enable or disable the external trigger input.

        Parameters
        ----------
        on : bool
            ``True`` to enable.
        """
        self._ftdi.set_trigger_input(bool(on))

    def set_output_trigger(self, on: bool = False) -> None:
        """
        Enable or disable the trigger output signal.

        Parameters
        ----------
        on : bool
            ``True`` to enable.
        """
        self._ftdi.set_trigger_output(bool(on))
