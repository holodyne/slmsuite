"""
Hardware control for Santec SLMs.
Tested with Santec LCoS SLM-200, SLM-210, and SLM-300.

Note
~~~~
The ``"dll"`` backend requires dynamically linked libraries from Santec in the
runtime directory: ``SLMFunc.dll`` and ``FTD3XX.dll`` (Windows only).

The ``"usb"`` backend requires ``PyD3XX`` (cross-platform): ``pip install PyD3XX``.

Note
~~~~
Santec provides base wavefront correction files accounting for SLM surface
curvature. Load these via :meth:`.load_vendor_phase_correction`.
"""

import importlib
import time
from enum import IntEnum
from typing import ClassVar

import numpy as np

from slmsuite._logging import make_logger
from slmsuite.hardware.slms.slm import SLM
from slmsuite.misc.xp import as_numpy

from ._dll_driver import _SantecDLLDriver
from ._usb_driver import _SantecUSBDriver

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None

_PACKAGE = __package__

logger = make_logger(__name__)


class _BACKEND(IntEnum):
    """Backend selection for Santec SLMs."""

    NULL = 0
    DLL = 1
    USB = 2


class Santec(SLM):
    """
    Interfaces with Santec SLMs via DLL or USB backend.

    The ``"dll"`` backend requires ``SLMFunc.dll`` and ``FTD3XX.dll`` in the
    runtime directory (Windows only). The ``"usb"`` backend requires
    ``PyD3XX`` (cross-platform, install with ``pip install PyD3XX``).

    The ``"usb"`` backend operates in USB/Memory mode (``VI=0``). Phase patterns
    are double-buffered across slots 1 and 2 so the active slot is never written
    while being displayed. In the future, the DVI interface may be supported.

    Attributes
    ----------
    backend : _BACKEND
        Active backend enum value.
    driveboard_id : str
        Drive board serial number string.
    optionboard_id : str
        Option board serial number string.
    """

    backend: _BACKEND
    driveboard_id: str = ""
    optionboard_id: str = ""
    product_code_id: str = ""  # DLL backend only

    # keyed by _BACKEND; populated lazily by _load_lib()
    _lib: ClassVar[dict] = {}

    # single canonical error bit definitions
    _DRIVEBOARD_ERROR: ClassVar[dict[int, str]] = {
        0x01: "Startup error 1 (Drive board)",
        0x02: "Startup error 2 (Drive board)",
        0x04: "Video signal error (No signal)",
        0x08: "Drive board temperature error (70 deg C or higher)",
    }
    _OPTIONBOARD_ERROR: ClassVar[dict[int, str]] = {
        0x01: "Startup error 1 (Option board)",
        0x02: "Startup error 2 (Option board)",
        0x04: "Voltage level error (DC 5.0V)",
        0x08: "Option board temperature error (70 deg C or higher)",
    }

    @staticmethod
    def _load_lib(backend: str | _BACKEND, dll_path: str | None = None) -> _BACKEND:
        """
        Lazily load and cache the library for a backend.

        Idempotent: safe to call multiple times; returns immediately if already
        loaded.

        Parameters
        ----------
        backend : str OR _BACKEND
            ``"dll"`` or ``"usb"`` (case-insensitive), or a :class:`._BACKEND` value.
        dll_path : str OR None
            Reserved for future use; currently unused.

        Returns
        -------
        _BACKEND
            The resolved backend enum value.

        Raises
        ------
        ImportError
            If the required library (``SLMFunc.dll`` or ``PyD3XX``) is unavailable.
        RuntimeError
            If ``backend`` is unrecognized.
        """
        if isinstance(backend, str):
            try:
                be = _BACKEND[backend.upper()]
            except KeyError:
                raise RuntimeError(f"Unknown backend {backend!r}. Choose 'dll' or 'usb'.") from None
        else:
            be = _BACKEND(backend)
        if be in Santec._lib:
            return be
        if be == _BACKEND.DLL:
            try:
                slm_win = importlib.import_module("._slm_win", package=_PACKAGE)
                Santec._lib[_BACKEND.DLL] = slm_win
            except Exception as exc:
                raise ImportError(
                    "Santec DLL backend unavailable. Copy SLMFunc.dll and FTD3XX.dll "
                    f"to the runtime directory.\nOriginal error: {exc}"
                ) from exc
            return _BACKEND.DLL
        if be == _BACKEND.USB:
            try:
                import PyD3XX

                Santec._lib[_BACKEND.USB] = PyD3XX
            except (ImportError, OSError) as exc:
                raise ImportError(
                    "Santec USB backend unavailable. Install PyD3XX: "
                    f"pip install PyD3XX\nOriginal error: {exc}"
                ) from exc
            return _BACKEND.USB
        raise RuntimeError(f"Unknown backend {be!r}. Choose 'dll' or 'usb'.")

    @staticmethod
    def info(backend: str = "usb", verbose: bool = True) -> list:
        """
        Discover connected Santec SLMs.

        Parameters
        ----------
        backend : str
            ``"dll"`` lists Windows display numbers and names;
            ``"usb"`` lists FTDI serial number strings.
        verbose : bool
            Whether to print discovered devices.

        Returns
        -------
        list
            For ``"dll"``, a list of ``(display_number, display_name)`` tuples.
            For ``"usb"``, a list of FTDI serial number strings.

        Raises
        ------
        ImportError
            If the backend library is unavailable.
        """
        Santec._load_lib(backend)
        be = _BACKEND[backend.upper()]
        if be == _BACKEND.DLL:
            import ctypes

            _sf = Santec._lib[_BACKEND.DLL]
            display_list = []
            for display_number in range(1, 9):
                width = ctypes.c_ushort(0)
                height = ctypes.c_ushort(0)
                display_name = ctypes.create_string_buffer(128)
                status = int(_sf.SLM_Disp_Info2(display_number, width, height, display_name))
                if status not in (0, -1):
                    name, note = _sf.SLM_STATUS_DICT.get(status, ("SLM_UNKNOWN", ""))
                    raise RuntimeError(f"Santec error {name}; '{note}'")
                name = display_name.value.decode("mbcs")
                if len(name) > 0:
                    if verbose:
                        print(f"{display_number} ({name})")
                    display_list.append((display_number, name))
            return display_list
        if be == _BACKEND.USB:
            serials = _SantecUSBDriver.list_devices()
            if verbose:
                for s in serials:
                    print(s)
            return serials
        raise RuntimeError(f"Unknown backend {backend!r}.")

    def __init__(
        self,
        ftdi_serial: str | None = None,
        backend: str = "usb",
        slm_number: int = 1,
        display_number: int = 2,
        resolution: tuple[int, int] = (1920, 1200),
        bitdepth: int = 10,
        wav_um: float = 1,
        pitch_um: tuple[float, float] = (8, 8),
        **kwargs,
    ) -> None:
        r"""
        Open a Santec SLM and initialize the phase calibration table.

        Arguments
        ---------
        ftdi_serial : str or None
            FTDI chip serial number for ``backend="usb"``. Use :meth:`info` to
            list connected devices; this is the string it returns (e.g.
            ``"000000000001"``). Not to be confused with the SLM firmware serial
            returned by :meth:`get_firmware_serial`. Ignored for ``backend="dll"``.
        backend : str
            ``"dll"`` for the Windows DLL backend (DVI mode) or ``"usb"`` for the
            cross-platform USB backend (memory mode). Default ``"usb"``.
        slm_number : int
            USB control port number for ``backend="dll"``; ignored for USB.
            Default 1.
        display_number : int
            Windows display number for ``backend="dll"``; ignored for USB.
            Default 2.
        resolution : (int, int)
            ``(width, height)`` in pixels for the USB backend; ignored for DLL
            (auto-detected from ``SLM_Disp_Info2``). Default ``(1920, 1200)``
            for SLM-200, SLM-210, and SLM-300.
        bitdepth : int
            SLM pixel well depth in bits. Default 10.
        wav_um : float
            Wavelength of operation in microns. Default 1 um.
        pitch_um : (float, float)
            Pixel pitch in microns. Default ``(8, 8)``.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.

        Note
        ----
        The phase table is reconfigured based on ``wav_design_um`` (defaults to
        ``wav_um``). This process takes roughly 40 seconds when a wavelength change
        is needed. If the resulting maximum phase deviates from 2pi by more than 2%,
        ``wav_design_um`` is corrected automatically and a warning is logged.

        Caution
        ~~~~~~~
        Defaults to 8 um square pixels and 10-bit depth. Valid for SLM-200,
        SLM-210, and SLM-300; may differ for future models.
        """
        try:
            self.backend = _BACKEND[backend.upper()]
        except KeyError:
            raise RuntimeError(f"Unknown backend {backend!r}. Choose 'dll' or 'usb'.") from None

        if self.backend == _BACKEND.USB and ftdi_serial is None:
            raise ValueError(
                "ftdi_serial is required for backend='usb'. "
                "Use Santec.info() to list connected devices."
            )

        Santec._load_lib(self.backend)

        wav_design_um: float = kwargs.pop("wav_design_um", None)
        if wav_design_um is None:
            wav_design_um = wav_um

        _id_label = ftdi_serial if self.backend == _BACKEND.USB else slm_number
        logger.debug("Santec (%s) %s initializing...", backend, _id_label)

        if self.backend == _BACKEND.DLL:
            _sf = Santec._lib[_BACKEND.DLL]
            self._driver = _SantecDLLDriver(slm_number, display_number, _sf)
        else:
            self._driver = _SantecUSBDriver(ftdi_serial, resolution)

        self._driver.open()

        try:
            self.get_error(raise_error=True)

            if self.backend == _BACKEND.USB:
                self._driver.prime()

            # --- wavelength calibration (shared between backends) ---
            current_nm, current_phase_pi = self._driver.get_wavelength()
            wav_desired_nm = int(wav_design_um * 1e3)

            attempt = 1
            while current_nm != wav_desired_nm and attempt <= 5:
                if attempt == 1:
                    logger.debug(
                        "Current phase table: wav=%d nm, maxphase=%.2fpi",
                        current_nm,
                        current_phase_pi,
                    )
                    logger.debug("Desired phase table: wav=%d nm, maxphase=2.00pi", wav_desired_nm)
                else:
                    logger.debug("(attempt %d)", attempt)
                logger.debug("...Updating phase table (this may take 40 seconds)...")

                self._driver.set_wavelength(wav_desired_nm, 2.0)
                self._driver.save_wavelength()

                if self.backend == _BACKEND.USB:
                    # FPGA may not respond to ReadWL for up to 300 s after calibration
                    _wl_read_ok = False
                    for _retry in range(30):
                        try:
                            current_nm, current_phase_pi = self._driver.get_wavelength()
                            _wl_read_ok = True
                            break
                        except RuntimeError:
                            time.sleep(10)
                    if not _wl_read_ok:
                        current_nm = wav_desired_nm
                        current_phase_pi = 2.0
                        logger.warning(
                            "Wavelength set to %d nm but ReadWL unavailable (FPGA settling "
                            "after calibration). Restart Santec to obtain exact phase "
                            "deviation and correct wav_design_um.",
                            wav_desired_nm,
                        )
                else:
                    current_nm, current_phase_pi = self._driver.get_wavelength()
                    logger.debug(
                        "Updated phase table: wav=%d nm, maxphase=%.2fpi",
                        current_nm,
                        current_phase_pi,
                    )

                # stop retrying if firmware rejected the first calibration attempt
                if attempt == 1 and current_nm != wav_desired_nm:
                    break

                attempt += 1

            if current_nm != wav_desired_nm or abs(current_phase_pi - 2.0) > 1.0:
                raise RuntimeError(
                    f"Failed to update Santec phase table to {wav_desired_nm} nm "
                    f"(current: {current_nm} nm, {current_phase_pi:.2f}pi). "
                    "Check that wav_design_um matches the SLM's supported wavelength "
                    "range."
                )

            if abs(current_phase_pi - 2.0) > 0.04:
                wav_design_fixed_um = wav_design_um * (current_phase_pi / 2.0)
                logger.warning(
                    "Phase table maximum deviates >2%% from 2pi (%.2fpi). "
                    "wav_design_um adjusted to %.4f um (was %.4f um).",
                    current_phase_pi,
                    wav_design_fixed_um,
                    wav_design_um,
                )
                if wav_um / wav_design_fixed_um != 1:
                    logger.warning(
                        "phase_scaling=%.4f != 1; speed implications apply (see set_phase()).",
                        wav_um / wav_design_fixed_um,
                    )
                wav_design_um = wav_design_fixed_um

            # --- backend-specific post-calibration setup ---
            if self.backend == _BACKEND.DLL:
                logger.debug("Looking for display_number=%s...", display_number)
                width, height = self._driver.get_display_dims()
                self.product_code_id = self._driver.product_code_id
                self._driver.get_board_serials()
                logger.debug("Opening display %s...", self._driver.get_firmware_serial())
                self._driver.open_display()
            else:
                width, height = resolution

            self.driveboard_id = self._driver.get_board_serial()
            self.optionboard_id = self._driver.get_option_board_serial()

            super().__init__(
                (width, height),
                bitdepth=bitdepth,
                name=kwargs.pop("name", self._driver.get_firmware_serial()),
                wav_um=wav_um,
                wav_design_um=wav_design_um,
                pitch_um=pitch_um,
                **kwargs,
            )

            self.set_phase(None)

        except Exception as init_error:
            try:
                self._driver.close()
            except Exception as close_error:
                logger.warning(
                    "Could not close Santec %s after init failure: %s", _id_label, close_error
                )
            raise init_error

    def close(self) -> None:
        """See :meth:`.SLM.close`."""
        self._driver.close()

    def _set_phase_hw(self, display: np.ndarray | int, index: int | None = None) -> None:
        """
        Hardware-specific phase write.

        See :meth:`.SLM._set_phase_hw` for the base class documentation.

        Parameters
        ----------
        display : numpy.ndarray or int
            - ``ndarray, index=None``: write to back buffer and swap (default path).
            - ``ndarray, index=i``: write to zero-indexed slot ``i`` without
              switching the display; USB backend only.
            - ``int``: switch the displayed slot to zero-indexed slot ``display``;
              USB backend only.
        index : int or None
            Slot override for ndarray writes. ``None`` uses the double-buffer path.

        Raises
        ------
        NotImplementedError
            If ``display`` is an int or ``index`` is not ``None`` on the DLL
            backend (DVI mode has no slot concept).
        """
        if not isinstance(display, (int, np.integer)):
            display = as_numpy(display)  # the drivers need host memory
        self._driver.write_frame(display, index)

    def get_temperature(self) -> tuple[float, float]:
        """
        Read drive and option board temperatures.

        Returns
        -------
        (float, float)
            ``(drive_temp_C, option_temp_C)``.
        """
        return self._driver.get_temperature()

    def get_error(self, raise_error: bool = True, return_codes: bool = False):
        """
        Read drive and option board error flags.

        Parameters
        ----------
        raise_error : bool
            Raise ``RuntimeError`` if any errors are present.
        return_codes : bool
            Return raw ``(drive_bits, option_bits)`` instead of a list of strings.

        Returns
        -------
        list of str or (int, int)
            Error message strings, or raw integer error codes when
            ``return_codes=True``.
        """
        drive_bits, option_bits = self._driver.get_errors()
        errors = []
        for bit, msg in Santec._DRIVEBOARD_ERROR.items():
            if drive_bits & bit:
                errors.append(msg)
        for bit, msg in Santec._OPTIONBOARD_ERROR.items():
            if option_bits & bit:
                errors.append(msg)
        if errors:
            error_str = "Santec error: " + ", ".join("'" + e + "'" for e in errors)
            if raise_error:
                raise RuntimeError(error_str)
            self.logger.warning(error_str)
        if return_codes:
            return (drive_bits, option_bits)
        return errors

    def get_status(self, raise_error: bool = True) -> tuple[int, str, str]:
        """
        Read and parse current device status.

        Parameters
        ----------
        raise_error : bool
            Raise ``RuntimeError`` when status is not OK.

        Returns
        -------
        (int, str, str)
            Status in ``(code, name, note)`` form.
        """
        code, name, note = self._driver.get_status()
        if code != 0:
            msg = f"Santec status {name}; '{note}'"
            if raise_error:
                raise RuntimeError(msg)
            self.logger.warning(msg)
        return (code, name, note)

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
            Apply Gaussian blur to smooth the correction map. Requires ``cv2``;
            skipped with a warning if unavailable.
        overwrite : bool
            Overwrite the existing ``source["phase"]``.

        Note
        ~~~~
        This correction is only fully valid at the wavelength at which it was
        collected.

        Returns
        -------
        numpy.ndarray
            Phase correction array in radians.
        """
        try:
            phase_map = np.loadtxt(file_path, skiprows=1, dtype=int, delimiter=",")[:, 1:]
            if phase_map.shape != tuple(self.shape):
                raise ValueError(
                    f"Correction shape {phase_map.shape} does not match SLM shape {self.shape}."
                )
            phase = (-2 * np.pi / self.bitresolution) * phase_map.astype(float)
            if smooth:
                if _cv2 is None:
                    self.logger.warning(
                        "cv2 not installed; skipping smoothing. Install opencv-python to enable smooth=True."
                    )
                else:
                    size_blur = 15
                    real = _cv2.GaussianBlur(np.cos(phase), (size_blur, size_blur), 0)
                    imag = _cv2.GaussianBlur(np.sin(phase), (size_blur, size_blur), 0)
                    phase = np.arctan2(imag, real) + np.pi
            if overwrite:
                self.source["phase"] = phase
            return phase
        except Exception as e:
            self.logger.warning("Error while loading phase correction.\n%s", e)
            return self.source.get("phase")

    def set_input_trigger(self, on: bool = False) -> None:
        """
        Enable or disable the external trigger input.

        Parameters
        ----------
        on : bool
            ``True`` to enable.

        Raises
        ------
        NotImplementedError
            If not using the USB backend.
        """
        if self.backend != _BACKEND.USB:
            raise NotImplementedError(
                "Input trigger control is not implemented for the DLL backend."
            )
        self._driver.set_trigger_input(bool(on))

    def set_output_trigger(self, on: bool = False) -> None:
        """
        Enable or disable the trigger output signal.

        Parameters
        ----------
        on : bool
            ``True`` to enable.

        Raises
        ------
        NotImplementedError
            If not using the USB backend.
        """
        if self.backend != _BACKEND.USB:
            raise NotImplementedError(
                "Output trigger control is not implemented for the DLL backend."
            )
        self._driver.set_trigger_output(bool(on))

    def load_csv(self, filename: str) -> None:
        """
        Write the phase image contained in a CSV file to the SLM.

        Parameters
        ----------
        filename : str
            Path to the CSV file.

        Raises
        ------
        NotImplementedError
            If not using the DLL backend.
        """
        if self.backend != _BACKEND.DLL:
            raise NotImplementedError("load_csv() requires the DLL backend.")
        self._driver.load_csv(filename)
