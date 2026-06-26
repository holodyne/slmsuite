"""DLL backend adapter for Santec SLMs (wraps _slm_win ctypes module)."""
import ctypes
import warnings

import numpy as np


class _SantecDLLDriver:
    """
    Private DLL backend driver.

    Wraps the ``_slm_win`` ctypes bindings with the common driver interface.
    The ``_slm_win`` module is received as a constructor argument so this file
    never imports it at module level, keeping the DLL load deferred.

    Attributes
    ----------
    product_code_id : str
        Product code string from ``SLM_Disp_Info2`` (populated by
        :meth:`get_display_dims`).
    """

    def __init__(self, slm_number: int, display_number: int, slm_win) -> None:
        """
        Initialize the driver (no DLL calls yet).

        Args:
            slm_number: USB control port number used by the Santec SDK.
            display_number: Windows display number used by the Santec SDK.
            slm_win: Imported ``_slm_win`` module providing ctypes bindings.
        """
        self._sf = slm_win
        self._slm_number = slm_number
        self._display_number = display_number
        self._driveboard_id = ""
        self._optionboard_id = ""
        self._firmware_serial = ""
        self.product_code_id = ""

    def _check(self, status: int, raise_error: bool = True) -> tuple[int, str, str]:
        """
        Parse an SLM_STATUS return code and raise or warn on error.

        Args:
            status: Integer status code from a DLL call.
            raise_error: Raise RuntimeError on non-zero status when True.

        Returns:
            (code, name, note) tuple.

        Raises:
            ValueError: If status code is not in SLM_STATUS_DICT.
            RuntimeError: If status is non-zero and raise_error is True.
        """
        status = int(status)
        if status not in self._sf.SLM_STATUS_DICT:
            raise ValueError("SLM status '{}' not recognized.".format(status))
        name, note = self._sf.SLM_STATUS_DICT[status]
        if status != 0:
            msg = "Santec error {}; '{}'".format(name, note)
            if raise_error:
                raise RuntimeError(msg)
            warnings.warn(msg)
        return (status, name, note)

    def open(self) -> None:
        """
        Open USB control, wait until the device is ready, and set DVI video mode.

        Raises:
            RuntimeError: On any non-OK, non-busy status from the device.
        """
        self._check(self._sf.SLM_Ctrl_Open(self._slm_number))
        while True:
            status = self._sf.SLM_Ctrl_ReadSU(self._slm_number)
            if status == 0:
                break
            elif status == 2:
                continue
            else:
                self._check(status)
        self._check(self._sf.SLM_Ctrl_WriteVI(self._slm_number, 1))

    def open_display(self) -> None:
        """
        Open the SLM display window.

        Raises:
            RuntimeError: On DLL error.
        """
        self._check(self._sf.SLM_Disp_Open(self._display_number))

    def close(self) -> None:
        """Close the display window and USB control channel."""
        self._sf.SLM_Disp_Close(self._display_number)
        self._sf.SLM_Ctrl_Close(self._slm_number)

    def get_display_dims(self) -> tuple[int, int]:
        """
        Query display dimensions and cache display metadata.

        Populates :attr:`product_code_id` and the firmware serial used by
        :meth:`get_firmware_serial`.

        Returns:
            (width, height) in pixels.

        Raises:
            ValueError: If the display number does not correspond to an LCOS-SLM.
            RuntimeError: On DLL error.
        """
        width = ctypes.c_ushort(0)
        height = ctypes.c_ushort(0)
        display_name = ctypes.create_string_buffer(128)
        self._check(self._sf.SLM_Disp_Info2(self._display_number, width, height, display_name))
        name = display_name.value.decode("mbcs")
        names = name.split(",")
        if names[0] != "LCOS-SLM":
            raise ValueError(
                "SLM not found at display_number={}. "
                "Use Santec.info(backend='dll') to list available displays.".format(
                    self._display_number
                )
            )
        self.product_code_id = names[2]
        self._firmware_serial = names[-1]
        return (int(width.value), int(height.value))

    def get_board_serials(self) -> None:
        """
        Query and cache drive and option board serial numbers.

        Raises:
            RuntimeError: On DLL error.
        """
        driveboard_buf = ctypes.create_string_buffer(16)
        optionboard_buf = ctypes.create_string_buffer(16)
        self._check(
            self._sf.SLM_Ctrl_ReadSDO(self._slm_number, driveboard_buf, optionboard_buf)
        )
        self._driveboard_id = driveboard_buf.value.decode("mbcs")
        self._optionboard_id = optionboard_buf.value.decode("mbcs")

    def write_frame(self, frame: np.ndarray | int, index: int | None = None) -> None:
        """
        Write a frame to the SLM display (DVI mode only).

        Args:
            frame: ndarray to display. Passing an int (slot switch) or a non-None
                index (slot-addressed write) raises NotImplementedError because DVI
                mode has no slot concept.
            index: Must be None for DVI mode.

        Raises:
            NotImplementedError: If frame is an int or index is not None.
        """
        if isinstance(frame, (int, np.integer)):
            raise NotImplementedError(
                "Slot switching is not supported for the DLL backend (DVI mode only)."
            )
        if index is not None:
            raise NotImplementedError(
                "Slot-addressed writes are not supported for the DLL backend (DVI mode only)."
            )
        matrix = frame.astype(self._sf.USHORT)
        n_h, n_w = frame.shape
        c = matrix.ctypes.data_as(ctypes.POINTER((self._sf.USHORT * n_h) * n_w)).contents
        self._check(
            self._sf.SLM_Disp_Data(self._display_number, n_w, n_h, 0, c), raise_error=False
        )

    def get_wavelength(self) -> tuple[int, float]:
        """
        Read the current phase table wavelength and maximum phase.

        The DLL encodes max phase as ``int(phase_rad * 100 / pi)``; this method
        converts to the ``(nm, phase_pi)`` convention shared with the USB driver.

        Returns:
            (wav_nm, phase_pi) where phase_pi is in units of pi.

        Raises:
            RuntimeError: On DLL error.
        """
        wav_nm = ctypes.c_uint32(0)
        phase_val = ctypes.c_ulong(0)
        self._check(self._sf.SLM_Ctrl_ReadWL(self._slm_number, wav_nm, phase_val))
        return (int(wav_nm.value), float(phase_val.value) / 100.0)

    def set_wavelength(self, wav_nm: int, max_phase_pi: float = 2.0) -> None:
        """
        Update the phase calibration wavelength.

        Args:
            wav_nm: Wavelength in nanometres.
            max_phase_pi: Maximum phase in units of pi; converted to
                ``int(round(max_phase_pi * 100))`` for the DLL call.

        Raises:
            RuntimeError: On DLL error.
        """
        phase_val = int(round(max_phase_pi * 100))
        self._check(
            self._sf.SLM_Ctrl_WriteWL(
                self._slm_number, ctypes.c_uint32(wav_nm), phase_val
            )
        )

    def save_wavelength(self) -> None:
        """
        Persist the current phase table to EEPROM.

        Raises:
            RuntimeError: On DLL error.
        """
        self._check(self._sf.SLM_Ctrl_WriteAW(self._slm_number))

    def get_temperature(self) -> tuple[float, float]:
        """
        Read drive and option board temperatures.

        Returns:
            (drive_temp_C, option_temp_C).

        Raises:
            RuntimeError: On DLL error.
        """
        drive_temp = ctypes.c_uint32(0)
        option_temp = ctypes.c_uint32(0)
        self._check(self._sf.SLM_Ctrl_ReadT(self._slm_number, drive_temp, option_temp))
        return (drive_temp.value / 10.0, option_temp.value / 10.0)

    def get_errors(self) -> tuple[int, int]:
        """
        Read drive and option board error bitfields.

        Returns:
            (drive_error_bits, option_error_bits) as raw integers.

        Raises:
            RuntimeError: On DLL error.
        """
        drive_error = ctypes.c_uint32(0)
        option_error = ctypes.c_uint32(0)
        self._check(
            self._sf.SLM_Ctrl_ReadEDO(self._slm_number, drive_error, option_error)
        )
        return (int(drive_error.value), int(option_error.value))

    def get_status(self) -> tuple[int, str, str]:
        """
        Read and decode current device status.

        Returns:
            (code, name, note) tuple.
        """
        status = int(self._sf.SLM_Ctrl_ReadSU(self._slm_number))
        if status not in self._sf.SLM_STATUS_DICT:
            return (status, "SLM_UNKNOWN", "Unknown status code {}.".format(status))
        name, note = self._sf.SLM_STATUS_DICT[status]
        return (status, name, note)

    def get_board_serial(self) -> str:
        """
        Return the cached drive board serial number.

        Returns:
            Drive board serial string. Call :meth:`get_board_serials` first.
        """
        return self._driveboard_id

    def get_option_board_serial(self) -> str:
        """
        Return the cached option board serial number.

        Returns:
            Option board serial string. Call :meth:`get_board_serials` first.
        """
        return self._optionboard_id

    def get_firmware_serial(self) -> str:
        """
        Return the cached firmware version string (SerialNumberID from SLM_Disp_Info2).

        Returns:
            Firmware serial string, e.g. ``"2018021001"``. Call
            :meth:`get_display_dims` first.
        """
        return self._firmware_serial

    def load_csv(self, filename: str) -> None:
        """
        Write the phase image from a CSV file to the display.

        Args:
            filename: Path to the CSV file.

        Raises:
            RuntimeError: On DLL error.
        """
        self._check(self._sf.SLM_Disp_ReadCSV(self._display_number, 0, filename))
