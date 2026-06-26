"""USB backend adapter for Santec SLMs (wraps SantecFTDI)."""
import numpy as np

from ._santec_ftdi import SantecFTDI

# maps FPGA status strings to (code, name, note) matching the DLL SLM_STATUS_DICT convention
_STATUS_MAP: dict[str, tuple[int, str, str]] = {
    "OK": (0, "SLM_OK", "All good!"),
    "BS": (2, "SLM_BS", "SLM is busy."),
    "NG": (-1, "SLM_NG", "Command not supported or error."),
    "NO RESPONSE": (-2, "SLM_NORESPONSE", "No response from firmware."),
}


class _SantecUSBDriver:
    """
    Private USB backend driver.

    Wraps :class:`._santec_ftdi.SantecFTDI` with the common driver interface and
    manages the double-buffer slot state for USB/Memory mode writes.

    Slots exposed by SantecFTDI are 1-indexed; this class converts from the
    zero-indexed slot convention used by the public API.
    """

    def __init__(self, serial_number: str, resolution: tuple[int, int]) -> None:
        """
        Initialize (does not open the device yet; call :meth:`open`).

        Args:
            serial_number: FTDI chip serial number.
            resolution: (width, height) of the SLM in pixels.
        """
        self._serial_number = serial_number
        self._resolution = resolution
        self._ftdi: SantecFTDI | None = None
        self._active_slot = 2

    def open(self) -> None:
        """
        Open the FTDI device and wait until the FPGA is no longer busy.

        Raises:
            RuntimeError: If PyD3XX is not installed or the device is not found.
        """
        self._ftdi = SantecFTDI(self._serial_number)
        self._ftdi.open()
        while self._ftdi.get_status() == "BS":
            pass

    def prime(self) -> None:
        """
        Set USB/Memory video mode, upload a zero frame to slot 2, and display it.

        Establishes the initial double-buffer state: ``_active_slot = 2``, so the
        first :meth:`write_frame` call will write to slot 1.
        """
        self._ftdi.set_video_mode(0)
        w, h = self._resolution
        self._ftdi.upload_image(2, np.zeros((h, w), dtype=np.uint16))
        self._ftdi.display_slot(2)
        self._active_slot = 2

    def close(self) -> None:
        """Close the USB connection if it is open."""
        if self._ftdi is not None:
            self._ftdi.close()

    def write_frame(self, frame: np.ndarray | int, index: int | None = None) -> None:
        """
        Write or switch a frame.

        Args:
            frame: ndarray to upload, or int to switch the displayed slot
                (zero-indexed externally; SantecFTDI uses 1-indexed slots internally).
            index: Target slot for ndarray writes (zero-indexed). None uses the
                double-buffer path (writes to inactive slot and switches display).
                A specific index writes to that slot without switching the display.
        """
        if isinstance(frame, (int, np.integer)):
            self._ftdi.display_slot(int(frame) + 1)
            self._active_slot = int(frame) + 1
            return
        if index is None:
            write_slot = 3 - self._active_slot
            self._ftdi.upload_image(write_slot, frame)
            self._ftdi.display_slot(write_slot)
            self._active_slot = write_slot
        else:
            self._ftdi.upload_image(index + 1, frame)

    def get_wavelength(self) -> tuple[int, float]:
        """
        Read the current phase table wavelength and maximum phase.

        Returns:
            (wav_nm, phase_pi) where phase_pi is in units of pi.
        """
        return self._ftdi.get_wavelength()

    def set_wavelength(self, wav_nm: int, max_phase_pi: float = 2.0) -> None:
        """
        Set the phase calibration wavelength.

        Args:
            wav_nm: Target wavelength in nanometres.
            max_phase_pi: Maximum phase in units of pi; rounded to the nearest
                integer before passing to the firmware.
        """
        self._ftdi.set_wavelength(wav_nm, int(round(max_phase_pi)))

    def save_wavelength(self) -> None:
        """Persist the current phase table to EEPROM (AW command)."""
        self._ftdi.save_wavelength()

    def get_temperature(self) -> tuple[float, float]:
        """
        Read board temperatures.

        Returns:
            (drive_temp_C, option_temp_C).
        """
        return self._ftdi.get_temperature()

    def get_errors(self) -> tuple[int, int]:
        """
        Read error bitfields.

        Returns:
            (drive_error_bits, option_error_bits).
        """
        return self._ftdi.get_errors()

    def get_status(self) -> tuple[int, str, str]:
        """
        Read and decode FPGA status.

        Returns:
            (code, name, note) tuple.
        """
        resp = self._ftdi.get_status()
        return _STATUS_MAP.get(resp, (-99, "SLM_UNKNOWN", resp))

    def get_board_serial(self) -> str:
        """
        Read the drive board serial number.

        Returns:
            Drive board serial string.
        """
        return self._ftdi.get_board_serial()

    def get_option_board_serial(self) -> str:
        """
        Read the option board serial number.

        Returns:
            Option board serial string.
        """
        return self._ftdi.get_option_board_serial()

    def get_firmware_serial(self) -> str:
        """
        Read the FPGA firmware version string.

        Returns:
            Firmware version string, e.g. ``"2018021001"``.
        """
        return self._ftdi.get_firmware_serial()
