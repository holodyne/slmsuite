"""
Low-level USB protocol driver for Santec SLMs via FTDI FT601.

This module provides :class:`SantecFTDI`, a self-contained class that wraps the
binary USB protocol used by Santec LCOS SLMs (SLM-200, and likely SLM-210 and
SLM-300). It has no dependency on Santec's vendor DLLs and requires only the
``PyD3XX`` Python package, which bundles the FTDI D3XX libraries and requires no
separate OS-level driver installation.

Note
~~~~
The protocol was reverse-engineered by observing ``FT_WritePipe`` and
``FT_ReadPipe`` calls from the Santec vendor software into ``D3XX.dll``.
Command codes follow the DLL naming convention: suffix of ``SLM_Ctrl_*``
maps to the 2-letter USB command (e.g. ``SLM_Ctrl_WriteVI`` -> ``VI``).

Note
~~~~
Install ``PyD3XX`` via pip::

    pip install PyD3XX

``PyD3XX`` bundles the FTDI D3XX shared libraries for Windows, Linux, and
macOS. No separate driver installation is required.

Note
~~~~
See ``context/santec_dll_api.md``, ``context/herosdevices_api.md``, and
``context/mapping_api.md`` in the slmsuite repository for full documentation of
the DLL API, the reverse-engineered protocol, and their mapping.
"""

import ctypes as _ctypes
import struct
import time
import warnings

import numpy as np
import numpy.typing as npt

try:
    import PyD3XX as _PyD3XX
except ImportError, OSError:
    _PyD3XX = None

# --- packet framing constants ---

# struct format for the 16-byte packet header:
#   4s = b"SEND" magic, H = magic_length, x = pad, B = cmd_id, I = seq_id, 4x = pad
_PACKET_HEADER_FMT = ">4sHxBIxxxx"
_HEADER_SIZE = 16
_CONTROL_PAYLOAD_SIZE = 1024  # control command payloads are padded to this size
_IMAGE_ROW_PAYLOAD_SIZE = 4096  # image row payloads are padded to this size
_STATUS_READ_SIZE = _HEADER_SIZE + _CONTROL_PAYLOAD_SIZE  # 1040 bytes per status response

# command type tuples: (cmd_id, magic_length)
_CMD_CONTROL = (1, 0xFF)  # ASCII control commands (VI, WL, DS, ...)
_CMD_STATUSREQUEST = (2, 0)  # poll for last response
_CMD_STATUSRESPONSE = (3, 0xFF)  # response from FPGA (read only)
_CMD_IMAGEDATA = (4, 0x03BF)  # binary image row data

# pipe timeouts
_PIPE_TIMEOUT_MS = 1000  # default read pipe timeout
_WRITE_TIMEOUT_MS = 5000  # longer timeout for large image uploads (~4.7 MB per frame)

# D3XX WinUSB status: async op started but data not yet available; some driver
# versions return this from null-overlapped reads instead of blocking
_FT_IO_PENDING = 32

# firmware version strings validated during open(); known versions for SLM-200.
# SLM-210 and SLM-300 may return different strings -- add them when known.
KNOWN_FIRMWARE_VERSIONS: frozenset[str] = frozenset(
    {
        "2018021001",
        "2018021101",
        "2018020001",
        "2017080002",
        "2015010001",
    }
)


class SantecFTDI:
    """
    Low-level USB protocol driver for Santec SLMs via FTDI FT601.

    Wraps the binary USB protocol spoken by Santec LCOS SLMs. All device
    configuration uses ASCII control commands (``CMD_CONTROL`` packets); image
    data uses a binary multi-packet protocol (``CMD_IMAGEDATA`` packets).

    Supports the context manager protocol::

        with SantecFTDI("AB000001") as dev:
            dev.set_video_mode(0)
            dev.upload_image(1, frame)
            dev.display_slot(1)

    Attributes
    ----------
    serial_number : str
        FTDI chip serial number used to identify the device.
    channel : int
        FTDI FIFO channel index (0 for single-channel devices).
    MIN_SLOT : int
        Lowest valid memory slot index.
    MAX_SLOT : int
        Highest valid memory slot index.
    """

    MIN_SLOT: int = 1
    MAX_SLOT: int = 128

    def __init__(self, serial_number: str, channel: int = 0) -> None:
        """
        Parameters
        ----------
        serial_number : str
            FTDI chip serial number string.
        channel : int
            FTDI FIFO channel index. Defaults to 0.

        Raises
        ------
        RuntimeError
            If ``PyD3XX`` is not installed.
        """
        if _PyD3XX is None:
            raise RuntimeError("PyD3XX not available; cannot use SantecFTDI.\n" "Install it with:  pip install PyD3XX")
        self.serial_number = serial_number
        self.channel = channel
        # FIFO index (int) for FT_ReadPipeEx / FT_WritePipeEx on Linux/macOS
        self._fifo_out = channel
        self._fifo_in = channel
        # FT_Pipe objects obtained from FT_GetPipeInformation after open();
        # used for FT_AbortPipe on all platforms, and FT_WritePipeEx/FT_ReadPipeEx on Windows
        self._pipe_out = None
        self._pipe_in = None
        self._device = None

    # -------------------------------------------------------------------------
    # context manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> "SantecFTDI":
        """
        Open the device connection.

        Returns
        -------
        SantecFTDI
            Self.
        """
        self.open()
        return self

    def __exit__(self, *_) -> None:
        """Close the device connection."""
        self.close()

    # -------------------------------------------------------------------------
    # lifecycle
    # -------------------------------------------------------------------------

    def open(self) -> None:
        """
        Open the FTDI USB connection and verify the firmware.

        Flushes the input pipe and retries the ``SN`` firmware query up to 10
        times to confirm the FPGA is ready.

        Raises
        ------
        RuntimeError
            If the device serial number is not found, the FTDI handle cannot be
            opened, or firmware verification fails after 10 attempts.
        """
        status, count = _PyD3XX.FT_CreateDeviceInfoList()
        if status != _PyD3XX.FT_OK or count == 0:
            raise RuntimeError("No FTDI FT60x devices found.")
        device_index = None
        device = None
        serials = []
        for i in range(count):
            s, info = _PyD3XX.FT_GetDeviceInfoDetail(i)
            if s == _PyD3XX.FT_OK:
                serials.append(info.SerialNumber)
                if info.SerialNumber == self.serial_number:
                    device_index = i
                    device = info  # reuse FT_Device from FT_GetDeviceInfoDetail (required by FT_Create)
        if device_index is None:
            raise RuntimeError("Device '{}' not found. Available: {}.".format(self.serial_number, serials))
        # pass the same FT_Device object returned by FT_GetDeviceInfoDetail, not a fresh one
        status = _PyD3XX.FT_Create(device_index, _PyD3XX.FT_OPEN_BY_INDEX, device)
        if status != _PyD3XX.FT_OK:
            raise RuntimeError(
                "Failed to open FTDI device '{}' at index {} (status {}).".format(
                    self.serial_number, device_index, status
                )
            )
        # FT_GetPipeInformation pipe indices vary by OS and D3XX driver version.
        # On Linux: interface 0, idx 0=OUT(0x02), idx 1=IN(0x82) per FIFO channel.
        # On Windows: interface 0 exposes 4 pipes (0x01 vendor-OUT, 0x81 interrupt-IN,
        #   0x02 data-OUT, 0x82 data-IN); the formula 2*channel gives the vendor pipe,
        #   not the data pipe. Bypass FT_GetPipeInformation on Windows entirely and
        #   construct FT_Pipe objects with the fixed FT601 data endpoint addresses.
        if _PyD3XX.Platform == "windows":
            self._pipe_out = _PyD3XX.FT_Pipe()
            self._pipe_out._PipeID = _ctypes.c_char(bytes([0x02 + 2 * self.channel]))
            self._pipe_in = _PyD3XX.FT_Pipe()
            self._pipe_in._PipeID = _ctypes.c_char(bytes([0x82 + 2 * self.channel]))
        else:
            _, self._pipe_out = _PyD3XX.FT_GetPipeInformation(device, 0, 2 * self.channel + 1)
            _, self._pipe_in = _PyD3XX.FT_GetPipeInformation(device, 0, 2 * self.channel)
        # Windows uses FT_SetPipeTimeout for blocking sync reads; Linux uses per-call timeout
        if _PyD3XX.Platform == "windows":
            _PyD3XX.FT_SetPipeTimeout(device, self._pipe_in, _PIPE_TIMEOUT_MS)
        _PyD3XX.FT_AbortPipe(device, self._pipe_in)
        # D3XX on Windows needs a brief settle after FT_AbortPipe before reads are reliable
        if _PyD3XX.Platform == "windows":
            time.sleep(0.2)
        self._device = device
        try:
            # drain any stale data left in the pipe from a previous crashed session;
            # use a short timeout so this returns quickly when the pipe is clean
            for _ in range(32):
                if _PyD3XX.Platform == "windows":
                    _, ft_buf, n = _PyD3XX.FT_ReadPipeEx(device, self._pipe_in, _STATUS_READ_SIZE, _PyD3XX.NULL)
                else:
                    _, ft_buf, n = _PyD3XX.FT_ReadPipeEx(device, self._fifo_in, _STATUS_READ_SIZE, 50)
                if n == 0:
                    break
            last_error: str | Exception = "no attempts made"
            for _ in range(10):
                try:
                    firmware = self.get_firmware_serial()
                    if firmware not in KNOWN_FIRMWARE_VERSIONS:
                        warnings.warn(
                            "Unrecognized firmware version '{}' for device '{}'. "
                            "Proceeding; add to KNOWN_FIRMWARE_VERSIONS if the device "
                            "works correctly.".format(firmware, self.serial_number)
                        )
                    return
                except Exception as e:
                    last_error = e
            raise RuntimeError(
                "Could not verify firmware for device '{}'. "
                "Last error: {}. Known versions: {}.".format(
                    self.serial_number, last_error, sorted(KNOWN_FIRMWARE_VERSIONS)
                )
            )
        except Exception:
            self._device = None
            _PyD3XX.FT_Close(device)
            raise

    def close(self) -> None:
        """Close the FTDI USB connection."""
        if self._device is not None:
            try:
                _PyD3XX.FT_Close(self._device)
            finally:
                self._device = None

    @staticmethod
    def list_devices() -> list[str]:
        """
        List serial numbers of all connected FTDI FT60x devices.

        Returns
        -------
        list of str
            FTDI serial number strings. Empty if no devices are found.

        Raises
        ------
        RuntimeError
            If ``PyD3XX`` is not installed.
        """
        if _PyD3XX is None:
            raise RuntimeError("PyD3XX not available.")
        status, count = _PyD3XX.FT_CreateDeviceInfoList()
        if status != _PyD3XX.FT_OK or count == 0:
            return []
        serials = []
        for i in range(count):
            s, info = _PyD3XX.FT_GetDeviceInfoDetail(i)
            if s == _PyD3XX.FT_OK and info.SerialNumber:
                serials.append(info.SerialNumber)
        return serials

    # -------------------------------------------------------------------------
    # private transport
    # -------------------------------------------------------------------------

    def _assemble_packet(
        self,
        cmd_id: int,
        magic_length: int,
        payload: bytes,
        seq_id: int = 0,
    ) -> bytes:
        header = struct.pack(_PACKET_HEADER_FMT, b"SEND", magic_length, cmd_id, seq_id)
        if magic_length == 0xFF:
            # control packets: zero-pad payload to _CONTROL_PAYLOAD_SIZE
            padded = bytearray(_CONTROL_PAYLOAD_SIZE)
            padded[: len(payload)] = payload
            return header + bytes(padded)
        return header + payload

    def _disassemble_packet(self, buffer: bytes) -> dict:
        _, magic_length, cmd_id = struct.unpack(">4sHxBxxxxxxxx", buffer[:_HEADER_SIZE])
        return {
            "magic_length": magic_length,
            "id": cmd_id,
            "payload": buffer[_HEADER_SIZE:],
        }

    def _write(self, buffer: bytes) -> None:
        if self._device is None:
            raise RuntimeError("Device not open.")
        ft_buf = _PyD3XX.FT_Buffer.from_bytes(buffer)
        if _PyD3XX.Platform == "windows":
            # Windows: FT_Pipe object + NULL overlapped (blocking sync)
            status, bytes_written = _PyD3XX.FT_WritePipeEx(
                self._device, self._pipe_out, ft_buf, len(buffer), _PyD3XX.NULL
            )
        else:
            # Linux/macOS: FIFO index (int) + timeout in ms; 0 is non-blocking
            status, bytes_written = _PyD3XX.FT_WritePipeEx(
                self._device, self._fifo_out, ft_buf, len(buffer), _WRITE_TIMEOUT_MS
            )
        if status != _PyD3XX.FT_OK:
            raise RuntimeError("USB write failed (status {}).".format(status))
        if bytes_written != len(buffer):
            raise RuntimeError("USB write incomplete: {} of {} bytes written.".format(bytes_written, len(buffer)))

    def _read(self, length: int) -> bytes:
        if self._device is None:
            raise RuntimeError("Device not open.")
        received = b""
        while len(received) < length:
            remaining = length - len(received)
            if _PyD3XX.Platform == "windows":
                # some D3XX WinUSB driver versions return FT_IO_PENDING immediately
                # instead of blocking when null-overlapped reads find an empty pipe;
                # poll at 10 ms intervals for up to _PIPE_TIMEOUT_MS
                deadline = time.monotonic() + _PIPE_TIMEOUT_MS / 1000.0
                while True:
                    status, ft_buf, bytes_read = _PyD3XX.FT_ReadPipeEx(
                        self._device, self._pipe_in, remaining, _PyD3XX.NULL
                    )
                    if bytes_read > 0:
                        break
                    if status != _FT_IO_PENDING:
                        raise RuntimeError("USB read failed (status {}).".format(status))
                    if time.monotonic() >= deadline:
                        raise RuntimeError("USB read timed out after {}ms.".format(_PIPE_TIMEOUT_MS))
                    time.sleep(0.010)
            else:
                # Linux/macOS: FIFO index (int) + timeout in ms; 0 is non-blocking
                status, ft_buf, bytes_read = _PyD3XX.FT_ReadPipeEx(
                    self._device, self._fifo_in, remaining, _PIPE_TIMEOUT_MS
                )
                if bytes_read == 0:
                    raise RuntimeError("USB read failed (status {}).".format(status))
            received += bytes(ft_buf.Value()[:bytes_read])
        return received

    def _poll_status(self, trials: int = 100, sleep: float = 0.015) -> str:
        """
        Send a status request and poll until the FPGA gives a real response.

        Parameters
        ----------
        trials : int
            Maximum polling attempts before returning ``"NO RESPONSE"``.
        sleep : float
            Seconds to wait between attempts.

        Returns
        -------
        str
            FPGA response: ``"OK"``, ``"NG"``, a queried value, or
            ``"NO RESPONSE"`` if the FPGA never answers within the trial budget.
        """
        response = "NO RESPONSE"
        for _ in range(trials):
            self._write(self._assemble_packet(_CMD_STATUSREQUEST[0], _CMD_STATUSREQUEST[1], b""))
            buf = self._read(_STATUS_READ_SIZE)
            pkt = self._disassemble_packet(buf)
            response = pkt["payload"].decode("utf-8").split("\x00")[0].strip()
            if response != "NO RESPONSE":
                break
            time.sleep(sleep)
        if response == "BS":
            # device is booting; back off and retry
            time.sleep(1.0)
            return self._poll_status(trials, sleep)
        return response

    def _control_command(
        self,
        command: bytes,
        params: list[int] | None = None,
        longrunning: bool = False,
    ) -> str:
        """
        Send an ASCII control command and return the FPGA response.

        Parameters
        ----------
        command : bytes
            Two-byte ASCII command code, e.g. ``b"VI"``.
        params : list of int, optional
            Integer parameters appended space-separated after the code.
        longrunning : bool
            Use a 1 s poll interval. Set for ``WL`` calibration (~40 s).

        Returns
        -------
        str
            FPGA response string (``"OK"``, ``"NG"``, or a queried value).
        """
        payload = command
        if params is not None:
            for p in params:
                payload += b" %i" % p
        payload += b"\x0d"
        self._write(self._assemble_packet(_CMD_CONTROL[0], _CMD_CONTROL[1], payload))
        return self._poll_status(sleep=1.0 if longrunning else 0.015)

    # -------------------------------------------------------------------------
    # status & identity
    # -------------------------------------------------------------------------

    def get_status(self) -> str:
        """
        Poll and return the current FPGA status string.

        Returns
        -------
        str
            One of ``"OK"``, ``"BS"``, ``"NG"``, or ``"NO RESPONSE"``.
        """
        return self._poll_status()

    def get_firmware_serial(self) -> str:
        """
        Read the firmware version string from the SLM FPGA.

        The firmware serial (e.g. ``"2018021001"``) identifies the firmware
        build. It is distinct from the FTDI chip serial number used to open the
        device, but matches the ``SerialNumberID`` field of
        ``SLM_Disp_Info2``.

        Returns
        -------
        str
            Firmware version string.
        """
        return self._control_command(b"SN")

    # -------------------------------------------------------------------------
    # video mode -- SLM_Ctrl_WriteVI / ReadVI
    # -------------------------------------------------------------------------

    def set_video_mode(self, mode: int) -> str:
        """
        Set the video input source.

        Parameters
        ----------
        mode : int
            ``0`` = USB/Memory mode, ``1`` = DVI mode.

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"VI", [mode])

    def get_video_mode(self) -> int:
        """
        Read the current video input source.

        Returns
        -------
        int
            ``0`` for USB/Memory, ``1`` for DVI.
        """
        return int(self._control_command(b"VI"))

    # -------------------------------------------------------------------------
    # phase table -- SLM_Ctrl_WriteWL / ReadWL / WriteAW
    # -------------------------------------------------------------------------

    def set_wavelength(self, wavelength_nm: int, max_phase_pi: int = 2) -> str:
        """
        Set the phase calibration table for a target wavelength.

        This operation recalculates the full LUT and takes roughly 40 seconds.

        Note
        ~~~~
        The DLL encodes max phase as ``floor(radians * 100 / pi)`` (200 for
        2*pi). The USB protocol uses plain integer pi-multiples (2 for 2*pi).
        Do not mix the two encodings when porting code from the DLL driver.

        Parameters
        ----------
        wavelength_nm : int
            Wavelength in nanometres.
        max_phase_pi : int
            Maximum phase in units of pi. Defaults to 2 (= 2*pi).

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"WL", [wavelength_nm, max_phase_pi], longrunning=True)

    def get_wavelength(self, longrunning: bool = False) -> tuple[int, float]:
        """
        Read the current phase calibration table settings.

        Response format confirmed on hardware: ``"NM PHASE"`` where PHASE is a
        float (e.g. ``"532 2.09"``).

        Parameters
        ----------
        longrunning : bool
            Use a 1 s poll interval. Set to ``True`` when calling immediately
            after :meth:`set_wavelength`, as the FPGA may be briefly unresponsive
            during post-calibration settling.

        Returns
        -------
        (int, float)
            ``(wavelength_nm, max_phase_pi)`` tuple.
        """
        resp = self._control_command(b"WL", longrunning=longrunning)
        if resp in ("NO RESPONSE", "NG", "BS"):
            raise RuntimeError("WL read returned {!r}; FPGA may still be busy.".format(resp))
        parts = resp.split()
        if len(parts) < 2:
            raise RuntimeError(
                "WL read returned unexpected response {!r}; expected 'NM PHASE' (e.g. '532 2.09'). "
                "Possible stale response in pipe -- try re-opening the device.".format(resp)
            )
        return (int(parts[0]), float(parts[1]))

    def save_wavelength(self) -> str:
        """
        Save the current phase table to non-volatile flash.

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"AW")

    # -------------------------------------------------------------------------
    # grayscale / contrast -- SLM_Ctrl_WriteGS / ReadGS
    # -------------------------------------------------------------------------

    def set_grayscale(self, value: int) -> str:
        """
        Set the contrast/gamma level of the LCOS panel.

        Parameters
        ----------
        value : int
            Integer in range 0-1023.

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"GS", [int(value)])

    def get_grayscale(self) -> int:
        """
        Read the current contrast/gamma level.

        Note
        ~~~~
        Returns ``NG`` on at least one firmware version (2018021001); the read
        variant of ``GS`` may not be supported by all firmware builds.

        Returns
        -------
        int
            Integer in range 0-1023.

        Raises
        ------
        RuntimeError
            If the device returns ``"NG"``.
        """
        resp = self._control_command(b"GS")
        if resp == "NG":
            raise RuntimeError("GS read returned NG; firmware may not support reading grayscale.")
        return int(resp)

    # -------------------------------------------------------------------------
    # memory slots -- SLM_Ctrl_WriteMI / WriteDS / ReadDS / WriteME
    # -------------------------------------------------------------------------

    def upload_image(self, slot: int, image: npt.NDArray[np.uint16]) -> str:
        """
        Upload a uint16 pixel array to a memory slot.

        Selects the target slot via the ``MI`` command, then streams the image
        as a sequence of ``CMD_IMAGEDATA`` packets (one per row), followed by a
        uint32 checksum in the last four bytes of the buffer.

        Note
        ~~~~
        The checksum is computed as ``int(image.sum()) % 2**32``. Integer
        overflow is intentional and matches the FPGA expectation.

        Parameters
        ----------
        slot : int
            Target memory slot, ``MIN_SLOT`` to ``MAX_SLOT``.
        image : numpy.ndarray
            2-D uint16 array of shape ``(height, width)``. Values should be in
            the range 0-1023 for a 10-bit SLM.

        Returns
        -------
        str
            FPGA response string.

        Raises
        ------
        ValueError
            If ``slot`` is out of range.
        """
        if not (self.MIN_SLOT <= slot <= self.MAX_SLOT):
            raise ValueError("Slot {} out of valid range ({}-{}).".format(slot, self.MIN_SLOT, self.MAX_SLOT))
        self._control_command(b"MI", [slot])
        image = image.astype(np.uint16)
        buffer = bytearray()
        for i, row in enumerate(image):
            linedata = bytearray(_IMAGE_ROW_PAYLOAD_SIZE)
            row_bytes = row.tobytes()
            linedata[: len(row_bytes)] = row_bytes
            buffer += self._assemble_packet(_CMD_IMAGEDATA[0], _CMD_IMAGEDATA[1], bytes(linedata), seq_id=i)
        # overwrite last 4 bytes with uint32 checksum; intentional integer overflow
        buffer[-4:] = struct.pack("I", int(image.sum()) % 2**32)
        self._write(bytes(buffer))
        return self._poll_status()

    def display_slot(self, slot: int) -> str:
        """
        Set the memory slot displayed on the SLM.

        Parameters
        ----------
        slot : int
            Memory slot to display, ``MIN_SLOT`` to ``MAX_SLOT``.

        Returns
        -------
        str
            FPGA response string.

        Raises
        ------
        ValueError
            If ``slot`` is out of range.
        """
        if not (self.MIN_SLOT <= slot <= self.MAX_SLOT):
            raise ValueError("Slot {} out of valid range ({}-{}).".format(slot, self.MIN_SLOT, self.MAX_SLOT))
        return self._control_command(b"DS", [slot])

    def get_displayed_slot(self) -> int:
        """
        Read the currently displayed memory slot.

        Note
        ~~~~
        Returns ``NG`` on at least one firmware version (2018021001); the read
        variant of ``DS`` may not be supported by all firmware builds.

        Returns
        -------
        int
            Slot number, ``MIN_SLOT`` to ``MAX_SLOT``.

        Raises
        ------
        RuntimeError
            If the device returns ``"NG"``.
        """
        resp = self._control_command(b"DS")
        if resp == "NG":
            raise RuntimeError("DS read returned NG; firmware may not support reading displayed slot.")
        return int(resp)

    def erase_slot(self, slot: int) -> str:
        """
        Erase the contents of a memory slot.

        Parameters
        ----------
        slot : int
            Memory slot to erase, ``MIN_SLOT`` to ``MAX_SLOT``.

        Returns
        -------
        str
            FPGA response string.

        Raises
        ------
        ValueError
            If ``slot`` is out of range.
        """
        if not (self.MIN_SLOT <= slot <= self.MAX_SLOT):
            raise ValueError("Slot {} out of valid range ({}-{}).".format(slot, self.MIN_SLOT, self.MAX_SLOT))
        return self._control_command(b"ME", [slot])

    # -------------------------------------------------------------------------
    # memory table / sequencing
    # SLM_Ctrl_WriteMT / ReadMS / WriteMR / ReadMR / WriteMP / WriteMZ /
    # WriteMW / ReadMW / WriteDR / WriteDB
    # -------------------------------------------------------------------------

    def set_table_entry(self, table_number: int, slot: int) -> str:
        """
        Map a memory table position to a memory slot.

        Parameters
        ----------
        table_number : int
            Table position index (1-128).
        slot : int
            Memory slot to assign at that position.

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"MT", [table_number, slot])

    def get_table_entry(self, table_number: int) -> int:
        """
        Read the memory slot assigned to a table position.

        Command code ``MS`` confirmed on hardware (``SLM_Ctrl_ReadMS``).

        Parameters
        ----------
        table_number : int
            Table position index.

        Returns
        -------
        int
            Memory slot number at that position.
        """
        return int(self._control_command(b"MS", [table_number]))

    def set_playback_range(self, start: int, end: int) -> str:
        """
        Set the table range used for sequential playback.

        Parameters
        ----------
        start : int
            First table position in the range (inclusive).
        end : int
            Last table position in the range (inclusive).

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"MR", [start, end])

    def get_playback_range(self) -> tuple[int, int]:
        """
        Read the current sequential playback range.

        Returns
        -------
        (int, int)
            ``(start, end)`` table position indices.
        """
        parts = self._control_command(b"MR").split()
        return (int(parts[0]), int(parts[1]))

    def set_table_position(self, position: int) -> str:
        """
        Set the current position in the memory table.

        Parameters
        ----------
        position : int
            Table position index.

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"MP", [position])

    def reset_table_position(self) -> str:
        """
        Reset the memory table position to zero.

        Command code ``MZ`` confirmed on hardware (``SLM_Ctrl_WriteMZ``).

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"MZ")

    def set_framerate(self, frames: int) -> str:
        """
        Set the auto-trigger period in 60 Hz frame units.

        One frame equals 1/60 s. Valid range is 1 (1/60 s) to 120 (2 s).

        Parameters
        ----------
        frames : int
            Number of 60 Hz video frames per trigger period.

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"MW", [int(frames)])

    def get_framerate(self) -> int:
        """
        Read the current auto-trigger period in 60 Hz frame units.

        Returns
        -------
        int
            Frame count (1-120).
        """
        return int(self._control_command(b"MW"))

    def start_playback(self, ascending: bool = True) -> str:
        """
        Start continuous sequential playback through the memory table.

        Parameters
        ----------
        ascending : bool
            ``True`` to step forward through slots, ``False`` for reverse.

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"DR", [1 if ascending else 0])

    def stop_playback(self) -> str:
        """
        Stop continuous sequential playback.

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"DB")

    # -------------------------------------------------------------------------
    # triggers -- SLM_Ctrl_WriteTI / ReadTI / WriteTM / ReadTM /
    #             WriteTC / ReadTC / WriteTS
    # -------------------------------------------------------------------------

    def set_trigger_input(self, enabled: bool) -> str:
        """
        Enable or disable the external trigger input (SMB jack).

        Parameters
        ----------
        enabled : bool
            ``True`` to enable the external trigger input.

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"TI", [1 if enabled else 0])

    def get_trigger_input(self) -> bool:
        """
        Read whether the external trigger input is enabled.

        Returns
        -------
        bool
            ``True`` if the external trigger input is active.
        """
        return bool(int(self._control_command(b"TI")))

    def set_trigger_output(self, enabled: bool) -> str:
        """
        Enable or disable the trigger output signal.

        Parameters
        ----------
        enabled : bool
            ``True`` to enable the trigger output.

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"TM", [1 if enabled else 0])

    def get_trigger_output(self) -> bool:
        """
        Read whether the trigger output is enabled.

        Returns
        -------
        bool
            ``True`` if the trigger output is active.
        """
        return bool(int(self._control_command(b"TM")))

    def set_trigger_direction(self, ascending: bool) -> str:
        """
        Set the edge direction used when stepping through memory slots.

        Parameters
        ----------
        ascending : bool
            ``True`` for rising edge / forward step.

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"TC", [1 if ascending else 0])

    def get_trigger_direction(self) -> bool:
        """
        Read the current trigger edge direction.

        Returns
        -------
        bool
            ``True`` if the direction is ascending / rising edge.
        """
        return bool(int(self._control_command(b"TC")))

    def fire_software_trigger(self) -> str:
        """
        Fire a software trigger to advance to the next memory slot.

        Only effective when the device has been configured for software-trigger
        mode via :meth:`set_trigger_direction` or equivalent setup.

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"TS")

    def set_trigger_enable(self, enabled: bool) -> str:
        """
        Enable or disable the trigger subsystem.

        Command code ``TE`` confirmed on hardware: accepts ``0`` or ``1``,
        returns ``"OK"`` or ``"NG"``. Exact semantics (global trigger gate vs.
        some other enable) are not yet documented by Santec.

        Parameters
        ----------
        enabled : bool
            ``True`` to enable, ``False`` to disable.

        Returns
        -------
        str
            FPGA response string.
        """
        return self._control_command(b"TE", [1 if enabled else 0])

    def get_trigger_enable(self) -> bool:
        """
        Read whether the trigger subsystem is enabled.

        Returns
        -------
        bool
            ``True`` if enabled.
        """
        return bool(int(self._control_command(b"TE")))

    # -------------------------------------------------------------------------
    # diagnostics -- SLM_Ctrl_ReadT / ReadEDO / ReadSDO
    # -------------------------------------------------------------------------

    def get_temperature(self) -> tuple[float, float]:
        """
        Read drive board and option board temperatures in degrees Celsius.

        Calls :meth:`get_drive_temperature` and :meth:`get_option_temperature`
        in sequence. The combined ``SLM_Ctrl_ReadT`` DLL call maps to these two
        separate USB commands (``T`` gives NO RESPONSE; ``TD``/``TO`` are the
        per-board variants confirmed on hardware).

        Returns
        -------
        (float, float)
            ``(drive_temp_celsius, option_temp_celsius)``.
        """
        return (self.get_drive_temperature(), self.get_option_temperature())

    def get_drive_temperature(self) -> float:
        """
        Read drive board temperature in degrees Celsius.

        Command code ``TD`` confirmed on hardware (returns e.g. ``"47.1"``).
        Note: the programmer's guide documents the DLL out-param as INT32*10,
        but the USB firmware returns the value as a float string directly.

        Returns
        -------
        float
            Drive board temperature in degrees Celsius.
        """
        return float(self._control_command(b"TD"))

    def get_option_temperature(self) -> float:
        """
        Read option board temperature in degrees Celsius.

        Command code ``TO`` confirmed on hardware (returns e.g. ``"56.6"``).
        Note: the programmer's guide documents the DLL out-param as INT32*10,
        but the USB firmware returns the value as a float string directly.

        Returns
        -------
        float
            Option board temperature in degrees Celsius.
        """
        return float(self._control_command(b"TO"))

    def get_errors(self) -> tuple[int, int]:
        """
        Read drive board and option board error bitfields.

        Command code ``ED`` confirmed on hardware. Response is a 4-character
        hex string: first 2 chars = drive board bits, last 2 chars = option
        board bits (e.g. ``"0000"`` = no errors).

        Drive board error bits (from ``_slm_win.py`` ``SLM_DRIVEBOARD_ERROR``):

        - ``0x01``: startup error 1
        - ``0x02``: startup error 2
        - ``0x04``: video signal error (no signal)
        - ``0x08``: temperature error (>= 70 C)

        Option board error bits (from ``_slm_win.py`` ``SLM_OPTIONBOARD_ERROR``):

        - ``0x01``: startup error 1
        - ``0x02``: startup error 2
        - ``0x04``: voltage level error (DC 5.0 V)
        - ``0x08``: temperature error (>= 70 C)

        Returns
        -------
        (int, int)
            ``(drive_error_bits, option_error_bits)`` as integers.
        """
        resp = self._control_command(b"ED")
        return (int(resp[:2], 16), int(resp[2:4], 16))

    def get_board_serial(self) -> str:
        """
        Read the drive board serial number string.

        Command code ``SD`` confirmed on hardware (programmer's guide p.67,
        ``SLM_Ctrl_ReadSD``). Response is an 8-character hex string
        (e.g. ``"22030184"``).

        Returns
        -------
        str
            Drive board serial number string.
        """
        return self._control_command(b"SD")

    def get_option_board_serial(self) -> str:
        """
        Read the option board serial number string.

        Command code ``SO`` confirmed on hardware (programmer's guide p.68,
        ``SLM_Ctrl_ReadSO``). Response is an 8-character hex string
        (e.g. ``"22030064"``).

        Returns
        -------
        str
            Option board serial number string.
        """
        return self._control_command(b"SO")

    def get_version(self) -> str:
        """
        Read detailed firmware version strings for all subsystems.

        Command code ``VR`` confirmed on hardware. Returns a multi-line string
        with version and SVN revision for the option CPU, drive CPU, and Xilinx
        FPGA, separated by ``\\r\\n``.

        Example response::

            OptionCPU      : 00000001
            OptionCPU(SVN) : 0321
            DriveCPU       : 00000001
            DriveCPU(SVN)  : 0322
            XilinxFPGA     : 0110

        Returns
        -------
        str
            Multi-line version string.
        """
        return self._control_command(b"VR")
