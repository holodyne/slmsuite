"""
Hardware control for Hamamatsu SLMs in USB/Trigger mode.
For DVI mode, reset the SLM to DVI mode externally and
project information onto the appropriate screen using
:class:`~slmsuite.hardware.slms.screenmirrored.ScreenMirrored`.
Tested with Hamamatsu LCOS-SLM X15213.

Important
~~~~~~~~~
:class:`.Hamamatsu` requires dynamically linked libraries from Hamamatsu to be present in the
runtime directory:

- hpkSLMdaLV.dll
- hpkSLMda.dll

These two files must be in the same directory.

Note
~~~~
Hamamatsu provides base wavefront correction accounting for the curvature of the SLM surface.
Consider loading these files via :meth:`.SLM.load_vendor_phase_correction()`

Note
~~~~
Reading/writing data to a microSD card is currently unsupported.
"""

from ctypes import (
    POINTER,
    WinDLL,
    byref,
    c_char,
    c_char_p,
    c_double,
    c_int,
    c_int32,
    c_short,
    c_ubyte,
    c_uint,
    c_uint8,
    c_uint32,
    c_ulonglong,
    c_ushort,
    c_void_p,
    c_wchar,
    windll,
)
import os
import warnings

import numpy as np

from slmsuite.hardware.slms.slm import SLM

try:
    _libname = "hpkSLMdaLV.dll"

    if hasattr(os, "add_dll_directory"):  # python >= 3.8
        os.add_dll_directory(os.getcwd())
        os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))
        Lcoslib = WinDLL(_libname)
    else:  # python < 3.8
        _libpath = os.path.dirname(os.path.abspath(__file__))
        os.environ["PATH"] = _libpath + os.pathsep + os.environ["PATH"]
        Lcoslib = windll.LoadLibrary(_libname)
except Exception as e:
    warnings.warn(
        "Hamamatsu DLLs not installed; must be present in the runtime directory:\n"
        "  - hpkSLMdaLV.dll\n  - hpkSLMda.dll\n"
        "Install to use Hamamatsu SLMs.\n"
        f"Original error: {e}"
    )
    Lcoslib = None


class Hamamatsu(SLM):
    r"""
    Initializes an instance of a Hamamatsu SLM.

    Attributes
    ----------
    serial_number : str
        Serial number of the connected device.
    """

    def __init__(
        self,
        serial_number=None,
        wav_um=1,
        resolution=(1272, 1024),
        pitch_um=(12.5, 12.5),
        verbose=True,
        **kwargs,
    ):
        r"""
        Initializes an instance of a Hamamatsu SLM.
        Keep in mind that this initializes into USB/Trigger mode.
        For DVI mode, preset the SLM to DVI mode externally and use
        :class:`slmsuite.hardware.slms.screenmirrored.ScreenMirrored`.

        Caution
        ~~~~~~~
        This interface currently supports connecting to only one device.

        Caution
        ~~~~~~~
        The default values of ``resolution`` and ``pitch_um``
        are relative to the model LCOS-SLM X15213-02.

        Parameters
        ----------
        serial_number : str OR None
            Serial number of the connected device.
            If ``None``, the first connected device will be used.
        wav_um : float
            Wavelength of operation in microns. Defaults to 1 um.
        pitch_um : (float, float)
            Pixel pitch in microns. Defaults to 12.5 micron square pixels.
        """
        # Search for one device.
        if verbose:
            print("Initializing Hamamatsu SDK...", end="")
        n_dev, board_ids = self._Open_Device(bID_size=1)
        self.board_id = list(board_ids)[0]

        if n_dev == 0:
            raise RuntimeError("No Hamamatsu devices found!")

        if verbose:
            print("success")

        # Read the serial number of the device.
        if serial_number is None:
            if verbose:
                print("Looking for SLM...", end="")
        else:
            if verbose:
                print(f"Looking for '{serial_number}'...", end="")

        self.serial_number = self._Check_HeadSerial(board_id=self.board_id)

        # Error if the desired serial number is not found.
        if serial_number is not None:
            if serial_number in self.serial_number or self.serial_number in serial_number:
                pass
            else:
                self._Close_Device(board_ids, bID_size=1)
                raise RuntimeError(
                    f"Could not find '{serial_number}'. Found '{self.serial_number}'."
                )

        if verbose:
            print("success")

        # Force the SLM to USB/Trigger mode.
        try:
            if verbose:
                print("Checking SLM mode...", end="")
            mode = self._Mode_Check(board_id=self.board_id)

            if mode == 0:
                if verbose:
                    print("found DVI mode...switching to USB...", end="")
                self._Mode_Select(board_id=self.board_id, mode=1)

                if verbose:
                    print("rebooting...", end="")
                self._Reboot(board_id=self.board_id)
            elif mode == 1:
                if verbose:
                    print("found USB mode...", end="")
            else:
                raise RuntimeError(f"Unknown SLM mode {mode}.")

            if verbose:
                print("success")
        except Exception as e:
            self._Close_Device(board_ids, bID_size=1)
            raise e

        # Use the superclass to construct the other variables.
        super().__init__(
            resolution=resolution,
            bitdepth=8,  # SDK only supports 8-bit, so we're hardcoding.
            wav_um=wav_um,
            pitch_um=pitch_um,
            **kwargs,
        )

        # Zero the display using the superclass `set_phase()` function.
        self.set_phase(None)

    def _set_phase_hw(self, display, slot_number=0):
        r"""
        Hardware-specific implementation for Hamamatsu SLM devices.

        See :meth:`SLM._set_phase_hw` for the base class documentation.

        Parameters
        ----------
        display
            Integer data to display on the SLM. See :meth:`.SLM._set_phase_hw`.
        slot_number : int
            Hardware-specific implementation. The number of the frame memory
            slot where to write the pattern. The default value is 0.
            This can be passed as a ``**kwargs``argument to :meth:`set_phase()`;
            this variable may be renamed in a future slmsuite release to
            conform with eventual implementation of this feature in other SLMs.
        """
        array_size = int(self.shape[0] * self.shape[1])

        write_fmemarray = Lcoslib.Write_FMemArray
        write_fmemarray.argtyes = [
            c_uint8,
            c_uint8 * array_size,
            c_int32,
            c_uint32,
            c_uint32,
            c_uint32,
        ]

        # TODO: do python ints need to be converted explicitly to c_uint32?
        v = write_fmemarray(
            self.board_id,
            display.ctypes.data_as(POINTER(c_uint8 * array_size)).contents,
            array_size,
            self.shape[1],
            self.shape[0],
            slot_number,
        )

        if v != 1:
            raise RuntimeError("Failed to write to Hamamatsu SLM.")

        # TODO: is this necessary to call every frame?
        self._Change_DispSlot(slot_number)

    def set_slot_number(self, slot_number=0):
        r"""
        Switches the displayed pattern to the one in the specified slot number, from the
        frame memory. This function may be deprecated in the future.

        Parameters
        ----------
        slot_number : int
            The number of the frame memory slot. The default value is 0.
        """
        self._Change_DispSlot(slot_number)

    def get_display(self):
        r"""
        Reads the current displayed pattern from the SLM.
        """
        display = np.zeros(self.shape, dtype=np.uint8)
        array_size = int(self.shape[0] * self.shape[1])

        get_display = Lcoslib.Check_Disp_IMG
        get_display.argtyes = [c_uint8, c_int32, c_uint32, c_uint32, c_uint8 * array_size]
        v = get_display(
            self.board_id,
            array_size,
            self.shape[1],
            self.shape[0],
            display.ctypes.data_as(POINTER(c_uint8 * array_size)).contents,
        )

        if v != 1:
            raise RuntimeError("Failed to read from Hamamatsu SLM.")

        return display

    @staticmethod
    def _Mode_Select(board_id, mode):
        r"""
        Sets the SLM to the desired mode.

        Parameters
        ----------
        board_id : int
            The ID of the SLM board.
        mode : int
            The desired mode of the SLM.

            - ``0`` : DVI mode
            - ``1`` : USB/Trigger mode
        """
        mode_select = Lcoslib.Mode_Select
        mode_select.argtyes = [c_uint8, c_uint32]
        v = mode_select(board_id, mode)

        if v != 1:
            raise RuntimeError("Failed to set Hamamatsu SLM mode.")

    @staticmethod
    def _Mode_Check(board_id):
        r"""
        Returns the current mode of the SLM.

        Parameters
        ----------
        board_id : int
            The ID of the SLM board.

        Returns
        -------
        mode : int
            The current mode of the SLM.

            - ``0`` : DVI mode
            - ``1`` : USB/Trigger mode
        """
        mode_check = Lcoslib.Mode_Check
        mode_check.argtyes = [c_uint8]
        mode = c_uint32(0)
        v = mode_check(board_id, byref(mode))

        if v != 1:
            raise RuntimeError("Failed to read Hamamatsu SLM mode.")

        return mode.value

    @staticmethod
    def _Reboot(board_id):
        r"""
        Allows to restart the controller board.
        """
        reboot = Lcoslib.Reboot
        reboot.argtyes = [c_uint8]
        reboot(board_id)

    @staticmethod
    def _Open_Device(bID_size=1):
        r"""
        Establishes the communication with all the LCOS-SLM controllers connected to the USB.

        NB: make sure that the bID_list has the same length of the number of
            connected devices to avoid problem with other functions.

        Returns
        -------
        conn_dev :
            Number of connected devices.
        ID_list :
            ID of the connected devices.
        """
        open_dev = Lcoslib.Open_Dev
        open_dev.argtyes = [c_uint8 * bID_size, c_int32]
        open_dev.restype = c_int
        array = c_uint8 * bID_size
        ID_list = array(0)
        conn_dev = open_dev(byref(ID_list), bID_size)

        return conn_dev, ID_list

    @staticmethod
    def _Close_Device(bID_list, bID_size=1):
        r"""
        Interrupts the communication with the target devices.
        """
        close_dev = Lcoslib.Close_Dev
        close_dev.argtyes = [c_uint8 * bID_size, c_int32]
        close_dev.restype = c_int

        v = close_dev(byref(bID_list), bID_size)

        if v != 1:
            raise RuntimeError("Failed to close Hamamatsu device.")

    def close(self) -> None:
        """
        Closes the connection to the SLM device.
        """
        bID_list = (c_uint8 * len([self.board_id]))(*[self.board_id])
        self._Close_Device(bID_list=bID_list, bID_size=1)

    @staticmethod
    def _Check_HeadSerial(board_id):
        r"""
        Reads the LCOS-SLM head serial number with the desired ID.
        """
        check_serial = Lcoslib.Check_HeadSerial
        check_serial.argtyes = [c_uint8, c_char * 11, c_int32]
        hs = c_char * 11
        head_serial = hs(0)
        v = check_serial(board_id, byref(head_serial), 11)

        if v != 1:
            raise RuntimeError("Failed to read Hamamatsu serial number.")

        return head_serial.value.decode("utf-8").strip("\x00")

    def _Change_DispSlot(self, slot_number):
        r"""
        Changes the displayed pattern to the one in the specified slot number, from the frame memory.
        """
        change_slot = Lcoslib.Change_DispSlot
        change_slot.argtyes = [c_uint8, c_uint32]
        v = change_slot(self.board_id, slot_number)

        if v != 1:
            raise RuntimeError("Failed to change the display slot for Hamamatsu SLM.")

    def get_temperature(self):
        r"""
        Reads the temperature of the LCOS-SLM head and controller.

        Returns
        -------
        (head_temperature, controller_temperature) : (float, float)
            The temperatures of the SLM's head and controller.
        """
        check_temp = Lcoslib.Check_Temp
        head_temperature = c_double(0)
        controller_temperature = c_double(0)
        check_temp.argtyes = [c_uint8, c_double, c_double]

        v = check_temp(self.board_id, byref(head_temperature), byref(controller_temperature))

        if v != 1:
            raise RuntimeError(f"Could not check Hamamatsu temperature (error={v}).")

        return (head_temperature.value, controller_temperature.value)

    def get_led_status(self):
        r"""
        Checks the lighting status of the LED.

        Returns
        -------
        led_status : list of int
            List of 10 integers representing the LED status.
        """
        check_led = Lcoslib.Check_LED
        ls = c_uint32 * 10
        led_status = ls(0)
        check_led.argtyes = [c_uint8, c_uint32 * 10]
        v = check_led(self.board_id, byref(led_status))

        if v != 1:
            raise RuntimeError(f"Could not check Hamamatsu LED status (error={v}).")

        return list(led_status)
