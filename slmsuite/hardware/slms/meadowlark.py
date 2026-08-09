"""
Hardware control for Meadowlark SLMs.

Meadowlark distributes several different interfaces for their products.
The following versions are supported in slmsuite at Meadowlark's suggestion
(contact Meadowlark to upgrade to one of these versions):

.. csv-table:: Meadowlark Compatibility
   :file: meadowlark.csv
   :widths: 25,25,25,25
   :header-rows: 1

Note
~~~~
When multiple SDKs are installed, the most recent SDK is chosen by default.
To choose otherwise, pass the path to the desired SDK.
"""
import os
import ctypes
import ctypes.wintypes
from enum import IntEnum
from pathlib import Path
import numpy as np
from platform import system
from typing import Union, Optional, Tuple, List

from slmsuite.hardware.slms.slm import SLM
from slmsuite._logging import make_logger

logger = make_logger(__name__)

#: str: Default location in which Meadowlark Optics software is installed
_DEFAULT_MEADOWLARK_PATH = "C:\\Program Files\\Meadowlark Optics\\"

class _SDK_MODE(IntEnum):
    #: No connection
    NULL = 0
    #: HDMI connection
    HDMI = 1
    #: High-Speed PCIe connection
    PCIE_MODERN_3 = 2
    PCIE_MODERN_6 = 3
    PCIE_MODERN_8 = 4
    PCIE_LEGACY = 5

    @property
    def is_pcie(self):
        """Whether this SDK mode uses a PCIe connection."""
        return self not in {_SDK_MODE.NULL, _SDK_MODE.HDMI}

_SDK_MODE_NAMES = {
    _SDK_MODE.NULL: "NULL",
    _SDK_MODE.HDMI: "HDMI",
    _SDK_MODE.PCIE_MODERN_3: "PCIe (modern, 3)",
    _SDK_MODE.PCIE_MODERN_6: "PCIe (modern, 6)",
    _SDK_MODE.PCIE_MODERN_8: "PCIe (modern, 8)",
    _SDK_MODE.PCIE_LEGACY: "PCIe (legacy)",
}

# To figure out which SDK is present, we parse the C header and determine the number of
# arguments if the traces for creating the SDK and writing an image. The allowed
# signatures (number of arguments) are defined here.
_SLM_LIB_TRACES = {
    _SDK_MODE.NULL: [],
    _SDK_MODE.HDMI: [(0, 2), (1, 2), (0, 3)],
    _SDK_MODE.PCIE_MODERN_3: [(2, 3)],
    _SDK_MODE.PCIE_MODERN_6: [(2, 6)],
    _SDK_MODE.PCIE_MODERN_8: [(2, 8)],
    _SDK_MODE.PCIE_LEGACY: [(8, 8)],
}

class Meadowlark(SLM):
    """
    Interfaces with Meadowlark SLMs.

    Attributes
    ----------
    sdk_mode : _SDK_MODE
        Mode of this SLM.
    slm_number : str
        Number
    """

    # Class attribute that loads each type of SDK library *once* per .dll for all class instances.
    _slm_lib = {}           # _SDK_Mode : ctypes.cdll
    _slm_lib_trace = {}     # _SDK_Mode : (int, int)    # See documentation for _SLM_LIB_TRACES
    _sdk_path = {}          # _SDK_Mode : str
    _number_of_boards = {}  # _SDK_Mode : int

    def __init__(  # noqa: R0913, R0917
        self,
        slm_number: int = 1,
        sdk_path: Optional[str] = None,
        lut_path: Optional[str] = None,
        wav_um: float = 1,
        pitch_um: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        r"""
        Initializes an instance of a Meadowlark SLM.

        Parameters
        ----------
        slm_number : int
            The board number of the SLM to connect to,
            in the case of multiple PCIe SLMs. Defaults to 1.
            Ignored for HDMI SLMs when the suggested 1.1.4.120 version of the SDK is
            used. For non-standard SDK version (for example in Blink 1.1.4.124),
            the `slm_number` argument may be used if C header requires.
        sdk_path : str
            Path of the Blink SDK installation folder.

            Important
            ~~~~~~~~~
            If the installation is not in the default folder,
            then this path needs to be specified. If there are multiple installations,
            then the most recent installation is chosen. The user must further specify
            the path otherwise. Keep in mind that different versions of the SDK may not
            be compatible with given hardware (HDMI, PCIe, etc.).
            See the compatibility table at
            :module:`slmsuite.hardware.slms.meadowlark` for more information.
        lut_path : str OR None
            Passed to :meth:`load_lut`. Looks for the voltage 'look-up table' data
            which is necessary to run the SLM.

            Tip
            ~~~
            See :meth:`load_lut` for how the default
            argument and other options are parsed.
        wav_um : float
            Wavelength of operation in microns. Defaults to 1 μm.
        pitch_um : (float, float) OR None
            Pixel pitch in microns.
            If ``None`` and using a PCIe SLM, this is automatically detected from the SDK.
            Otherwise, defaults to 8 micron square pixels (true for HDMI SLMs thus far).
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """
        # Validates the DPI awareness of this context, which is presumably important
        # for scaling.
        logger.debug("Validating DPI awareness...")

        awareness = ctypes.c_int()
        error_get = ctypes.windll.shcore.GetProcessDpiAwareness(
            0, ctypes.byref(awareness)
        )
        error_set = ctypes.windll.shcore.SetProcessDpiAwareness(2)
        success = ctypes.windll.user32.SetProcessDPIAware()

        if not success:
            raise RuntimeError(
                "Meadowlark failed to validate DPI awareness. "
                f"Errors: get={error_get}, set={error_set}, awareness="
                f"{awareness.value}"
            )

        logger.debug("Constructing Blink SDK...")

        # Locate the blink wrapper file in the sdk_path. If no sdk_path is provided,
        # search for the most recently installed SDK (if it exists)
        self.sdk_mode: _SDK_MODE = self._load_lib(sdk_path)
        if self.sdk_mode == _SDK_MODE.NULL:
            raise ValueError(f"Could not find SDK within path '{sdk_path}'")

        # Parse slm_number
        self.slm_number = int(slm_number)
        if self.sdk_mode != _SDK_MODE.HDMI:
            clip = (1, Meadowlark._number_of_boards[self.sdk_mode])
            if self.slm_number < clip[0] or self.slm_number > clip[1]:
                raise ValueError(
                    f"SLM number {self.slm_number} is outside the valid range {clip}"
                )

        # Triggering defaults
        self._wait_for_trigger = False
        self._output_pulse_image_flip = False
        self._output_pulse_image_refresh = False

        # Adjust pre- and post-ramp slopes for accurate voltage setting
        # (otherwise, custom LUT calibration is not properly implemented [this feature
        # is not implemented in slmsuite]). You may need a special version of the SDK
        # sent to you from Meadowlark to have access to these parameters.
        # Meadowlark._slm_lib[self.sdk_mode].SetPreRampSlope(20) # default is 7
        # Meadowlark._slm_lib[self.sdk_mode].SetPostRampSlope(24) # default is 24

        # If using a legacy 512x512, the 'true frames' need to be set
        if self.sdk_mode == _SDK_MODE.PCIE_LEGACY:
            Meadowlark._slm_lib[self.sdk_mode].Set_true_frames(ctypes.c_int(3))

        # Load LUT.
        logger.debug("Loading LUT file...")
        true_lut_path = self.load_lut(lut_path)
        if true_lut_path != lut_path:
            logger.debug("Loaded LUT from '%s'", true_lut_path)

        # If using a legacy 512x512, then the SLM needs to be powered on
        if self.sdk_mode == _SDK_MODE.PCIE_LEGACY:
            Meadowlark._slm_lib[self.sdk_mode].SLM_power(ctypes.c_bool(True))

        # Construct other variables.
        super().__init__(
            (
                Meadowlark._get_width(self.sdk_mode, self.slm_number),
                Meadowlark._get_height(self.sdk_mode, self.slm_number)
            ),
            bitdepth=Meadowlark._get_bitdepth(self.sdk_mode, self.slm_number),
            name=kwargs.pop("name", Meadowlark._get_serial(self.sdk_mode, self.slm_number)),
            wav_um=wav_um,
            pitch_um=(pitch_um if pitch_um else Meadowlark._get_pitch(self.sdk_mode, self.slm_number)),
            **kwargs,
        )

        if self.bitdepth > 8:
            logger.warning(
                f"Bitdepth of {self.bitdepth} > 8 detected; "
                "this has not been tested and might fail.",
                stacklevel=2,
            )

        self.set_phase(None)

    def close(self) -> None:
        """
        Use :meth:`.Meadowlark.close_sdk` to close the SDK, though this might break
        other SLMs on the same SDK. See :meth:`.SLM.close`.
        """
        pass

    def close_sdk(self) -> None:
        """
        See :meth:`.SLM.close`.
        """
        # If using a legacy SLM, it needs to be powered off
        if self.sdk_mode == _SDK_MODE.PCIE_LEGACY:
            Meadowlark._slm_lib[self.sdk_mode].SLM_power(ctypes.c_bool(False))

        # Review: This feels hacky because it is. If you try to unload a DLL
        # by setting to None and calling garbage collector it will not work
        OS = system()
        if OS == "Windows":
            # Warn: Don't call FreeLibrary directly, as it will not work due to
            #  the CDLL handle will be a 64-bit integer (integer overflow)
            unloader = ctypes.WinDLL('kernel32',
                                     use_last_error=True).FreeLibrary
            unloader.argtypes = [ctypes.wintypes.HMODULE] # Handle to memory addr
            unloader.restype = ctypes.wintypes.BOOL
        elif OS == "Linux":
            unloader = ctypes.cdll.LoadLibrary('libc.so.6').dlclose
            unloader.argtypes = [ctypes.c_void_p]
            unloader.restype = None
        else:
            unloader = lambda x: True

        try:
            Meadowlark._slm_lib[self.sdk_mode].Delete_SDK()
        except OSError as exc:
            logger.warning(f"Failed to delete SDK: {exc}", stacklevel=2)
        finally:
            # noinspection PyProtectedMember
            if not unloader(Meadowlark._slm_lib[self.sdk_mode]._handle):
                raise OSError(f"Failed to unload DLL {Meadowlark._sdk_path[self.sdk_mode]}; "
                              f"error_code: {ctypes.get_last_error()}.")
            del Meadowlark._slm_lib[self.sdk_mode]

    # General SDK inspection methods.
    @staticmethod
    def info(verbose: bool = True, sdk_path: str = None) -> List[Tuple[int, str]]:
        """
        Discover the SLMs connected to the selected SDK.
        Note that for HDMI, the SLM's display window will open at this stage.

        Parameters
        ----------
        verbose : bool
            Whether to print the information discovered.
        sdk_path : str
            Path of the Blink SDK installation folder to explore.

        Returns
        -------
        list
            The number and a descriptive string for each potential SLM.

        Raises
        ------
        NotImplementedError
            If multiple SLMs are not supported for this SDK
        """
        mode = Meadowlark._load_lib(sdk_path=sdk_path)

        if not mode in Meadowlark._sdk_path:
            raise RuntimeError("SDK failed to load.")

        info = [
            (
                board,
                f"{Meadowlark._get_serial(mode, board)} "
                f"({Meadowlark._get_width(mode, board)}x"
                f"{Meadowlark._get_height(mode, board)}, "
                f"{Meadowlark._get_bitdepth(mode, board)}-bit)"
            )
            for board in range(1, Meadowlark._number_of_boards[mode] + 1)
        ]
        if verbose:
            print(f"Using {_SDK_MODE_NAMES[mode]} SDK at '{Meadowlark._sdk_path[mode]}'")
            if len(info):
                for board, dims in info:
                    print(f"SLM {board}: {dims}")
            else:
                print("No boards found.")
        return info

    @staticmethod
    def _get_serial(sdk_mode: _SDK_MODE, slm_number: int) -> int:
        """
        Get the serial of the SLM.

        Returns
        -------
        str
            The serial of the SLM. Returns ``"Meadowlark HDMI"`` if HDMI SLM.
        """
        sdk = Meadowlark._slm_lib[sdk_mode]
        if sdk_mode == _SDK_MODE.HDMI:
            return "Meadowlark HDMI"
        elif sdk_mode.is_pcie:
            serial = sdk.Read_Serial_Number(ctypes.c_int(slm_number))
            return "Failed to load board" if serial == -1 else serial
        else:
            raise NotImplementedError("Serial retrieval not supported for this model")

    @staticmethod
    def _get_width(sdk_mode: _SDK_MODE, slm_number: int) -> int:
        """
        Get the width of the SLM.

        Returns
        -------
        int
            The width of the SLM.

        Raises
        ------
        NotImplementedError
            If the width retrieval is not supported for the SLM.
        """
        sdk = Meadowlark._slm_lib[sdk_mode]
        if sdk_mode == _SDK_MODE.HDMI:
            return sdk.Get_Width()
        elif sdk_mode.is_pcie:
            return sdk.Get_image_width(ctypes.c_int(slm_number))
        else:
            raise NotImplementedError("Width retrieval not supported for this model")

    @staticmethod
    def _get_height(sdk_mode: _SDK_MODE, slm_number: int) -> int:
        """
        Get the height of the SLM

        Returns
        -------
        int
            The height of the SLM.

        Raises
        ------
        NotImplementedError
            If the height retrieval is not supported for the SLM.
        """
        sdk = Meadowlark._slm_lib[sdk_mode]
        if sdk_mode == _SDK_MODE.HDMI:
            return sdk.Get_Height()
        elif sdk_mode.is_pcie:
            return sdk.Get_image_height(ctypes.c_int(slm_number))
        else:
            raise NotImplementedError("Height retrieval not supported for this model")

    @staticmethod
    def _get_bitdepth(sdk_mode: _SDK_MODE, slm_number: int) -> int:
        """
        Get the image depth of the SLM.

        Returns
        -------
        int
            The image depth of the SLM.

        Raises
        ------
        NotImplementedError
            If the image depth retrieval is not supported for the SLM.
        """
        sdk = Meadowlark._slm_lib[sdk_mode]
        if sdk_mode == _SDK_MODE.HDMI:
            return sdk.Get_Depth()
        elif sdk_mode.is_pcie:
            return sdk.Get_image_depth(ctypes.c_int(slm_number))
        else:
            raise NotImplementedError(
                "Image depth retrieval not supported for this model"
            )

    @staticmethod
    def _get_pitch(sdk_mode: _SDK_MODE, slm_number: int) -> Tuple[float, float]:
        """
        Get the pitch of the SLM.

        Returns
        -------
        tuple[float, float]
            The pitch of the SLM. Defaults to (8,8) if this method is not supported.
        """
        sdk = Meadowlark._slm_lib[sdk_mode]
        try:
            if sdk_mode.is_pcie:
                sdk.Get_pitch.restype = ctypes.c_double
                pitch = sdk.Get_pitch(ctypes.c_int(slm_number))
                return pitch, pitch
            else:
                raise NotImplementedError(
                    "Pitch retrieval not supported for this model"
                )
        except (NotImplementedError, AttributeError):
            return 8, 8

    # Assorted public methods
    def get_last_error_message(self) -> str:
        """
        Get the last error message from the SLM.

        Returns
        -------
        str
            The last error message.

        Raises
        ------
        NotImplementedError
            If the error message retrieval is not supported for the SLM.
        """
        if self.sdk_mode.is_pcie:
            Meadowlark._slm_lib[self.sdk_mode].Get_last_error_message.restype = ctypes.c_char_p
            return Meadowlark._slm_lib[self.sdk_mode].Get_last_error_message().decode("utf-8")
        else:
            raise NotImplementedError(
                "Error message retrieval not supported for this model"
            )

    def get_version_info(self) -> str:
        """
        Get the version information of the SLM.

        Returns
        -------
        str
            Version information.
        """
        sdk = Meadowlark._slm_lib[self.sdk_mode]
        sdk.Get_version_info.restype = ctypes.c_char_p
        return sdk.Get_version_info().decode("utf-8")

    def get_temperature(self) -> float:
        """
        Read the temperature of the SLM.

        Returns
        -------
        float
            Temperature in degrees celsius.

        Raises
        ------
        Not Implemented Error
            If the temperature reading is not supported for the SLM.
        """
        sdk = Meadowlark._slm_lib[self.sdk_mode]
        if self.sdk_mode == _SDK_MODE.HDMI:
            sdk.Get_SLMTemp.restype = ctypes.c_double
            return float(sdk.Get_SLMTemp())
        elif self.sdk_mode == _SDK_MODE.PCIE_MODERN_3:
            sdk.Read_SLM_temperature.restype = ctypes.c_double
            return float(sdk.Read_SLM_temperature(ctypes.c_int(self.slm_number)))
        elif self.sdk_mode in {
            _SDK_MODE.PCIE_MODERN_6,
            _SDK_MODE.PCIE_MODERN_8,
        }:
            sdk.Get_SLMTemp.restype = ctypes.c_double
            return float(sdk.Get_SLMTemp(ctypes.c_int(self.slm_number)))
        else:
            raise NotImplementedError(
                "Temperature reading not supported for this model."
            )

    def get_coverglass_voltage(self) -> float:
        """
        Read the voltage of the SLM coverglass.

        Returns
        -------
        float
            Voltage of the SLM coverglass.

        Raises
        ------
        Not Implemented Error
            If the coverglass voltage reading is not supported for the SLM.
        """
        sdk = Meadowlark._slm_lib[self.sdk_mode]
        if self.sdk_mode == _SDK_MODE.HDMI:
            sdk.Get_SLMVCom.restype = ctypes.c_double
            return float(sdk.Get_SLMVCom())
        elif self.sdk_mode in {
            _SDK_MODE.PCIE_MODERN_3,
            _SDK_MODE.PCIE_MODERN_6,
            _SDK_MODE.PCIE_MODERN_8,
        }:
            sdk.Get_cover_voltage.restype = ctypes.c_double
            return float(sdk.Get_cover_voltage(ctypes.c_int(self.slm_number)))
        else:
            raise NotImplementedError(
                "Coverglass voltage reading not supported for this model."
            )

    # Triggering methods
    def set_input_trigger(
        self,
        on : bool = False,
    ):
        """
        Configure the SLM to wait for an external input trigger before writing an image.
        No support for HDMI SLMs, partially supported for PCIe SLMs depending on
        the SDK version.

        Parameters
        ----------
        on : bool, optional
            If ``True``, enable waiting for an external input trigger before the image writes.
        """
        slm_number = ctypes.c_uint(self.slm_number)
        if self.sdk_mode == _SDK_MODE.HDMI:
            raise NotImplementedError("HDMI SLMs do not support input triggering.")
        elif self.sdk_mode.is_pcie:
            if self.sdk_mode == _SDK_MODE.PCIE_MODERN_3:
                Meadowlark._slm_lib[self.sdk_mode].SetWaitForTrigger(slm_number, ctypes.c_bool(on))
                Meadowlark._slm_lib[self.sdk_mode].SetFlipImmediate(slm_number, ctypes.c_bool(False))
            else:
                pass    # Handled with _wait_for_trigger in _set_phase_hw
            self._wait_for_trigger = on
        else:
            # We should never end up here, but if for some reason we did... We would
            # really want to know!!!
            raise RuntimeError("Failed to change input trigger on SLM due to unknown SDK mode")

    def set_output_trigger(
        self,
        on : bool = False,
        on_refresh : Optional[bool] = None,
    ):
        """
        Configure the SLM to send an external output triggers synchronized with image
        writes. No support for HDMI SLMs, partially supported for PCIe SLMs depending on
        the SDK version.

        Parameters
        ----------
        on : bool, optional
            If ``True``, an output trigger pulse is sent every time the image data changes.
        on_refresh : bool, optional
            If ``True``, an output trigger pulse is sent at the refresh rate of the SLM
            (even if the image data does not change).
            Not supported for all SDK versions.
        """
        slm_number = ctypes.c_uint(self.slm_number)
        if self.sdk_mode == _SDK_MODE.HDMI:
            raise NotImplementedError("HDMI SLMs do not support output triggering.")
        elif self.sdk_mode.is_pcie:
            if self.sdk_mode == _SDK_MODE.PCIE_MODERN_3:
                Meadowlark._slm_lib[self.sdk_mode].SetOutputPulse(slm_number, ctypes.c_bool(on))
            elif self.sdk_mode == _SDK_MODE.PCIE_MODERN_6:
                if on_refresh is not None:
                    logger.warning(
                        "on_refresh argument is ignored for this SDK version.",
                        stacklevel=2,
                    )
                # Handled with _output_pulse_image_flip in _set_phase_hw
            elif self.sdk_mode == _SDK_MODE.PCIE_MODERN_8:
                if on_refresh is not None:
                    self._output_pulse_image_refresh = on_refresh
                # Handled with _output_pulse_image_flip in _set_phase_hw

            self._output_pulse_image_flip = on

        else:
            # We should never end up here, but if for some reason we did... We would
            # really want to know!!!
            raise RuntimeError("Failed to change output trigger on SLM due to unknown SDK mode")

    # Main write function
    def _set_phase_hw(
            self,
            display: np.ndarray,
            execute : bool = True,
            block : bool = True,
            timeout_s : float = 5.0,
        ) -> None:
        """
        Hardware-specific implementation for Meadowlark SLM devices.

        See :meth:`SLM._set_phase_hw` for the base class documentation.

        Parameters
        ----------
        display : np.ndarray
            Integer data to display on the SLM. See :meth:`.SLM._set_phase_hw`.
        execute : bool
            Whether to actually send the image to the SLM. See :meth:`.SLM._set_phase_hw`.
        block : bool
            Whether to block the thread until the image is fully written.
            See :meth:`.SLM._set_phase_hw`.
        timeout_s : float
            Timeout for SLM trigger.
        """
        if hasattr(display, "get"):
            display = display.get()

        slm_number = ctypes.c_uint(self.slm_number)
        if self.sdk_mode == _SDK_MODE.HDMI:
            if Meadowlark._slm_lib_trace[_SDK_MODE.HDMI][1] == 2:       # 2 arguments
                Meadowlark._slm_lib[self.sdk_mode].Write_image(
                    display.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                    ctypes.c_uint(self.bitdepth == 8),  # Is 8-bit
                )
            elif Meadowlark._slm_lib_trace[_SDK_MODE.HDMI][1] == 3:     # 3 arguments
                Meadowlark._slm_lib[self.sdk_mode].Write_image(
                    slm_number,
                    display.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                    ctypes.c_uint(self.bitdepth == 8),  # Is 8-bit
                )
        elif self.sdk_mode.is_pcie:
            wait_for_trigger = ctypes.c_bool(self._wait_for_trigger)
            # WARN: Do not change this, as doing so will loses the guarantee that
            #  all the pixels are synchronized to the same image (that is, the earliest
            #  pixels can be updated to the next image before the last pixels are
            #  updated to the current image).
            flip_immediate = ctypes.c_bool(False)
            output_pulse_image_flip = ctypes.c_bool(self._output_pulse_image_flip)
            output_pulse_image_refresh = ctypes.c_bool(self._output_pulse_image_refresh)
            trigger_timeout = ctypes.c_uint(int(timeout_s * 1000))

            if execute:
                # Switch between the different supported cases for number of Write_image arguments:
                if self.sdk_mode == _SDK_MODE.PCIE_MODERN_3:
                    status = Meadowlark._slm_lib[self.sdk_mode].Write_image(
                        slm_number,
                        display.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                        trigger_timeout,
                    )
                elif self.sdk_mode == _SDK_MODE.PCIE_MODERN_6:
                    status = Meadowlark._slm_lib[self.sdk_mode].Write_image(
                        slm_number,
                        display.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                        wait_for_trigger,
                        flip_immediate,
                        output_pulse_image_flip,
                        trigger_timeout,
                    )
                elif self.sdk_mode in {_SDK_MODE.PCIE_MODERN_8, _SDK_MODE.PCIE_LEGACY}:
                    status = Meadowlark._slm_lib[self.sdk_mode].Write_image(
                        slm_number,
                        display.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                        ctypes.c_uint(self.shape[0] * self.shape[1]),
                        wait_for_trigger,
                        flip_immediate,
                        output_pulse_image_flip,
                        output_pulse_image_refresh,
                        trigger_timeout,
                    )
                else:
                    raise RuntimeError("Failed to set phase on SLM due to unknown SDK mode")

                if status != 1:
                    raise RuntimeError("DMA Failed")

            if block:
                status = Meadowlark._slm_lib[self.sdk_mode].ImageWriteComplete(slm_number, trigger_timeout)
                if status != 1:
                    raise RuntimeError("ImageWriteComplete failed, trigger never received?")
        else:
            # We should never end up here, but if for some reason we did...
            raise RuntimeError("Failed to set phase on SLM due to unknown SDK mode")

    # Load library helpers
    @staticmethod
    def _load_lib(sdk_path: str = None) -> int:
        """
        Infers the location of the Meadowlark SDK's dynamic linked library.

        Parameters
        ----------
        sdk_path : str
            Path to search for the Meadowlark SDK.

        Returns
        -------
        str
            The path to the Meadowlark SDK folder.

        Raises
        ------
        FileNotFoundError
            If no Blink_C_Wrapper.dll files are found in provided path.

        """
        # Parse sdk_path.
        if sdk_path is None:
            sdk_path = _DEFAULT_MEADOWLARK_PATH
        else:
            sdk_path = os.path.dirname(sdk_path)

        # Locate the Meadowlark SDK. If there are multiple, default to the
        # most recent one.
        files = list(Path(sdk_path).rglob("*Blink_C_Wrapper*dll"))
        files.sort(key=os.path.getmtime, reverse=True)
        cases = []
        for file in files:
            if not ("Cal Kit" in str(file)):
                mode, dll_path, trace = Meadowlark._parse_header(file.parents[0], warn=True)
                if mode:
                    cases.append((mode, dll_path, trace))

        if len(cases) == 0:
            raise FileNotFoundError(
                f"No Blink_C_wrapper.dll file found in '{sdk_path}'."
            )

        mode, dll_path, trace = cases[0]

        # Check to see if we've already loaded this SDK.
        if mode in Meadowlark._sdk_path:
            if Meadowlark._sdk_path[mode] == dll_path:
                return mode
            else:
                raise RuntimeError(
                    f"Meadowlark {_SDK_MODE_NAMES[mode]} SDK already loaded from path "
                    f"'{Meadowlark._sdk_path[mode]}', "
                    f"cannot also load from '{dll_path}'."
                )

        # If we got to here, we need to actually load the SDK.
        if len(cases) > 1:
            options = ',\n'.join([f"'{case[1]}' ({_SDK_MODE_NAMES[case[0]]})" for case in cases])
            logger.warning(
                f"Multiple Meadowlark SDKs located. "
                f"Defaulting to the most recent one"
                f" '{dll_path}'. "
                f"This is a {_SDK_MODE_NAMES[mode]} SDK.\n"
                f"Other options:\n{options}"
            )

        # First load the .dll
        try:
            Meadowlark._slm_lib[mode] = ctypes.CDLL(dll_path, mode=ctypes.RTLD_LOCAL)
            Meadowlark._slm_lib_trace[mode] = trace
            Meadowlark._sdk_path[mode] = dll_path
            Meadowlark._number_of_boards[mode] = 1
        except OSError as exc:
            raise ImportError(
                f"Meadowlark .dlls did not did not import correctly. "
                f"Is '{dll_path}' the correct path?"
            ) from exc

        try:
            if mode == _SDK_MODE.HDMI:
                if trace[0] == 0:
                    Meadowlark._slm_lib[mode].Create_SDK()
                elif trace[0] == 1:
                    bool_cpp_or_python = ctypes.c_uint(1)
                    Meadowlark._slm_lib[mode].Create_SDK(bool_cpp_or_python)
            elif mode == _SDK_MODE.PCIE_LEGACY:
                # Technically, some of these boards had "OverDrive Plus" technology that
                # allowed increased switching speed. It greatly simplifies the implementation
                # to load all boards without this feature.
                bitdepth = ctypes.c_uint(8)  # all the old HSP512 were 8-bit as far as I know
                number_of_boards = ctypes.c_uint(0)
                constructed_okay = ctypes.c_bool(False)
                is_nematic_type = ctypes.c_bool(True)
                ram_write_enable = ctypes.c_bool(True)
                # Only for ODP or for Meadowlark's ImageGen software to make images
                use_gpu = ctypes.c_bool(False)
                max_transients = ctypes.c_uint(10)  # Only for ODP; leaving at default '10'
                regional_lut = ctypes.c_int(0)  # Only for ODP; leaving at default 'NULL'

                Meadowlark._slm_lib[mode].Create_SDK(
                    bitdepth,
                    ctypes.byref(number_of_boards),
                    ctypes.byref(constructed_okay),
                    is_nematic_type,
                    ram_write_enable,
                    use_gpu,
                    max_transients,
                    regional_lut,
                )

                if not constructed_okay.value:
                    del Meadowlark._slm_lib[mode]
                    del Meadowlark._slm_lib_trace[mode]
                    del Meadowlark._sdk_path[mode]
                    del Meadowlark._number_of_boards[mode]
                    raise RuntimeError("SDK call failed.")

                Meadowlark._number_of_boards[mode] = number_of_boards.value
            elif mode in {
                _SDK_MODE.PCIE_MODERN_3,
                _SDK_MODE.PCIE_MODERN_6,
                _SDK_MODE.PCIE_MODERN_8,
            }:
                number_of_boards = ctypes.c_uint(-1)
                constructed_okay = ctypes.c_int(-1)

                Meadowlark._slm_lib[mode].Create_SDK(
                    ctypes.byref(number_of_boards),
                    ctypes.byref(constructed_okay),
                )

                if not (constructed_okay.value in [0,1]):   # 0 is success but no boards found
                    del Meadowlark._slm_lib[mode]
                    del Meadowlark._slm_lib_trace[mode]
                    del Meadowlark._sdk_path[mode]
                    del Meadowlark._number_of_boards[mode]
                    raise RuntimeError("SDK call failed.")

                Meadowlark._number_of_boards[mode] = number_of_boards.value
        except Exception:
            logger.error("Failed to construct SDK.")
            raise

        return mode

    @staticmethod
    def _parse_header(file: str, warn: bool = False) -> Tuple[_SDK_MODE, str, Tuple[int, int]]:
        """Checks if a path has an appropriate header"""
        dll_path = os.path.join(file, "Blink_C_wrapper.dll")
        dll_present = os.path.isfile(dll_path)
        header_path = os.path.join(file, "Blink_C_wrapper.h")
        header_present = os.path.isfile(header_path)

        if header_present and dll_present:
            with open(header_path, "r") as f:
                data = f.read()

            trace = []

            for name in ["Create_SDK(", "Write_image("]:
                # noinspection PyBroadException
                try:
                    index = data.find(name)
                    split1 = data[index:].split("(")[1]
                    split2 = split1.split(")")[0]

                    if len(split2) < 2:
                        trace.append(0)
                    else:
                        trace.append(len(split2.split(",")))
                except Exception:
                    trace = None
                    break

            if trace is None:    # Header was unparseable; treat as "not recognized".
                return _SDK_MODE.NULL, "", None

            trace = tuple(trace)

            for mode in _SDK_MODE:
                if trace in _SLM_LIB_TRACES[mode]:
                    return mode, dll_path, trace

            if warn:
                logger.warning(
                    f"Your SDK's header has (create, write) argument trace {trace}, which is not "
                    "recognized. Contact Meadowlark and slmsuite support to update your SDK version."
                )

            return _SDK_MODE.NULL, "", None
        elif dll_present:
            if warn:
                logger.warning(f"Found dll '{dll_path}' but not header '{header_path}'.")
        elif header_present:
            if warn:
                logger.warning(f"Found header '{header_path}' but not dll '{dll_path}'.")
        return _SDK_MODE.NULL, "", None

    # LUT stuff
    def load_lut(self, lut_path: Optional[str] = None) -> str:
        """
        Loads a voltage 'look-up table' (LUT) to the SLM.
        This converts requested phase values to physical voltage perturbing
        the liquid crystals.

        Parameters
        ----------
        lut_path : str OR None
            Path to look for an LUT file in.

            -   If this is a .lut file, then this file is loaded to the SLM.
            -   If this is a directory, then searches all files inside the
                directory, and loads the most recently created .lut file or if possible
                the .lut file starting with ``"slm"`` which is more likely to
                correspond to the LUT customized to an SLM, as Meadowlark sends such
                files prefixed by ``"slm"`` such as ``"slm5758_at532.lut"``.

        Returns
        -------
        str
            The path which was used to load the LUT.

        Raises
        ------
        RuntimeError
            If the LUT file fails to load
        FileNotFoundError
            If no .lut files are found in provided path or the specified file does not
            exist.
        """
        # REVIEW: To facilitate selection of the correct LUT in the case where
        #  multiple different SLMs are installed, use the dimensions of the SLM
        #  to narrow down the search.
        slm_dims = (self.shape[0], self.shape[1]) if hasattr(self, "shape") else(
            Meadowlark._get_width(self.sdk_mode, self.slm_number),
            Meadowlark._get_height(self.sdk_mode, self.slm_number),
        )
        # None, default to the SDK path. Use parent since SDK path has
        # 'Blink_C_Wrapper.dll' appended
        if lut_path is None:
            lut_path = Path(self._sdk_path[self.sdk_mode]).parent

        # Find the exact LUT file
        if os.path.isdir(lut_path):
            try:
                lut_path = Meadowlark._locate_lut_file(lut_path, slm_dims)
            except FileNotFoundError:
                # If no LUT file is file, try the parent directory as sometimes the LUTs
                # are in the parent directory of the SDK path in 'LUT'
                lut_path = Meadowlark._locate_lut_file(Path(lut_path).parent, slm_dims)

        # If the search path doesn't exist, short circuit
        if not os.path.exists(lut_path):
            raise FileNotFoundError(f"Failed to locate LUT at: '{lut_path}'")

        # Finally, actually load the LUT file
        try:
            if self.sdk_mode == _SDK_MODE.HDMI:
                Meadowlark._slm_lib[self.sdk_mode].Load_lut(lut_path)
            elif self.sdk_mode.is_pcie:
                success = Meadowlark._slm_lib[self.sdk_mode].Load_LUT_file(
                    ctypes.c_int(self.slm_number), lut_path.encode("utf-8")
                )
                if success != 1:
                    logger.warning(f"Failed to load LUT file: '{lut_path}'")
            else:
                raise RuntimeError("Failed to load LUT file due to unknown SDK mode")
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to load LUT file: '{lut_path}'") from exc
        else:
            return lut_path

    @staticmethod
    def _locate_lut_file(
        search_path: Union[str, Path], slm_shape: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Locates the LUT file in the given path. If there are multiple, returns the
        most recent file.. If there are none with ".slm" in the
        name, simply returns the most recent ".lut" file.

        Parameters
        ----------
        search_path : str
            Path to search for the LUT file in.
        slm_shape : (int, int)
            Shape of the SLM to search for, in standard ``(height, width)`` form.

        Returns
        -------
        str
            The path to the LUT file.

        Raises
        ------
        FileNotFoundError
            If no .lut files are found in provided path.
        """
        files = {file for file in Path(search_path).rglob("*.lut")}

        if len(files) == 1:
            return str(files.pop())
        elif len(files) > 1:
            # REVIEW: If there are multiple LUTs, first check if only one matches the
            #  current slm's dimensions (if possible).
            if slm_shape:
                files = {
                    file
                    for file in files
                    if (f"{slm_shape[1]}" in file.stem and f"{slm_shape[0]}" in file.stem)
                }
                if len(files) == 1:
                    return str(files.pop())
            # If there are still multiple LUTs, default to the most recent one.
            lut_path_ = max(files, key=os.path.getctime)
            logger.warning(
                f"Multiple LUT files located. Defaulting to the most recent one: "
                f"{lut_path_}.",
                stacklevel=3,
            )
            return str(lut_path_)
        else:
            raise FileNotFoundError(f"No .lut file found in '{search_path}'.")
