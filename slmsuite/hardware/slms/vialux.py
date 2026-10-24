r"""
Hardware control for ViALUX Phase Light Modulators (PLMs).
Tested with the V-9903P (Texas Instruments DLPM990 on the ALP-5.1 controller).

Unlike a :class:`~slmsuite.hardware.slms.texasinstruments.PLM`, which is driven as a
display, a ViALUX PLM is driven by the ALP controller: patterns are uploaded over USB
(or PCIe) into the controller's memory, which displays them and maps each phase level
onto the PLM's electrodes itself.

.. highlight:: python
.. code-block:: python

    import numpy as np
    from slmsuite.hardware.slms.vialux import PLM

    # List the connected ALP devices and their serial numbers.
    PLM.info()

    # Open the first available PLM, or a specific one by serial number.
    plm = PLM(wav_um=0.633)
    plm = PLM(serial=12345, wav_um=0.633)

    # Set a phase pattern.
    phase = np.random.rand(1088, 2048) * 2 * np.pi
    plm.set_phase(phase)

Note
~~~~
:class:`PLM` requires ViALUX's ALP-5.1 software, whose ``alp51.dll`` is loaded from
``C:\\Program Files\\ALP-5.1\\ALP-5.1 API\\x64\\`` by default. Pass ``sdk_path`` to
load it from elsewhere. The constants used here are those of the ALP-5.1 ``alp.h``
(header version 29 or later, which adds ``ALP_DMDTYPE_2K_099A_PLM`` and the
``ALP_PLM_*_BIAS_VOLTAGE`` controls) and the ALP-5.1 API description.

Caution
~~~~~~~
``alp.h`` describes :attr:`DMDMode.INTERFACE_ERROR` as an error on the DMD interface,
latched until ``AlpDevHalt()``, which the V5199 board shows on LED LD14. The ALP raises
it at random when loading the DMD, while displaying patterns correctly. See
:meth:`PLM.get_dmd_mode` and :meth:`PLM.reset_interface`.
"""

import ctypes
import os
import time
from contextlib import contextmanager
from enum import IntEnum

import numpy as np

from slmsuite._logging import make_logger
from slmsuite.hardware.slms._plm import _PLMModel
from slmsuite.hardware.slms.slm import SLM
from slmsuite.misc.xp import as_numpy

logger = make_logger(__name__)

# ALP Constants
_DEFAULT_ALP_PATH = "C:\\Program Files\\ALP-5.1\\ALP-5.1 API\\x64\\alp51.dll"
ALP_DEFAULT = 0
ALP_INVALID_ID = 0xFFFFFFFF

# Models in texasinstruments.yaml by the DMD type which the ALP reports.
_DMD_TYPE_MODELS = {14: "p99"}  # ALP_DMDTYPE_2K_099A_PLM

_alp_libs = {}  # str path : ctypes.CDLL, loaded once per path.


class _ALPReturn(IntEnum):
    """
    ALP return codes, as defined in ``alp.h`` and section 1.4 of the ALP-5.1 API
    description.
    """

    OK = 0
    NOT_ONLINE = 1001  # The ALP has not been found or is not ready.
    NOT_IDLE = 1002  # The ALP is not in idle state.
    NOT_AVAILABLE = 1003  # The ALP identifier is not valid.
    NOT_READY = 1004  # The ALP is already allocated.
    PARM_INVALID = 1005  # One of the parameters is invalid.
    ADDR_INVALID = 1006  # Error accessing user data.
    MEMORY_FULL = 1007  # The requested memory is not available.
    SEQ_IN_USE = 1008  # The sequence is currently in use.
    HALTED = 1009  # The ALP was stopped during image data transfer.
    ERROR_INIT = 1010  # Initialization error.
    ERROR_COMM = 1011  # Communication error.
    DEVICE_REMOVED = 1012  # The ALP has been removed.
    NOT_CONFIGURED = 1013  # The onboard FPGA is not configured.
    LOADER_VERSION = 1014  # Unsupported by this version of VlxUsbLd.sys.
    ERROR_POWER_DOWN = 1018  # Waking up the DMD from ALP_DMD_POWER_FLOAT failed.
    DRIVER_VERSION = 1019  # Update the drivers and power-cycle the device.
    SDRAM_INIT = 1020  # SDRAM initialization failed.
    CONFIG_MISMATCH = 1021  # The device is not configured for this call.
    TIMEOUT_INTERNAL = 1022  # Internal synchronization timeout.
    TIMEOUT_COMM = 1023  # Timeout transferring data to the ALP.
    ERROR_INTERFACE = 1024  # Interface failure; reset with ALP_INTERFACE_RESET.
    FIRMWARE_VERSION = 1025  # The FPGA firmware is out of date.


class ALPControl(IntEnum):
    """
    ALP control and inquire types used by :class:`PLM`.

    Each value is the ``ControlType`` or ``InquireType`` passed to the ALP API, as
    defined in ``alp.h`` and the ALP-5.1 API description. Any other type can be passed
    to the raw access methods (e.g. :meth:`PLM.dev_control`) as a plain ``int``.
    """

    # AlpDevControl / AlpDevInquire, by section of the API description.
    DEVICE_NUMBER = 2000  #: 2.3 -- Serial number
    SYNCH_POLARITY = 2004  #: 2.2 -- Frame synch output polarity
    TRIGGER_EDGE = 2005  #: 2.2 -- Trigger input edge (slave mode)
    DEV_DMDTYPE = 2021  #: 2.2 -- DLP chip type
    DDC_FPGA_TEMPERATURE = 2050  #: 2.3 -- DLPC FPGA temperature, in 1/256 degrees Celsius
    APPS_FPGA_TEMPERATURE = 2051  #: 2.3 -- Applications FPGA temperature, in 1/256 degrees Celsius
    PCB_TEMPERATURE = 2052  #: 2.3 -- Board temperature, in 1/256 degrees Celsius
    DEV_DISPLAY_HEIGHT = 2057  #: 2.3 -- Mirror rows
    DEV_DISPLAY_WIDTH = 2058  #: 2.3 -- Mirror columns
    DEV_DMD_MODE = 2064  #: 2.2 -- Mirror state; see DMDMode
    MAX_DDC_FPGA_TEMPERATURE = 2145  #: 2.3 -- Recommended maximum, in degrees Celsius
    MAX_APPS_FPGA_TEMPERATURE = 2146  #: 2.3 -- Recommended maximum, in degrees Celsius
    MAX_PCB_TEMPERATURE = 2147  #: 2.3 -- Recommended maximum, in degrees Celsius
    DMD_TEMPERATURE = 2159  #: 2.3 -- DMD temperature, in 1/256 degrees Celsius
    DMD_PCB_TEMPERATURE = 2160  #: 2.3 -- DMD board temperature, in 1/256 degrees Celsius
    PLM_BIAS_VOLTAGE = 2363  #: alp.h -- PLM bias voltage, in millivolts
    PLM_MIN_BIAS_VOLTAGE = 2364  #: alp.h -- PLM bias voltage minimum, in millivolts
    PLM_MAX_BIAS_VOLTAGE = 2365  #: alp.h -- PLM bias voltage maximum, in millivolts

    # AlpSeqControl / AlpSeqInquire
    DATA_FORMAT = 2110  #: 2.8 -- Data format and alignment
    MIN_PICTURE_TIME = 2211  #: 2.10 -- Minimum picture time, in microseconds

    # AlpProjControl / AlpProjInquire
    PROJ_MODE = 2300  #: 2.13 -- Master or slave mode
    PROJ_STATE = 2400  #: 2.14 -- Projection active or idle


class DMDMode(IntEnum):
    """States of ``ALP_DEV_DMD_MODE``, as defined in ``alp.h``."""

    RESUME = 0  # Normal operation.
    POWER_FLOAT = 1  # Powered down, with the mirrors released to flat.
    INTERFACE_ERROR = 2  # DMD interface error, latched until AlpDevHalt().


# ControlValues for the types above.
ALP_LEVEL_HIGH = 2006
ALP_LEVEL_LOW = 2007
ALP_EDGE_FALLING = 2008
ALP_EDGE_RISING = 2009
ALP_DATA_LSB_ALIGN = 1
ALP_MASTER = 2301
ALP_PROJ_IDLE = 1201
ALP_SLAVE = 2302


def _load_lib(sdk_path=None):
    """
    Load ``alp51.dll`` and declare the signatures of the functions used here.

    Parameters
    ----------
    sdk_path : str OR None
        Path to ``alp51.dll``, or to the directory containing it.
        Defaults to the ALP-5.1 install location.

    Returns
    -------
    ctypes.CDLL
        The loaded library. Loaded once per path.
    """
    if sdk_path is None:
        sdk_path = _DEFAULT_ALP_PATH
    elif os.path.isdir(sdk_path):
        sdk_path = os.path.join(sdk_path, "alp51.dll")

    if sdk_path in _alp_libs:
        return _alp_libs[sdk_path]

    try:
        lib = ctypes.CDLL(sdk_path)
    except OSError as e:
        raise RuntimeError(
            f"Could not load the ViALUX ALP library at '{sdk_path}'. "
            "Install ViALUX's ALP-5.1 software, or pass sdk_path to its alp51.dll."
        ) from e

    alp_id = ctypes.c_uint32
    i32 = ctypes.c_int32
    p_i32 = ctypes.POINTER(ctypes.c_int32)
    p_id = ctypes.POINTER(ctypes.c_uint32)

    signatures = {
        "AlpDevAlloc": [i32, i32, p_id],
        "AlpDevHalt": [alp_id],
        "AlpDevFree": [alp_id],
        "AlpDevControl": [alp_id, i32, i32],
        "AlpDevInquire": [alp_id, i32, p_i32],
        "AlpSeqAlloc": [alp_id, i32, i32, p_id],
        "AlpSeqFree": [alp_id, alp_id],
        "AlpSeqControl": [alp_id, alp_id, i32, i32],
        "AlpSeqTiming": [alp_id, alp_id, i32, i32, i32, i32, i32],
        "AlpSeqInquire": [alp_id, alp_id, i32, p_i32],
        "AlpSeqPut": [alp_id, alp_id, i32, i32, ctypes.c_void_p],
        "AlpProjStart": [alp_id, alp_id],
        "AlpProjControl": [alp_id, i32, i32],
        "AlpProjInquire": [alp_id, i32, p_i32],
    }

    for name, argtypes in signatures.items():
        function = getattr(lib, name)
        function.argtypes = argtypes
        function.restype = i32

    _alp_libs[sdk_path] = lib

    return lib


def _error_name(ret):
    """The name of an ALP return code, e.g. ``"ALP_NOT_ONLINE"``."""
    try:
        return f"ALP_{_ALPReturn(ret).name}"
    except ValueError:
        return f"unknown error {ret}"


def _check(ret, call):
    """
    Raise if an ALP function did not return ``ALP_OK``.

    Parameters
    ----------
    ret : int
        Return code of the ALP function.
    call : str
        Name of the function, for the error message.
    """
    if ret != _ALPReturn.OK:
        raise RuntimeError(f"ViALUX {call} failed: {_error_name(ret)}.")


class PLM(_PLMModel, SLM):
    """
    Interfaces with ViALUX's Phase Light Modulators (PLMs), such as the V-9903P.

    Patterns are double-buffered between two ALP sequences. Each write uploads to the
    sequence that is not displayed, then displays it as a single picture, after which the
    ALP idles and the PLM holds the pattern without refreshing. The PLM therefore never
    displays a partially-uploaded pattern, and switches within :attr:`picture_time_us` of
    the upload, which defaults to the fastest that the ALP supports.

    Attributes
    ----------
    model_config : dict
        Model configuration from texasinstruments.yaml.
    serial : int
        Serial number of the ALP device.
    dmd_type : int
        DMD type code reported by the ALP.
    switch_timeout_s : float
        How long to wait, beyond :attr:`picture_time_us`, for the ALP to display a
        written pattern before raising. Relevant with :meth:`set_input_trigger`, where
        display awaits a trigger.
    """

    def __init__(
        self,
        serial=None,
        model_name=None,
        picture_time_us=None,
        bias_voltage_mv=None,
        sdk_path=None,
        **kwargs,
    ):
        """
        Initialize the PLM interface.

        Parameters
        ----------
        serial : int OR None
            Serial number of the ALP device to open. Use :meth:`info` to list them.
            If ``None``, opens the first available device.
        model_name : str OR None
            Model identifier from ``texasinstruments.yaml`` of the PLM chip which the ALP
            drives (e.g. ``"p99"``). If ``None``, the model is identified from the DMD
            type which the ALP reports.
        picture_time_us : int OR None
            See :attr:`picture_time_us`. If ``None``, uses the minimum which the ALP
            supports.
        bias_voltage_mv : int OR None
            If given, sets the PLM bias voltage; see :meth:`set_bias_voltage`.
            Otherwise, applies the ALP's default bias.
        sdk_path : str OR None
            Path to ``alp51.dll``, or to the directory containing it.
            Defaults to the ALP-5.1 install location.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options. ``settle_time_s``
            defaults to zero, as writes return once the pattern is displayed.
        """
        self._lib = _load_lib(sdk_path)
        self._device = None
        self._sequences = []
        self._active = None  # Index into _sequences of the displayed one.
        self._input_trigger = False
        self._picture_time_us = None
        self._synch_delay_us = ALP_DEFAULT
        self._synch_pulse_width_us = ALP_DEFAULT
        self.switch_timeout_s = 5.0

        device = ctypes.c_uint32(ALP_INVALID_ID)
        _check(
            self._lib.AlpDevAlloc(
                ALP_DEFAULT if serial is None else int(serial),
                ALP_DEFAULT,
                ctypes.byref(device),
            ),
            "AlpDevAlloc",
        )
        self._device = device

        try:
            self.serial = self.dev_inquire(ALPControl.DEVICE_NUMBER)
            self.dmd_type = self.dev_inquire(ALPControl.DEV_DMDTYPE)
            width = self.dev_inquire(ALPControl.DEV_DISPLAY_WIDTH)
            height = self.dev_inquire(ALPControl.DEV_DISPLAY_HEIGHT)

            logger.debug(
                "ALP %s opened: DMD type %s, %sx%s.", self.serial, self.dmd_type, width, height
            )

            if model_name is None:
                if self.dmd_type not in _DMD_TYPE_MODELS:
                    raise ValueError(
                        f"DMD type {self.dmd_type} is not a supported PLM. Supported DMD "
                        f"types: {_DMD_TYPE_MODELS}. If this is a PLM, pass its model_name "
                        "to open it regardless."
                    )
                model_name = _DMD_TYPE_MODELS[self.dmd_type]
            elif _DMD_TYPE_MODELS.get(self.dmd_type) != model_name:
                logger.warning(
                    "ALP %s reports DMD type %s, which is not model '%s'.",
                    self.serial,
                    self.dmd_type,
                    model_name,
                )
            self.model_config = self.load_model_config(model_name)

            model_shape = tuple(self.model_config["shape"])
            if model_shape != (height, width):
                raise ValueError(
                    f"ALP {self.serial} reports shape {(height, width)}, but model "
                    f"'{model_name}' expects {model_shape}."
                )

            kwargs.setdefault("settle_time_s", 0)
            super().__init__(
                (width, height),
                bitdepth=self._model_bitdepth,
                pitch_um=tuple(np.array(self.model_config["pitch"]) * 1e6),
                name=kwargs.pop("name", f"{model_name}_{self.serial}"),
                **kwargs,
            )

            self.set_gamma()

            # The double buffer: single-picture sequences of one byte per pixel, with the
            # phase level in the least significant bits.
            for _ in range(2):
                sequence = ctypes.c_uint32(ALP_INVALID_ID)
                _check(
                    self._lib.AlpSeqAlloc(self._device, self.bitdepth, 1, ctypes.byref(sequence)),
                    "AlpSeqAlloc",
                )
                self._sequences.append(sequence)
                _check(
                    self._lib.AlpSeqControl(
                        self._device, sequence, int(ALPControl.DATA_FORMAT), ALP_DATA_LSB_ALIGN
                    ),
                    "AlpSeqControl",
                )

            if picture_time_us is None:
                picture_time_us = self.seq_inquire(ALPControl.MIN_PICTURE_TIME)
            self.picture_time_us = picture_time_us

            # Opening resets the reported bias, but the PLM applies the last bias written.
            if bias_voltage_mv is None:
                bias_voltage_mv = self.get_bias_voltage()
            self.set_bias_voltage(bias_voltage_mv)

            self.set_phase(None)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def info(verbose=True, sdk_path=None):
        """
        Discovers the available ALP devices.

        The ALP API cannot enumerate devices, so each available device is opened in
        turn, identified, and closed again. Devices opened elsewhere are not listed.

        Parameters
        ----------
        verbose : bool
            Whether or not to print device information.
        sdk_path : str OR None
            See :meth:`__init__`.

        Returns
        -------
        list of (int, int) tuples
            The serial number and DMD type of each available device.
        """
        lib = _load_lib(sdk_path)

        devices = []
        device_list = []

        try:
            while True:
                device = ctypes.c_uint32(ALP_INVALID_ID)
                ret = lib.AlpDevAlloc(ALP_DEFAULT, ALP_DEFAULT, ctypes.byref(device))
                if ret in (_ALPReturn.NOT_ONLINE, _ALPReturn.NOT_READY):
                    break
                _check(ret, "AlpDevAlloc")
                devices.append(device)

                values = []
                for inquire in (ALPControl.DEVICE_NUMBER, ALPControl.DEV_DMDTYPE):
                    value = ctypes.c_int32(0)
                    _check(
                        lib.AlpDevInquire(device, inquire, ctypes.byref(value)),
                        "AlpDevInquire",
                    )
                    values.append(value.value)

                device_list.append(tuple(values))
        finally:
            for device in devices:
                lib.AlpDevFree(device)

        if verbose:
            for serial, dmd_type in device_list:
                print(f"{serial} (DMD type {dmd_type})")
            if not device_list:
                print("No ALP devices found.")

        return device_list

    def close(self, park=False):
        """
        Close the PLM, stopping projection and releasing the ALP device.

        Parameters
        ----------
        park : bool
            Whether to release the mirrors to their flat state before closing;
            see :meth:`park`. Defaults to ``False``.
        """
        if self._device is None:
            return

        def release(call, *args):
            ret = getattr(self._lib, call)(self._device, *args)
            if ret != _ALPReturn.OK:
                logger.warning("ViALUX %s failed while closing: %s.", call, _error_name(ret))

        try:
            release("AlpDevHalt")
            if park:
                release("AlpDevControl", int(ALPControl.DEV_DMD_MODE), int(DMDMode.POWER_FLOAT))
            for sequence in self._sequences:
                release("AlpSeqFree", sequence)
            release("AlpDevFree")
        finally:
            self._sequences = []
            self._active = None
            self._device = None

    # Sequences

    @property
    def picture_time_us(self):
        """
        Duration of the single picture that displays each written pattern, in
        microseconds, which a blocking write waits for. The PLM holds the pattern
        afterwards.
        """
        return self._picture_time_us

    @picture_time_us.setter
    def picture_time_us(self, value):
        previous = self._picture_time_us
        self._picture_time_us = int(value)

        try:
            self._wait_for_idle()
            self._apply_timing()
        except RuntimeError:
            self._picture_time_us = previous
            raise

    def _apply_timing(self):
        """
        Apply :attr:`picture_time_us` and the synch output settings to both sequences.
        These take effect when a sequence is next started.
        """
        for sequence in self._sequences:
            _check(
                self._lib.AlpSeqTiming(
                    self._device,
                    sequence,
                    ALP_DEFAULT,  # Illuminate time: the full picture time.
                    self._picture_time_us,
                    int(self._synch_delay_us),
                    int(self._synch_pulse_width_us),
                    ALP_DEFAULT,  # Trigger in delay.
                ),
                "AlpSeqTiming",
            )

    def _wait_for_idle(self):
        """Wait until the ALP has displayed every started pattern and idles."""
        timeout_s = self.switch_timeout_s + 1e-6 * self._picture_time_us
        deadline = time.perf_counter() + timeout_s

        # No sleep: each inquiry is already a round trip to the ALP.
        while self.proj_inquire(ALPControl.PROJ_STATE) != ALP_PROJ_IDLE:
            if time.perf_counter() > deadline:
                raise RuntimeError(
                    f"PLM did not finish displaying a pattern within {timeout_s:.3g} s."
                    + (" Is the input trigger firing?" if self._input_trigger else "")
                )

    def _halt(self):
        """Stop projection immediately, clearing the mirrors."""
        _check(self._lib.AlpDevHalt(self._device), "AlpDevHalt")

    @contextmanager
    def _halted(self):
        """Halt projection, clearing the mirrors, for the context; then redisplay."""
        self._halt()
        try:
            yield
        finally:
            if self._active is not None:
                _check(
                    self._lib.AlpProjStart(self._device, self._sequences[self._active]),
                    "AlpProjStart",
                )

    def _set_phase_hw(self, display, block=True):
        """
        Hardware-specific implementation. See :meth:`.SLM._set_phase_hw`.

        Parameters
        ----------
        display
            Phase levels to display on the PLM.
        block : bool
            Whether to wait until the PLM displays ``display`` before returning.
            If ``False``, or while :meth:`set_input_trigger` is enabled, returns once the
            pattern is uploaded; the next write still waits for this one to be displayed
            before uploading.
        """
        display = np.ascontiguousarray(as_numpy(display), dtype=np.uint8)

        # The idle sequence must not still be displayed, or queued for display.
        self._wait_for_idle()
        index = 0 if self._active is None else 1 - self._active
        sequence = self._sequences[index]

        _check(
            self._lib.AlpSeqPut(
                self._device, sequence, 0, 1, display.ctypes.data_as(ctypes.c_void_p)
            ),
            "AlpSeqPut",
        )
        _check(self._lib.AlpProjStart(self._device, sequence), "AlpProjStart")
        self._active = index

        if block and not self._input_trigger:
            self._wait_for_idle()

    # Triggering

    def set_input_trigger(self, on=False, edge="rising"):
        r"""
        Configures the input trigger (pin 7 of the ALP's synchronization connector): a
        written pattern is displayed upon the next trigger edge (``ALP_SLAVE`` mode).

        While enabled, writes do not block, and a write waits up to
        :attr:`switch_timeout_s` for the previous pattern to be triggered.

        Parameters
        ----------
        on : bool
            Whether to enable the input trigger.
        edge : str
            ``"rising"`` or ``"falling"``: the edge that triggers.
        """
        values = {"rising": ALP_EDGE_RISING, "falling": ALP_EDGE_FALLING}
        if edge not in values:
            raise ValueError(f"Unknown edge '{edge}'. Valid: {list(values)}")

        self._wait_for_idle()
        if on:
            self.dev_control(ALPControl.TRIGGER_EDGE, values[edge])
        self._proj_control(ALPControl.PROJ_MODE, ALP_SLAVE if on else ALP_MASTER)
        self._input_trigger = bool(on)

    def set_output_trigger(self, on=False, polarity="high", delay_us=0, pulse_width_us=None):
        r"""
        Configures the frame synch output of the ALP, which pulses at the start of every
        displayed pattern.

        Note
        ~~~~
        The ALP always drives this output; it cannot be disabled. ``on=False`` restores
        the default configuration, ignoring the other arguments.

        Parameters
        ----------
        on : bool
            Whether to apply the configuration below, or to restore the defaults.
        polarity : str
            ``"high"`` or ``"low"``: the level of the active pulse.
        delay_us : int
            Delay from the start of the pulse to the start of the picture, in
            microseconds.
        pulse_width_us : int OR None
            Duration of the pulse, in microseconds. If ``None``, uses the ALP's default.
        """
        values = {"high": ALP_LEVEL_HIGH, "low": ALP_LEVEL_LOW}
        if polarity not in values:
            raise ValueError(f"Unknown polarity '{polarity}'. Valid: {list(values)}")

        previous = (self._synch_delay_us, self._synch_pulse_width_us)
        if on:
            self._synch_delay_us = int(delay_us)
            self._synch_pulse_width_us = (
                ALP_DEFAULT if pulse_width_us is None else int(pulse_width_us)
            )
        else:
            polarity = "high"
            self._synch_delay_us = ALP_DEFAULT
            self._synch_pulse_width_us = ALP_DEFAULT

        # The ALP refuses to change the polarity while projecting.
        try:
            self._wait_for_idle()
            self._apply_timing()
            self.dev_control(ALPControl.SYNCH_POLARITY, values[polarity])
        except RuntimeError:
            (self._synch_delay_us, self._synch_pulse_width_us) = previous
            raise

    # Bias voltage

    def set_bias_voltage(self, voltage_mv):
        """
        Set the PLM bias voltage (``ALP_PLM_BIAS_VOLTAGE``), which scales the mirrors'
        displacement and thus the phase stroke.

        Parameters
        ----------
        voltage_mv : int
            Bias voltage in millivolts. Must lie within :meth:`get_bias_voltage_range`.
        """
        voltage_mv = int(voltage_mv)
        (minimum, maximum) = self.get_bias_voltage_range()

        if not minimum <= voltage_mv <= maximum:
            raise ValueError(f"Bias voltage {voltage_mv} mV is outside [{minimum}, {maximum}] mV.")

        self.dev_control(ALPControl.PLM_BIAS_VOLTAGE, voltage_mv)

    def get_bias_voltage(self):
        """
        Get the PLM bias voltage.

        Returns
        -------
        int
            Bias voltage in millivolts.
        """
        return self.dev_inquire(ALPControl.PLM_BIAS_VOLTAGE)

    def get_bias_voltage_range(self):
        """
        Get the range of the PLM bias voltage.

        Returns
        -------
        (int, int)
            Minimum and maximum bias voltage in millivolts.
        """
        return (
            self.dev_inquire(ALPControl.PLM_MIN_BIAS_VOLTAGE),
            self.dev_inquire(ALPControl.PLM_MAX_BIAS_VOLTAGE),
        )

    # Status

    def get_temperature(self):
        """
        Read the temperatures of the PLM and the ALP controller.

        Returns
        -------
        dict
            Temperatures in degrees Celsius, keyed by ``"dmd"``, ``"dmd_pcb"``, ``"apps_fpga"``,
            ``"ddc_fpga"``, and ``"pcb"``. The recommended maxima are keyed by
            ``"apps_fpga_max"``, ``"ddc_fpga_max"``, and ``"pcb_max"``.
        """
        sensors = {
            "dmd": (ALPControl.DMD_TEMPERATURE, 256),
            "dmd_pcb": (ALPControl.DMD_PCB_TEMPERATURE, 256),
            "apps_fpga": (ALPControl.APPS_FPGA_TEMPERATURE, 256),
            "ddc_fpga": (ALPControl.DDC_FPGA_TEMPERATURE, 256),
            "pcb": (ALPControl.PCB_TEMPERATURE, 256),
            "apps_fpga_max": (ALPControl.MAX_APPS_FPGA_TEMPERATURE, 1),
            "ddc_fpga_max": (ALPControl.MAX_DDC_FPGA_TEMPERATURE, 1),
            "pcb_max": (ALPControl.MAX_PCB_TEMPERATURE, 1),
        }

        return {key: self.dev_inquire(inquire) / scale for key, (inquire, scale) in sensors.items()}

    def get_dmd_mode(self):
        """
        Get the state of the DMD (``ALP_DEV_DMD_MODE``). See the caution in the module
        documentation on :attr:`DMDMode.INTERFACE_ERROR`.

        Returns
        -------
        DMDMode OR int
            The state, or the raw value if ``alp.h`` does not define it.
        """
        mode = self.dev_inquire(ALPControl.DEV_DMD_MODE)
        try:
            return DMDMode(mode)
        except ValueError:
            return mode

    def reset_interface(self):
        """
        Clear a latched :attr:`DMDMode.INTERFACE_ERROR` with ``AlpDevHalt()``, as ViALUX
        advises, and redisplay the current pattern. The halt briefly clears the mirrors,
        and the redisplay awaits a trigger while :meth:`set_input_trigger` is enabled.

        Returns
        -------
        DMDMode OR int
            The state after the reset, before the pattern is redisplayed.
        """
        with self._halted():
            mode = self.get_dmd_mode()

        return mode

    def park(self):
        """
        Stop projection and release the mirrors to their flat state
        (``ALP_DMD_POWER_FLOAT``). Settings and memory are kept, but nothing is displayed
        until :meth:`resume`.
        """
        self._halt()
        self.dev_control(ALPControl.DEV_DMD_MODE, DMDMode.POWER_FLOAT)

    def resume(self):
        """
        Wake the mirrors after :meth:`park` (``ALP_DMD_RESUME``) and redisplay the current
        pattern.
        """
        with self._halted():
            self.dev_control(ALPControl.DEV_DMD_MODE, DMDMode.RESUME)

    # Raw access

    def dev_control(self, control_type, value):
        """
        Error-checked ``AlpDevControl``, for settings without a dedicated method.

        Parameters
        ----------
        control_type : ALPControl OR int
            ``ControlType``; see the ALP-5.1 API description.
        value : int
            ``ControlValue``.
        """
        _check(
            self._lib.AlpDevControl(self._device, int(control_type), int(value)),
            "AlpDevControl",
        )

    def dev_inquire(self, inquire_type):
        """
        Error-checked ``AlpDevInquire``.

        Parameters
        ----------
        inquire_type : ALPControl OR int
            ``InquireType``; see the ALP-5.1 API description.

        Returns
        -------
        int
            The inquired value.
        """
        value = ctypes.c_int32(0)
        _check(
            self._lib.AlpDevInquire(self._device, int(inquire_type), ctypes.byref(value)),
            "AlpDevInquire",
        )
        return value.value

    def seq_inquire(self, inquire_type):
        """
        Error-checked ``AlpSeqInquire`` of the displayed sequence (or the first, before
        any display).

        Parameters
        ----------
        inquire_type : ALPControl OR int
            ``InquireType``; see the ALP-5.1 API description.

        Returns
        -------
        int
            The inquired value.
        """
        sequence = self._sequences[0 if self._active is None else self._active]
        value = ctypes.c_int32(0)
        _check(
            self._lib.AlpSeqInquire(self._device, sequence, int(inquire_type), ctypes.byref(value)),
            "AlpSeqInquire",
        )
        return value.value

    def _proj_control(self, control_type, value):
        """Error-checked ``AlpProjControl``, which requires an idle ALP."""
        _check(
            self._lib.AlpProjControl(self._device, int(control_type), int(value)),
            "AlpProjControl",
        )

    def proj_inquire(self, inquire_type):
        """
        Error-checked ``AlpProjInquire``.

        Parameters
        ----------
        inquire_type : ALPControl OR int
            ``InquireType``; see the ALP-5.1 API description.

        Returns
        -------
        int
            The inquired value.
        """
        value = ctypes.c_int32(0)
        _check(
            self._lib.AlpProjInquire(self._device, int(inquire_type), ctypes.byref(value)),
            "AlpProjInquire",
        )
        return value.value
