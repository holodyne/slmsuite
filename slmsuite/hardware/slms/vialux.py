r"""
Hardware control for ViALUX Phase Light Modulators (PLMs).
Tested with the V-9903P (Texas Instruments DLPM990 on the ALP-5.1 controller).

Unlike a :class:`~slmsuite.hardware.slms.texasinstruments.PLM`, which is driven as a
display, a ViALUX PLM is driven by the ALP controller: patterns are uploaded over USB
(or PCIe) into the controller's memory, which displays them and maps each phase level
onto the PLM's electrodes itself. The controller also stores frames and plays them as
sequences at a fixed frame time; see :meth:`.SLM.set_frame`, :meth:`.SLM.set_phase` and
:meth:`.SLM.set_frametime`.

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

    # Store frames, then play them as a sequence.
    for i, phase in enumerate(phases):
        plm.set_frame(i, phase)
    plm.set_frametime(1e-3)
    plm.set_phase(frame=range(len(phases)), repeat=0)  # Loop until the next write.
    plm.set_phase(frame=0)

Note
~~~~
:class:`PLM` requires ViALUX's ALP-5.1 software, whose ``alp51.dll`` is loaded from
``C:\\Program Files\\ALP-5.1\\ALP-5.1 API\\x64\\`` by default. Pass ``sdk_path`` to
load it from elsewhere. The constants used here are those of the ALP-5.1 ``alp.h``
(header version 29 or later, which adds ``ALP_DMDTYPE_2K_099A_PLM`` and the
``ALP_PLM_*_BIAS_VOLTAGE`` controls) and the ALP-5.1 API description.

Tip
~~~
The PLM warms while the device is open and cools while it is closed, and both the
pointing of the beam and its wavefront follow the temperature
(:meth:`PLM.get_temperature`). Leave the device open until the temperature settles
before calibrating, or repeat the Fourier and wavefront calibrations once it has. The
pixel calibration does not depend on the temperature.

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
from enum import IntEnum
from typing import ClassVar

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
    to the private raw access methods (``PLM._dev_control`` and its siblings) as a plain
    ``int``.
    """

    # AlpDevControl / AlpDevInquire, by section of the API description.
    DEVICE_NUMBER = 2000  #: 2.3 -- Serial number
    AVAIL_MEMORY = 2003  #: 2.3 -- Free sequence memory, in frames
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
    SEQ_REPEAT = 2100  #: 2.8 -- Passes of a finite start
    FIRSTFRAME = 2101  #: 2.8 -- First frame of the sequence to display
    LASTFRAME = 2102  #: 2.8 -- Last frame of the sequence to display
    DATA_FORMAT = 2110  #: 2.8 -- Data format and alignment
    FLUT_MODE = 2118  #: 3.4 -- Frame lookup table mode
    FLUT_ENTRIES9 = 2120  #: 3.4 -- Frame lookup table entries used, counted as 9-bit
    PICTURE_TIME = 2203  #: 2.10 -- Frame time, in microseconds
    MIN_PICTURE_TIME = 2211  #: 2.10 -- Minimum frame time, in microseconds

    # AlpProjControl / AlpProjInquire / AlpProjControlEx
    PROJ_MODE = 2300  #: 2.13 -- Master or slave mode
    FLUT_MAX_ENTRIES9 = 2324  #: 3.4 -- Size of the frame lookup table, in 9-bit entries
    FLUT_WRITE_18BIT = 2326  #: 3.4 -- Transfer 18-bit lookup table entries
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
ALP_FLUT_NONE = 0
ALP_FLUT_18BIT = 2
ALP_MASTER = 2301
ALP_PROJ_IDLE = 1201
ALP_SLAVE = 2302

_SEQ_REPEAT_MAX = 1048576  # Section 2.8 of the ALP-5.1 API description.


class _FlutWrite(ctypes.Structure):
    """``tFlutWrite`` of ``alp.h``: frame lookup table entries to transfer."""

    _fields_: ClassVar = [
        ("nOffset", ctypes.c_int32),
        ("nSize", ctypes.c_int32),
        ("FrameNumbers", ctypes.c_uint32 * 4096),
    ]


class PLM(_PLMModel, SLM):
    """
    Interfaces with ViALUX's Phase Light Modulators (PLMs), such as the V-9903P.

    The ALP's memory holds :attr:`frame_count` frames, which :meth:`set_frame` writes and
    :meth:`set_phase` displays, singly or as a sequence in any order at the frame time
    of :meth:`set_frametime`. A direct write uploads to a reserved frame and displays it.
    Either way, the ALP shows each frame for one frame time and then idles while the
    PLM holds the last one, so a partially uploaded pattern is never displayed. Any
    write or setting ends a sequence in progress after its current pass.

    Attributes
    ----------
    model_config : dict
        Model configuration from texasinstruments.yaml.
    serial : int
        Serial number of the ALP device.
    dmd_type : int
        DMD type code reported by the ALP.
    frame_count : int
        Number of frames in the frame memory, numbered ``0`` to ``frame_count - 1``.
    switch_timeout_s : float
        How long to wait, beyond the time that the started frames take to display, for
        the ALP to finish before raising. Relevant with :meth:`set_input_trigger`, where
        each frame awaits a trigger.
    """

    def __init__(
        self,
        serial=None,
        model_name=None,
        frame_count=None,
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
        frame_count : int OR None
            See :attr:`frame_count`. If ``None``, reserves as many frames as the ALP's free
            memory holds.
        sdk_path : str OR None
            Path to ``alp51.dll``, or to the directory containing it.
            Defaults to the ALP-5.1 install location.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options. ``settle_time_s``
            defaults to zero, as writes return once the pattern is displayed, and
            ``pixel`` defaults to that of the model, if its entry has one.
        """
        self._lib = self._load_lib(sdk_path)
        self._device = None
        self._sequence = None
        self._held = None  # Frame that the PLM holds once the ALP idles.
        self._looping = False
        self._pass_s = 0.0  # Time that one pass of the started frames takes.
        self._duration_s = 0.0  # Longest time that the ALP takes to idle.
        self._input_trigger = False
        self._frametime_s = None
        self._min_frametime_s = None
        self._flut_entries = 0
        self._synch_us = (ALP_DEFAULT, ALP_DEFAULT)  # Delay and pulse width.
        self.switch_timeout_s = 5.0

        device = ctypes.c_uint32(ALP_INVALID_ID)
        self._check(
            self._lib.AlpDevAlloc(
                ALP_DEFAULT if serial is None else int(serial),
                ALP_DEFAULT,
                ctypes.byref(device),
            ),
            "AlpDevAlloc",
        )
        self._device = device

        try:
            self.serial = self._dev_inquire(ALPControl.DEVICE_NUMBER)
            self.dmd_type = self._dev_inquire(ALPControl.DEV_DMDTYPE)
            width = self._dev_inquire(ALPControl.DEV_DISPLAY_WIDTH)
            height = self._dev_inquire(ALPControl.DEV_DISPLAY_HEIGHT)

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
                pixel=kwargs.pop("pixel", self._model_pixel),
                name=kwargs.pop("name", f"{model_name}_{self.serial}"),
                **kwargs,
            )

            self.set_gamma()

            # One sequence of one byte per pixel, with the phase level in the least
            # significant bits: the frame memory plus a scratch frame for direct writes.
            if frame_count is None:
                frame_count = self._dev_inquire(ALPControl.AVAIL_MEMORY) - 1
            self.frame_count = int(frame_count)
            if self.frame_count < 0:
                raise ValueError(f"frame_count must be non-negative; got {frame_count}.")
            self._scratch = self.frame_count

            sequence = ctypes.c_uint32(ALP_INVALID_ID)
            ret = self._lib.AlpSeqAlloc(
                self._device, self.bitdepth, self.frame_count + 1, ctypes.byref(sequence)
            )
            if ret == _ALPReturn.MEMORY_FULL:
                raise RuntimeError(
                    f"The ALP cannot hold {self.frame_count} frames; pass a smaller frame_count."
                )
            self._check(ret, "AlpSeqAlloc")
            self._sequence = sequence
            self._seq_control(ALPControl.DATA_FORMAT, ALP_DATA_LSB_ALIGN)
            self._flut_entries = self._proj_inquire(ALPControl.FLUT_MAX_ENTRIES9)
            logger.debug("ALP %s holds %s frames.", self.serial, self.frame_count)

            self._min_frametime_s = self._seq_inquire(ALPControl.MIN_PICTURE_TIME) / 1e6
            self._frametime_s = self._min_frametime_s

            # Opening resets the reported bias, but the PLM applies the last bias written.
            self.set_bias_voltage(self.get_bias_voltage())

            self.set_phase(None)
        except BaseException:
            self.close()
            raise

    @staticmethod
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
            "AlpProjStartCont": [alp_id, alp_id],
            "AlpProjHalt": [alp_id],
            "AlpProjControl": [alp_id, i32, i32],
            "AlpProjControlEx": [alp_id, i32, ctypes.c_void_p],
            "AlpProjInquire": [alp_id, i32, p_i32],
        }

        for name, argtypes in signatures.items():
            function = getattr(lib, name)
            function.argtypes = argtypes
            function.restype = i32

        _alp_libs[sdk_path] = lib

        return lib

    @staticmethod
    def _error_name(ret):
        """The name of an ALP return code, e.g. ``"ALP_NOT_ONLINE"``."""
        try:
            return f"ALP_{_ALPReturn(ret).name}"
        except ValueError:
            return f"unknown error {ret}"

    @staticmethod
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
            raise RuntimeError(f"ViALUX {call} failed: {PLM._error_name(ret)}.")

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
        lib = PLM._load_lib(sdk_path)

        devices = []
        device_list = []

        try:
            while True:
                device = ctypes.c_uint32(ALP_INVALID_ID)
                ret = lib.AlpDevAlloc(ALP_DEFAULT, ALP_DEFAULT, ctypes.byref(device))
                if ret in (_ALPReturn.NOT_ONLINE, _ALPReturn.NOT_READY):
                    break
                PLM._check(ret, "AlpDevAlloc")
                devices.append(device)

                values = []
                for inquire in (ALPControl.DEVICE_NUMBER, ALPControl.DEV_DMDTYPE):
                    value = ctypes.c_int32(0)
                    PLM._check(
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
                logger.warning("ViALUX %s failed while closing: %s.", call, self._error_name(ret))

        try:
            release("AlpDevHalt")
            if park:
                release("AlpDevControl", int(ALPControl.DEV_DMD_MODE), int(DMDMode.POWER_FLOAT))
            if self._sequence is not None:
                release("AlpSeqFree", self._sequence)
            release("AlpDevFree")
        finally:
            self._sequence = None
            self._device = None

    # Frames

    def set_frametime(self, frametime_s):
        """
        Set the time that each frame of a sequence lasts, from the next sequence
        started. See :meth:`.SLM.set_frametime`. A single frame always switches in the
        minimum frame time; either way, a frame lasts its frame time before the next
        write is displayed.

        Parameters
        ----------
        frametime_s : float
            Frame time in seconds, rounded to the ALP's microseconds. Must be at least
            the minimum which the ALP supports, the default at open.

        Returns
        -------
        float
            The resulting frame time in seconds, from :meth:`get_frametime`.
        """
        frametime_s = round(float(frametime_s) * 1e6) / 1e6
        if frametime_s < self._min_frametime_s:
            raise ValueError(
                f"Frame time {frametime_s:.3g} s is below the minimum {self._min_frametime_s:.3g} s."
            )
        self._frametime_s = frametime_s

        return self.get_frametime()

    def get_frametime(self):
        """Frame time of sequences, in seconds. See :meth:`.SLM.get_frametime`."""
        return self._frametime_s

    def _apply_timing(self, frametime_s, synch_us=None):
        """Apply a frame time and the synch output delay and pulse width to the next start."""
        (delay_us, pulse_width_us) = self._synch_us if synch_us is None else synch_us
        self._check(
            self._lib.AlpSeqTiming(
                self._device,
                self._sequence,
                ALP_DEFAULT,  # Illuminate time: the full frame time.
                round(frametime_s * 1e6),
                delay_us,
                pulse_width_us,
                ALP_DEFAULT,  # Trigger in delay.
            ),
            "AlpSeqTiming",
        )

    def _wait_for_idle(self):
        """Wait until the ALP displays the last of the started frames and idles."""
        timeout_s = self.switch_timeout_s + self._duration_s
        deadline = time.perf_counter() + timeout_s

        # No sleep: each inquiry is already a round trip to the ALP.
        while self._proj_inquire(ALPControl.PROJ_STATE) != ALP_PROJ_IDLE:
            if time.perf_counter() > deadline:
                raise RuntimeError(
                    f"PLM did not finish displaying within {timeout_s:.3g} s."
                    + (" Is the input trigger firing?" if self._input_trigger else "")
                )

    def _stop(self):
        """End a sequence after its current pass, then wait for the ALP to idle."""
        self._check(self._lib.AlpProjHalt(self._device), "AlpProjHalt")
        self._looping = False
        self._wait_for_idle()

    def _halt(self):
        """Stop projection immediately, clearing the mirrors."""
        self._check(self._lib.AlpDevHalt(self._device), "AlpDevHalt")
        self._looping = False
        self._pass_s = 0.0

    def _redisplay(self):
        """Display the held frame again, as after :meth:`_halt`."""
        self._start(np.array([self._held]))

    def _put(self, display, frame):
        """Upload levels, or a stack of them, into the sequence starting at frame ``frame``."""
        display = np.ascontiguousarray(as_numpy(display), dtype=np.uint8)
        count = len(display) if display.ndim == 3 else 1
        data = display.ctypes.data_as(ctypes.c_void_p)
        deadline = time.perf_counter() + self.switch_timeout_s + self._pass_s

        # The ALP idles at once when a sequence is ended, but refuses uploads until its pass is over.
        while True:
            ret = self._lib.AlpSeqPut(self._device, self._sequence, int(frame), count, data)
            if ret != _ALPReturn.SEQ_IN_USE or time.perf_counter() > deadline:
                break
        self._check(ret, "AlpSeqPut")

    def _start(self, order, repeat=1, block=True):
        """
        Display the frames ``order``, in order, ``repeat`` times, or until stopped if
        ``repeat`` is zero; then wait for the last frame if ``block``, unless looping or
        awaiting triggers.
        """
        single = len(order) == 1 and repeat == 1
        frametime_s = self._min_frametime_s if single else self._frametime_s
        self._apply_timing(frametime_s)

        # The ALP keeps the first frame at or before the last at every step.
        (first, last) = (int(order.min()), int(order.max()))
        self._seq_control(ALPControl.FIRSTFRAME, 0)
        self._seq_control(ALPControl.LASTFRAME, last)
        self._seq_control(ALPControl.FIRSTFRAME, first)

        if np.all(np.diff(order) == 1):
            self._seq_control(ALPControl.FLUT_MODE, ALP_FLUT_NONE)
        else:
            # Any other order goes through the frame lookup table, relative to the first.
            table = _FlutWrite(0, len(order))
            table.FrameNumbers[: len(order)] = (order - first).tolist()
            self._check(
                self._lib.AlpProjControlEx(
                    self._device, int(ALPControl.FLUT_WRITE_18BIT), ctypes.byref(table)
                ),
                "AlpProjControlEx",
            )
            self._seq_control(ALPControl.FLUT_ENTRIES9, 2 * len(order))
            self._seq_control(ALPControl.FLUT_MODE, ALP_FLUT_18BIT)

        if repeat:
            self._seq_control(ALPControl.SEQ_REPEAT, repeat)

        call = "AlpProjStartCont" if repeat == 0 else "AlpProjStart"
        self._check(getattr(self._lib, call)(self._device, self._sequence), call)

        # A pass in progress plays out before these frames.
        self._duration_s = self._pass_s + len(order) * max(repeat, 1) * frametime_s
        self._pass_s = len(order) * frametime_s
        self._held = int(order[-1])
        self._looping = repeat == 0

        if block and repeat and not self._input_trigger:
            self._wait_for_idle()

    def _set_frame_hw(self, display, frame):
        """
        Store levels in frame ``frame`` of the ALP's memory. See :meth:`.SLM._set_frame_hw`.
        A finite sequence in progress finishes first; a looping one must be ended by
        displaying something, as the ALP refuses uploads into a started sequence.
        """
        self._set_frames_hw([display], [frame])

    def _set_frames_hw(self, displays, frames):
        """
        Store several frames as :meth:`_set_frame_hw` does, each run of consecutive
        frames in one upload. See :meth:`.SLM._set_frames_hw`.
        """
        if self._looping:
            raise RuntimeError(
                "Frames cannot be stored while a sequence loops; display a frame first."
            )

        self._wait_for_idle()
        runs = np.split(np.arange(len(frames)), np.flatnonzero(np.diff(frames) != 1) + 1)
        for run in runs:
            self._put(np.stack([as_numpy(displays[i]) for i in run]), frames[run[0]])

    def _set_phase_hw(self, display, block=True):
        """
        Hardware-specific implementation. See :meth:`.SLM._set_phase_hw`.

        Parameters
        ----------
        display
            Phase levels to display on the PLM.
        block : bool
            Whether to wait until the PLM has displayed the frame before returning.
            If ``False``, or while :meth:`set_input_trigger` is enabled, returns once the
            frame is started.
        """
        self._stop()
        self._put(display, self._scratch)
        self._start(np.array([self._scratch]), block=block)

    def _set_phase_frame_hw(self, frame, repeat=1, block=True):
        """
        Display stored frames. See :meth:`.SLM._set_phase_frame_hw`.

        Parameters
        ----------
        frame : int OR numpy.ndarray
            Stored frames to display, in order. An order other than a contiguous
            ascending range plays through the ALP's frame lookup table, which bounds its
            length.
        repeat : int
            Passes of ``frame``, at most 1048576; ``0`` loops until the next write.
        block : bool
            Whether to wait until the PLM has displayed every frame before returning.
            If ``False``, while :meth:`set_input_trigger` is enabled, or when looping,
            returns once the frames are started; the next write ends them after their
            current pass.
        """
        order = np.atleast_1d(frame)
        if repeat > _SEQ_REPEAT_MAX:
            raise ValueError(f"repeat must be at most {_SEQ_REPEAT_MAX}; got {repeat}.")
        if len(order) > self._flut_entries // 2 and np.any(np.diff(order) != 1):
            raise ValueError(
                f"At most {self._flut_entries // 2} frames can be played in arbitrary order; "
                f"got {len(order)}."
            )

        self._stop()
        self._start(order, repeat, block)

    # Triggering

    def set_input_trigger(self, on=False, edge="rising"):
        r"""
        Configures the input trigger (pin 7 of the ALP's synchronization connector): each
        frame is displayed upon the next trigger edge (``ALP_SLAVE`` mode).

        While enabled, writes do not block, and a write waits up to
        :attr:`switch_timeout_s` beyond the frames' duration for the previous frames to
        be triggered. Enabling ends a sequence in progress; disabling halts frames still
        awaiting a trigger and redisplays the held frame.

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

        if on:
            self._stop()
            self._dev_control(ALPControl.TRIGGER_EDGE, values[edge])
            self._proj_control(ALPControl.PROJ_MODE, ALP_SLAVE)
            self._input_trigger = True
        elif self._input_trigger:
            # Frames awaiting a trigger never finish, so halt rather than wait for them.
            self._halt()
            self._proj_control(ALPControl.PROJ_MODE, ALP_MASTER)
            self._input_trigger = False
            self._redisplay()

    def set_output_trigger(self, on=False, polarity="high", delay_s=0, pulse_width_s=None):
        r"""
        Configures the frame synch output of the ALP, which pulses at the start of every
        displayed frame.

        Note
        ~~~~
        The ALP always drives this output; it cannot be disabled. ``on=False`` restores
        the default configuration, ignoring the other arguments. Ends a sequence in
        progress.

        Parameters
        ----------
        on : bool
            Whether to apply the configuration below, or to restore the defaults.
        polarity : str
            ``"high"`` or ``"low"``: the level of the active pulse.
        delay_s : float
            Delay from the start of the pulse to the start of the frame, in seconds.
        pulse_width_s : float OR None
            Duration of the pulse, in seconds, within the minimum frame time. If
            ``None``, uses the ALP's default.
        """
        values = {"high": ALP_LEVEL_HIGH, "low": ALP_LEVEL_LOW}
        if polarity not in values:
            raise ValueError(f"Unknown polarity '{polarity}'. Valid: {list(values)}")

        synch_us = (ALP_DEFAULT, ALP_DEFAULT)
        if on:
            synch_us = (
                round(delay_s * 1e6),
                ALP_DEFAULT if pulse_width_s is None else round(pulse_width_s * 1e6),
            )
        else:
            polarity = "high"

        # The ALP refuses to change the polarity while projecting.
        self._stop()
        self._apply_timing(self._min_frametime_s, synch_us)
        self._synch_us = synch_us
        self._dev_control(ALPControl.SYNCH_POLARITY, values[polarity])

    # Bias voltage

    def set_bias_voltage(self, voltage_v):
        """
        Set the PLM bias voltage (``ALP_PLM_BIAS_VOLTAGE``), which scales the mirrors'
        displacement and thus the phase stroke.

        Parameters
        ----------
        voltage_v : float
            Bias voltage in volts, rounded to the ALP's millivolts. Must lie within
            :meth:`get_bias_voltage_range`.
        """
        voltage_v = float(voltage_v)
        (minimum, maximum) = self.get_bias_voltage_range()

        if not minimum <= voltage_v <= maximum:
            raise ValueError(f"Bias voltage {voltage_v} V is outside [{minimum}, {maximum}] V.")

        self._dev_control(ALPControl.PLM_BIAS_VOLTAGE, round(voltage_v * 1000))

    def get_bias_voltage(self):
        """
        Get the PLM bias voltage.

        Returns
        -------
        float
            Bias voltage in volts.
        """
        return self._dev_inquire(ALPControl.PLM_BIAS_VOLTAGE) / 1000

    def get_bias_voltage_range(self):
        """
        Get the range of the PLM bias voltage.

        Returns
        -------
        (float, float)
            Minimum and maximum bias voltage in volts.
        """
        return (
            self._dev_inquire(ALPControl.PLM_MIN_BIAS_VOLTAGE) / 1000,
            self._dev_inquire(ALPControl.PLM_MAX_BIAS_VOLTAGE) / 1000,
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

        return {
            key: self._dev_inquire(inquire) / scale for key, (inquire, scale) in sensors.items()
        }

    def get_dmd_mode(self):
        """
        Get the state of the DMD (``ALP_DEV_DMD_MODE``). See the caution in the module
        documentation on :attr:`DMDMode.INTERFACE_ERROR`.

        Returns
        -------
        DMDMode OR int
            The state, or the raw value if ``alp.h`` does not define it.
        """
        mode = self._dev_inquire(ALPControl.DEV_DMD_MODE)
        try:
            return DMDMode(mode)
        except ValueError:
            return mode

    def reset_interface(self):
        """
        Clear a latched :attr:`DMDMode.INTERFACE_ERROR` with ``AlpDevHalt()``, as ViALUX
        advises, and redisplay the held frame. The halt briefly clears the mirrors and
        ends a sequence in progress, and the redisplay awaits a trigger while
        :meth:`set_input_trigger` is enabled.

        Returns
        -------
        DMDMode OR int
            The state after the reset, before the frame is redisplayed.
        """
        self._halt()
        mode = self.get_dmd_mode()
        self._redisplay()

        return mode

    def park(self):
        """
        Stop projection and release the mirrors to their flat state
        (``ALP_DMD_POWER_FLOAT``). Settings and memory are kept, but nothing is displayed
        until :meth:`resume`.
        """
        self._halt()
        self._dev_control(ALPControl.DEV_DMD_MODE, DMDMode.POWER_FLOAT)

    def resume(self):
        """
        Wake the mirrors after :meth:`park` (``ALP_DMD_RESUME``) and redisplay the held
        frame.
        """
        self._halt()
        self._dev_control(ALPControl.DEV_DMD_MODE, DMDMode.RESUME)
        self._redisplay()

    # Raw access

    def _dev_control(self, control_type, value):
        """Error-checked ``AlpDevControl``."""
        self._check(
            self._lib.AlpDevControl(self._device, int(control_type), int(value)),
            "AlpDevControl",
        )

    def _dev_inquire(self, inquire_type):
        """Error-checked ``AlpDevInquire``; returns the inquired value."""
        value = ctypes.c_int32(0)
        self._check(
            self._lib.AlpDevInquire(self._device, int(inquire_type), ctypes.byref(value)),
            "AlpDevInquire",
        )
        return value.value

    def _seq_control(self, control_type, value):
        """Error-checked ``AlpSeqControl`` of the sequence."""
        self._check(
            self._lib.AlpSeqControl(self._device, self._sequence, int(control_type), int(value)),
            "AlpSeqControl",
        )

    def _seq_inquire(self, inquire_type):
        """Error-checked ``AlpSeqInquire`` of the sequence; returns the inquired value."""
        value = ctypes.c_int32(0)
        self._check(
            self._lib.AlpSeqInquire(
                self._device, self._sequence, int(inquire_type), ctypes.byref(value)
            ),
            "AlpSeqInquire",
        )
        return value.value

    def _proj_control(self, control_type, value):
        """Error-checked ``AlpProjControl``, which requires an idle ALP."""
        self._check(
            self._lib.AlpProjControl(self._device, int(control_type), int(value)),
            "AlpProjControl",
        )

    def _proj_inquire(self, inquire_type):
        """Error-checked ``AlpProjInquire``; returns the inquired value."""
        value = ctypes.c_int32(0)
        self._check(
            self._lib.AlpProjInquire(self._device, int(inquire_type), ctypes.byref(value)),
            "AlpProjInquire",
        )
        return value.value
