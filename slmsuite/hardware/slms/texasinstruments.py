"""
Hardware control for Texas Instruments Phase Light Modulators (PLMs).

This module provides GPU-accelerated control for TI PLMs via direct
implementation of phase quantization and electrode mapping. Supports both
:mod:`cupy` (GPU) and :mod:`numpy` (CPU) for maximum performance and
compatibility.

.. highlight:: python
.. code-block:: python

    from slmsuite.hardware.slms.texasinstruments import PLM

    # Configure USB and open every connected PLM in one step. This discovers
    # which display each PLM drives, which takes tens of seconds.
    plms = PLM.open_all("p47")

    # Once that mapping is known, pass it to skip discovery and open in seconds.
    plms = PLM.open_all("p47", display_numbers=[1, 2])

    # Or open a single PLM, finding the display it adds automatically.
    plm = PLM("p47", configure_usb=True)

    # Or without USB, if the EVM is already running (e.g. set up in TI's GUI).
    plm = PLM("p47", display_number=1)

    # Set phase pattern
    phase = np.random.rand(540, 960) * 2 * np.pi
    plm.write(phase)

The USB configuration is accomplished by :class:`DLPC900`, a USB HID interface
for configuring the DLPC900 evaluation module (EVM) that drives the PLM. This
automates the setup normally done through TI's DLPC900 GUI software. For
further information, refer to the `DLPC900 Programmer's Guide
<https://www.ti.com/lit/ug/dlpu018j/dlpu018j.pdf>`_.

Caution
~~~~~~~
Displaying phase to a PLM requires its EVM to be in video-pattern mode with its
pattern sequencer running; the mirrors ignore the video signal otherwise. Only
:meth:`PLM.open_all` and ``configure_usb=True`` set this up, and
:meth:`PLM.close` stops the sequencer again. Opening a PLM with neither (the
last example above) leaves whatever state the EVM was in, so phase written to
the display can silently fail to reach the mirrors.
"""

import yaml
import os
import time
import warnings
from enum import IntEnum
import numpy as np
from slmsuite.hardware._pyglet import ( _WindowThread, _screen_ids,
                                       _screen_index, _wait_for_new_screen,
                                       _wait_for_screens_settled)
from slmsuite.hardware.slms.screenmirrored import ScreenMirrored
from slmsuite.hardware.slms.slm import LUT_SIZE
from slmsuite._logging import make_logger

logger = make_logger(__name__)

try:
    import cupy as cp
except ImportError:
    cp = np
    warnings.warn(
        "cupy is not installed; using numpy. "
        "Install cupy for GPU-accelerated PLM control.",
    )

# HID availability (for DLPC900 USB control)
try:
    import hid as _hid
    HID_AVAILABLE = True
except ImportError:
    _hid = None
    HID_AVAILABLE = False

# PLM Constants
MODEL_DB_PATH = os.path.join(os.path.dirname(__file__), "texas_instruments.yaml")
DLPC900_VENDOR_ID = 0x0451
DLPC900_PRODUCT_ID = 0xC900
DLPC900_EXPOSURE_US = 694

class DisplayMode(IntEnum):
    """
    DLPC900 display modes.
    """
    VIDEO         = 0
    PATTERN       = 1
    VIDEO_PATTERN = 2
    OTF           = 3

class DLPC900Command(IntEnum):
    """
    DLPC900 USB command codes.

    Each value is the two byte command code sent over USB HID, as defined in
    the `DLPC900 Programmer's Guide (DLPU018J)
    <https://www.ti.com/lit/ug/dlpu018j/dlpu018j.pdf>`_.
    """
                              # Programmer Guide Sections
    POWER_MODE     = 0x0200   # 2.2.1 — Standby / wakeup / reset
    VERSION        = 0x0206   # 2.1.5 — Firmware version info
    HW_STATUS      = 0x1A0A   # 2.1.1 — Hardware status register
    MAIN_STATUS    = 0x1A0C   # 2.1.3 — Main status register
    INPUT_SOURCE   = 0x1A00   # 2.3.1 — Input source selection
    IT6535_POWER   = 0x1A01   # 2.3.2 — IT6535 receiver power mode
    PORT_CLOCK     = 0x1A03   # 2.3.3 — Port and clock configuration
    DISPLAY_MODE   = 0x1A1B   # 2.4.1 — Display mode selection
    PAT_STARTSTOP  = 0x1A24   # 2.4.4.3.1 — Pattern start / stop / pause
    PAT_LUT_CONFIG = 0x1A31   # 2.4.4.3.3 — Pattern LUT configuration
    PAT_LUT_DEFINE = 0x1A34   # 2.4.4.3.5 — Pattern LUT entry definition


class PLM(ScreenMirrored):
    """
    Interfaces with Texas Instruments' Phase Light Modulators (PLMs).

    This class combines :class:`ScreenMirrored` for display with GPU-accelerated
    phase quantization and electrode mapping. Automatically detects and uses
    :mod:`cupy` for GPU acceleration, falling back to NumPy if unavailable.

    Optionally configures the DLPC900 EVM via USB, replacing the manual setup
    normally done through TI's GUI software. Use :meth:`open_all` to bring up
    every connected PLM at once, which is the only way to configure several
    without them knocking each other out of source lock.

    Attributes
    ----------
    model_config : dict
        Model configuration from texas_instruments.yaml.
    dlpc900 : DLPC900 or None
        USB interface to DLPC900 EVM, if configured.
    electrode_layout : ndarray
        Physical electrode layout (CuPy or NumPy).
    memory_lut : ndarray
        Memory lookup table.
    data_flip : tuple
        Axis flip flags for electrode output.
    """
    _gamma_sign = +1        # Increasing displacement increases phase delay.

    def __init__(
        self,
        model_name,
        display_number=None,
        configure_usb=False,
        video_input="displayport",
        pixel_mode=None,
        usb_vendor_id=None,
        usb_product_id=None,
        usb_device_number=0,
        dlpc = None,
        gpu=None,
        **kwargs
    ):
        """
        Initialize the PLM interface.

        Parameters
        ----------
        model_name : str
            Model identifier from ``texas_instruments.yaml`` (e.g., ``"p47"``, ``"p67"``).
            Available models can be queried with :meth:`get_model_list()`.
        display_number : int OR None
            Monitor number for display.
            Use :func:`ScreenMirrored.info()` to list available displays and their numbers.
            If ``None``, ``configure_usb`` must be ``True``, and the display that the EVM
            adds during configuration is detected and used automatically.
        configure_usb : bool, optional
            If ``True``, automatically configure the DLPC900 EVM via USB before
            initializing the display, which includes starting the pattern
            sequencer that the mirrors need in order to display phase at all.
            Requires ``hidapi`` (see :class:`DLPC900`). Defaults to ``False``,
            which assumes the EVM was already configured — see the caution in
            the module documentation. For several PLMs, use :meth:`open_all`
            rather than ``configure_usb`` on each.
        video_input : str, optional
            Video input source: ``"displayport"`` or ``"hdmi"``.
            Defaults to ``"displayport"``.
        pixel_mode : str or None, optional
            Pixel clock mode: ``"single"`` (30 Hz) or ``"dual"`` (60 Hz).
            If None, defaults to ``"dual"`` for DisplayPort or ``"single"`` for HDMI.
            Only used when ``configure_usb=True``.
        usb_vendor_id : int or None, optional
            Override USB vendor ID for DLPC900.
        usb_product_id : int or None, optional
            Override USB product ID for DLPC900.
        usb_device_number : int, optional
            Index of the DLPC900 device to open when multiple units are connected.
            Defaults to ``0``. Only used when ``configure_usb=True``.
        gpu : bool or None, optional
            See :attr:`~slmsuite.hardware.slms.slm.SLM.xp`. Unlike the base class, this
            defaults to ``None``, using :mod:`cupy` whenever it is installed. Pass
            :mod:`cupy` arrays to :meth:`set_phase` consistently to avoid a CPU→GPU
            transfer on every call.
        **kwargs
            Additional arguments for :class:`ScreenMirrored`.
        """
        self.dlpc900 = dlpc

        # Load model configuration from YAML database
        self.model_config = self.load_model_config(model_name)

        # Extract model parameters
        model_shape = tuple(self.model_config["shape"])  # (rows, cols) - input phase shape
        pitch_um = tuple(np.array(self.model_config["pitch"]) * 1e6)  # Convert m to µm

        # Store electrode layout for later use
        self._electrode_layout_raw = np.array(self.model_config["electrode_layout"])

        # USB pre-config: set up PLM as display
        if configure_usb:
            # Configure USB.
            self.dlpc900 = DLPC900(
                vendor_id=usb_vendor_id,
                product_id=usb_product_id,
                device_number=usb_device_number
            )

            # Note the currently attached displays to detect the new one 
            known_ids = _screen_ids()

            PLM._usb_pre_configure(self.dlpc900, video_input, pixel_mode,
                                   display_number)

            # If display_number isn't provided, find it.
            if display_number is None:
                display_id = _wait_for_new_screen(known_ids)
                if display_id is None:
                    raise RuntimeError(
                        "PLM did not add a display during USB configuration. "
                        "Check the video cable and that the EVM is powered on."
                    )
                display_number = _screen_index(display_id)

        # No display number without USB config -> no idea what PLM is
        # connected to which display. (ID by resolution is not reliable)
        elif display_number is None:
            raise ValueError(
                "display_number is required unless configure_usb=True, "
                "which detects the added display automatically."
            )
        elif dlpc is None:
            # Nothing here touches USB, so the EVM keeps whatever state it was left in.
            logger.warning(
                "PLM opened without USB, so its pattern sequencer is assumed to be "
                "running already (from PLM.open_all(), configure_usb=True, or TI's GUI). "
                "If it is not, phase written here will never reach the mirrors. Note "
                "that close() stops the sequencer, so reopening this way will not work."
            )

        # Compute bitdepth from number of displacement ratios
        n_phases = len(self.model_config["displacement_ratios"])
        bitdepth = int(np.log2(n_phases))

        # Initialize parent ScreenMirrored class with model shape (vs display shape)
        # The SLM.shape should represent the input phase dimensions
        super().__init__(
            display_number,
            slm_resolution=model_shape[::-1],  # ScreenMirrored expects (width, height)
            bitdepth=bitdepth,
            pitch_um=pitch_um,
            name=kwargs.pop("name", model_name),
            gpu=gpu,
            **kwargs
        )

        logger.debug("PLM using %s backend", "GPU (cupy)" if self.xp is not np else "CPU (numpy)")

        # Calculate display shape after electrode mapping
        elec_shape = self._electrode_layout_raw.shape
        display_shape = (model_shape[0] * elec_shape[0], model_shape[1] * elec_shape[1])

        if display_shape != self.display_shape:
            raise ValueError(
                f"Calculated display shape {display_shape} does not match "
                f"ScreenMirrored display shape {self.display_shape}. "
                f"Check model configuration for consistency."
            )

        # Update window shape and recreate buffers for electrode-mapped output.
        # Must run on the window thread to satisfy OpenGL context thread affinity.
        def _reconfigure(window, shape):
            window.shape = shape
            window._setup_context()

        future = self._window_thread.submit(_reconfigure, self.window, self.display_shape)
        _WindowThread.wait(future)

        # USB post-config: wait for source lock and switch to video-pattern mode.
        if configure_usb:
            PLM._usb_post_configure(self.dlpc900, video_input, pixel_mode)

        # Pre-compute the quantization LUT from the model's non-uniform phase response.
        self.set_gamma(
            np.array(self.model_config["displacement_ratios"])
            * (self.bitresolution - 1) / self.bitresolution
        )

        # Convert model arrays to backend (GPU or CPU)
        self.memory_lut = self.xp.array(self.model_config["memory_lut"], dtype=np.uint8)
        self.electrode_layout = self.xp.array(self._electrode_layout_raw, dtype=np.uint8)
        self.data_flip = tuple(self.model_config["data_flip"])

        # Re-initialize self.display with the electrode-expanded shape so that
        # _format_phase_hw can write in-place (avoiding per-frame allocations).
        self.display = self.xp.zeros(self.display_shape, dtype=self.dtype)

    @staticmethod
    def load_model_config(model_name):
        """
        Load model configuration from texas_instruments.yaml.

        Parameters
        ----------
        model_name : str
            Model identifier (e.g., "p47", "p67")

        Returns
        -------
        dict
            Model configuration

        Raises
        ------
        ValueError
            If model not found in database
        """
        with open(MODEL_DB_PATH, 'r') as f:
            model_db = yaml.safe_load(f)

        if model_name not in model_db:
            available = list(model_db.keys())
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {available}"
            )

        return model_db[model_name]

    @staticmethod
    def configure_usb(
        vendor_id=None,
        product_id=None,
        device_number=0,
        video_input=None,
        pixel_mode=None,
        display_number=None,
    ):
        """
        Convenience method to configure USB in one step.

        Creates a temporary DLPC900 instance to run the pre- and post-configuration
        steps, then closes the USB connection. Useful for users who want to use the
        TI GUI software after running this setup once.

        Parameters
        ----------
        vendor_id : int or None
            USB vendor ID for DLPC900. Defaults to 0x0451 (Texas Instruments).
        product_id : int or None
            USB product ID for DLPC900. Defaults to 0xC900 (DLPC900 EVM).
        device_number : int
            Index of the DLPC900 device to open when multiple units are connected.
        video_input : str
            Video input source: "displayport" or "hdmi".
        pixel_mode : str or None
            Pixel clock mode: "single" (30 Hz) or "dual" (60 Hz).
            If None, defaults to "dual" for DisplayPort or "single" for HDMI.
        display_number : int OR None
            Monitor number for display. If ``None``, does not wait for display detection during pre-configure step.
        """
        dlpc = DLPC900(
            vendor_id=vendor_id,
            product_id=product_id,
            device_number=device_number,
        )

        PLM._usb_pre_configure(
            dlpc=dlpc,
            video_input=video_input,
            pixel_mode=pixel_mode,
            display_number=display_number,
        )

        PLM._usb_post_configure(
            dlpc=dlpc,
            video_input=video_input,
            pixel_mode=pixel_mode,
        )

        return dlpc

    @staticmethod
    def open_all(
        model_name,
        display_numbers=None,
        video_input=None,
        pixel_mode=None,
        names=None,
        cycle=None,
        retries=2,
        **kwargs
    ):
        """
        Configure and open every connected PLM in one step.

        This handles the ordering that makes multi-PLM setups awkward to bring
        up by hand. Each EVM's video receiver is powered up one at a time, so
        the display that it adds can be identified unambiguously. Only once
        *all* displays are attached is each EVM locked to its video source and
        its pattern sequence started: attaching a display retrains the other
        DisplayPort links, so configuring an EVM fully before the next one's
        display appears knocks the first back out of source lock. This is why
        doing this by hand requires running the configuration twice.

        Note
        ~~~~
        Discovering the display-to-EVM mapping requires detaching and reattaching
        the displays, which takes tens of seconds. Once it is known, pass it as
        ``display_numbers`` to skip discovery and open in a few seconds instead.
        A PLM must still be opened through this method (or with
        ``configure_usb=True``) rather than by :meth:`__init__` alone: without
        USB, the EVM's pattern sequencer is never started, so phase written to
        the display never reaches the mirrors.

        Parameters
        ----------
        model_name : str
            Model identifier from ``texas_instruments.yaml`` (e.g. ``"p47"``,
            ``"p67"``), applied to every PLM. See :meth:`get_model_list`.
        display_numbers : list of int OR None
            Display number driven by each EVM, in device order. When given, the
            displays are assumed to be attached already and the mapping is used
            as-is, skipping the (slow) discovery described above. Defaults to
            ``None``, which discovers the mapping.
        video_input : str OR None
            Video input source: ``"displayport"`` or ``"hdmi"``.
            Defaults to ``"displayport"``.
        pixel_mode : str OR None
            Pixel clock mode: ``"single"`` (30 Hz) or ``"dual"`` (60 Hz).
            If ``None``, defaults to ``"dual"`` for DisplayPort or ``"single"``
            for HDMI.
        names : list of str OR None
            :attr:`~slmsuite.hardware.slms.slm.SLM.name` for each PLM, in
            device order.
            Defaults to ``"{model_name}_{i}"`` when several PLMs are connected.
        cycle : bool OR None
            Whether to power the video receivers down before bringing them back
            up one at a time. This is what makes the display-to-EVM mapping
            unambiguous, at the cost of a few seconds of display churn.
            Defaults to ``None``, which cycles only when more than one PLM is
            connected (with one PLM there is nothing to disambiguate).
            Unused when ``display_numbers`` is given.
        retries : int
            How many times to attempt each EVM's display bring-up before giving
            up. Unused when ``display_numbers`` is given.
        **kwargs
            Additional arguments for :meth:`__init__`, applied to every PLM.

        Returns
        -------
        list of PLM
            One :class:`PLM` per connected EVM, ordered by USB device number.
            Use :meth:`DLPC900.info` to map device numbers to physical USB
            ports.
        """
        devices = DLPC900._enumerate()

        if not devices:
            raise RuntimeError(
                "No DLPC900 USB device found. "
                "Check that the EVM(s) are powered on and connected via USB."
            )

        if names is None:
            names = [
                f"{model_name}_{i}" if len(devices) > 1 else model_name
                for i in range(len(devices))
            ]
        elif len(names) != len(devices):
            raise ValueError(
                f"Got {len(names)} names for {len(devices)} connected PLM(s)."
            )

        if display_numbers is not None and len(display_numbers) != len(devices):
            raise ValueError(
                f"Got {len(display_numbers)} display numbers for "
                f"{len(devices)} connected PLM(s)."
            )

        if cycle is None:
            cycle = len(devices) > 1

        dlpcs = [DLPC900(device_number=i) for i in range(len(devices))]
        plms = []

        try:
            if display_numbers is None:
                display_numbers = PLM._discover_displays(
                    dlpcs, video_input, pixel_mode, cycle, retries
                )
            else:
                # The displays are already attached; just put the EVMs in video mode.
                for dlpc, display_number in zip(dlpcs, display_numbers):
                    PLM._usb_pre_configure(
                        dlpc, video_input, pixel_mode, display_number
                    )

            for dlpc, display_number, name in zip(dlpcs, display_numbers, names):
                plms.append(
                    PLM(model_name, display_number, dlpc=dlpc, name=name, **kwargs)
                )

            # Lock and start the sequencers only now that no more displays will appear.
            for plm in plms:
                PLM._usb_post_configure(plm.dlpc900, video_input, pixel_mode)
        except Exception:
            # PLM.close() also releases its DLPC900.
            for opened in plms + dlpcs:      
                try:
                    opened.close()
                except Exception:
                    pass
            raise

        return plms

    @staticmethod
    def _discover_displays(dlpcs, video_input, pixel_mode, cycle, retries):
        """
        Find which display each EVM drives, by bringing them up one at a time.

        Parameters
        ----------
        dlpcs : list of DLPC900
            Open USB connections to the EVMs.
        video_input, pixel_mode : str OR None
            See :meth:`open_all`.
        cycle : bool
            Whether to detach the displays first, so that every EVM's display is
            genuinely new and can therefore be attributed to it.
        retries : int
            How many times to attempt each EVM's display bring-up.

        Returns
        -------
        list of int
            Display number driven by each EVM, in the order of ``dlpcs``.
        """
        # Start from a known state: no PLM displays attached. Without this, an
        # already-attached display cannot be matched to the EVM driving it.
        if cycle:
            logger.debug("Powering down %s DLPC900 video receiver(s)...", len(dlpcs))
            for dlpc in dlpcs:
                dlpc.standby()
            _wait_for_screens_settled()

        # Bring the EVMs up one at a time, claiming the display that each one adds.
        known_ids = _screen_ids()
        display_ids = []

        for index, dlpc in enumerate(dlpcs):
            for attempt in range(retries):
                PLM._usb_pre_configure(dlpc, video_input, pixel_mode, None)
                display_id = _wait_for_new_screen(known_ids)
                if display_id is not None:
                    break
                logger.warning(
                    "PLM %s did not add a display (attempt %s of %s).",
                    index, attempt + 1, retries
                )
            else:
                raise RuntimeError(
                    f"PLM {index} did not add a display after {retries} attempts. Its "
                    "display may already be attached (pass its display_numbers, or "
                    "cycle=True to power the receivers down and rediscover), or check "
                    "the video cable and that the EVM is powered on."
                )

            logger.debug("PLM %s added display '%s'.", index, display_id)
            known_ids.add(display_id)
            display_ids.append(display_id)

        # Every display is attached, so the display numbering has settled and it is
        # finally safe to resolve identifiers to numbers.
        display_numbers = []

        for display_id in display_ids:
            display_number = _screen_index(display_id)
            if display_number is None:
                raise RuntimeError(
                    f"Display '{display_id}' detached during PLM configuration."
                )
            display_numbers.append(display_number)

        logger.debug("Discovered display numbers %s.", display_numbers)

        return display_numbers

    @staticmethod
    def _usb_pre_configure(dlpc, video_input, pixel_mode, display_number):
        """
        USB setup steps that must happen before the pyglet window is created.

        Sets input source, port clock configuration, and switches to video mode
        so the EVM is ready to accept video signal from the display. Polls pyglet
        to confirm the target display is available before proceeding.
        """
        from slmsuite.hardware.slms.screenmirrored import ScreenMirrored

        logger.debug("DLPC900 connected: firmware %s", dlpc.get_firmware_version())

        # Resolve video_input default
        if video_input is None:
            video_input = "displayport"

        # Resolve pixel mode default
        if pixel_mode is None:
            pixel_mode = "dual" if video_input == "displayport" else "single"

        # Configure port clock for single or dual pixel mode
        if pixel_mode == "dual":
            dlpc.set_port_clock(data_port=2)
        else:
            dlpc.set_port_clock(data_port=0)

        # Power up IT6535 receiver for the correct input before any display config
        dlpc.set_it6535_power(video_input)

        # Switch to video mode (required before video-pattern)
        dlpc.set_display_mode("video")

        # Wait for the target display to become available
        if display_number is not None:
            DLPC900._poll_until(
                lambda: display_number in [s[0] for s in ScreenMirrored.info(verbose=False)],
                error_msg=f"Display {display_number} not detected.",
            )

        logger.debug("DLPC900 pre-configured (video mode, display detected)")

    @staticmethod
    def _usb_post_configure(dlpc, video_input, pixel_mode):
        """
        USB setup steps that happen after the pyglet window is created.

        Waits for external source lock (video signal detected), then switches
        to video-pattern mode, configures the pattern LUT, and starts the
        pattern sequence.
        """
        # Resolve video_input and pixel mode defaults
        if video_input is None:
            video_input = "displayport"
        if pixel_mode is None:
            pixel_mode = "dual" if video_input == "displayport" else "single"

        # Wait for external source lock, re-asserting video mode once if it
        # does not come up: attaching another display retrains the DisplayPort
        # links and can leave an already-configured EVM unlocked.
        try:
            DLPC900._poll_until(lambda: dlpc.get_main_status()["source_locked"])
        except RuntimeError:
            logger.warning("DLPC900: video source not locked; re-asserting video mode.")
            dlpc.set_it6535_power(video_input)
            dlpc.set_display_mode("video")
            DLPC900._poll_until(
                lambda: dlpc.get_main_status()["source_locked"],
                error_msg="DLPC900: Video source failed to lock.",
            )

        logger.debug("DLPC900 source locked, switching to video-pattern mode...")

        # Switch to video-pattern mode and wait for confirmation
        dlpc.set_display_mode("video-pattern")
        DLPC900._poll_until(
            lambda: dlpc.get_display_mode() == DisplayMode.VIDEO_PATTERN,
            error_msg="DLPC900: Failed to switch to video-pattern mode.",
        )

        # Stop any existing sequence
        dlpc.stop_pattern()

        # Define a single 1-bit pattern entry (copied to all bits by PLM class):
        # - No clear
        # - Trigger out 2 enabled (per GUI instructions)
        # - Frame change on first bit (bit_position=0)
        dlpc.define_pattern(
            index=0,
            bitdepth=1,
            color=1, #shouldn't matter
            clear_after_exposure=False,
            wait_for_trigger=True,
            dark_time_us=0,
            trigger_out2=True,
            image_index=0,
            bit_position=0,
        )

        # Configure LUT: 1 entry, repeat indefinitely
        dlpc.configure_pattern_lut(num_entries=1, num_repeats=0)
        time.sleep(1) # Wait for small unresponsive time window

        # Start the pattern sequence and wait for confirmation
        dlpc.start_pattern()
        DLPC900._poll_until(
            lambda: dlpc.get_main_status()["sequencer_running"],
            timeout_s=2,
            error_msg=(
                "DLPC900: Pattern sequence failed to start after 2 seconds. "
                "Check hardware status with dlpc.get_hardware_status()."
            ),
        )

        logger.debug("DLPC900 configured successfully - pattern sequence running")

    def close(self, power_down=False):
        """
        Close the PLM, stopping the pattern sequence and releasing USB.

        Stopping the pattern sequence parks the mirrors, so the PLM will ignore its
        video signal until an EVM is configured over USB again — reopen with
        :meth:`open_all` or ``configure_usb=True``, not with :meth:`__init__` alone.

        Parameters
        ----------
        power_down : bool
            Whether to also power down the EVM's IT6535 video receiver. This
            detaches the PLM's display from the OS, so the display has to be
            brought back up and re-identified the next time the PLM is opened.
            Defaults to ``False``, which leaves the display attached and makes
            reopening fast.
        """
        if self.dlpc900 is not None:
            try:
                self.dlpc900.stop_pattern()
                if power_down:
                    self.dlpc900.standby()
                self.dlpc900.close()
            except Exception:
                pass
            self.dlpc900 = None
        super().close()

    def set_gamma(self, gamma=None, lut_size=LUT_SIZE):
        """
        See :meth:`~slmsuite.hardware.slms.slm.SLM.set_gamma`. A PLM addresses its phase
        states through the table, so it has no ideal linear response to clear back to.
        """
        if gamma is None:
            raise ValueError(
                "A PLM requires a lookup table; pass the model's displacement ratios."
            )
        return super().set_gamma(gamma, lut_size)

    def _init_quantize_lut(self, displacement_ratios=None):
        """
        Pre-compute a quantization lookup table (LUT) that maps discretized
        phase values directly to phase state indices.

        Replaces per-frame float modulo and ``searchsorted`` or ``digitize``
        with a single array index at runtime. The LUT has 2^16 entries (64 KB),
        built once from the model's non-uniform displacement ratios.
        """
        if displacement_ratios is None:
            displacement_ratios = np.array(self.model_config["displacement_ratios"])
        else:
            if len(displacement_ratios) != self.bitresolution:
                raise ValueError(
                    f"Expected {self.bitresolution} displacement ratios, "
                    f"got {len(displacement_ratios)}."
                )

        # Scale displacement ratios to (bitresolution - 1) / bitresolution
        ratio_scale = (self.bitresolution - 1) / self.bitresolution

        # Map displacement ratios to phase values in [0, 2pi)
        phase_disp = displacement_ratios * ratio_scale * (2 * np.pi)
        phase_disp = np.concatenate([phase_disp, [2 * np.pi]])

        # Bucket boundaries (midpoints between adjacent phase levels)
        phase_buckets = (phase_disp[:-1] + phase_disp[1:]) / 2

        # Build LUT: map each of the uniformly-spaced phase values to a state
        grid = np.arange(LUT_SIZE, dtype=np.float64) * (2 * np.pi / LUT_SIZE)
        lut = np.searchsorted(phase_buckets, grid, side='right')
        lut = (lut & (self.bitresolution - 1)).astype(np.uint8)
        self._quantize_lut = self.xp.asarray(lut)

    def _quantize(self, phase_map):
        """
        Quantize continuous phase (in any range) to discrete phase state indices via
        :attr:`~slmsuite.hardware.slms.slm.SLM.lut`.
        """
        return self.lut[self._phase2lut(self.xp.asarray(phase_map))]

    def _electrode_map(self, phase_state_idx):
        """
        Map phase state indices to electrode bit patterns.

        Converts quantized phase states to the physical electrode layout
        pattern required by the PLM hardware.

        Parameters
        ----------
        phase_state_idx : ndarray (uint8)
            Phase state indices, must be at least 2D
            Last 2 dimensions represent (rows, cols)

        Returns
        -------
        ndarray (uint8)
            Binary electrode pattern with expanded dimensions based on
            electrode_layout shape
        """
        xp = self.xp

        # Look up memory values for each phase state
        memory = self.memory_lut[phase_state_idx]

        # Broadcast and apply bitwise operations for electrode mapping
        # memory[..., None, None] adds 2 dims: (..., rows, cols, 1, 1)
        # electrode_layout has shape (elec_rows, elec_cols)
        # Result has shape (..., rows, cols, elec_rows, elec_cols)
        out = xp.right_shift(
            memory[..., None, None],
            self.electrode_layout) & 1

        # Rearrange axes and reshape to interleave electrode bits
        elec_h, elec_w = self.electrode_layout.shape
        new_shape = memory.shape[:-2] + (memory.shape[-2] * elec_h, memory.shape[-1] * elec_w)
        out = xp.swapaxes(out, -2, -3).reshape(new_shape)

        # Apply data flip if specified
        flip_axes = tuple(-2 + idx for idx, flip in enumerate(self.data_flip) if flip)
        if flip_axes:
            out = xp.flip(out, flip_axes)

        return out

    def _format_phase_hw(self, phase, replicate_bits=True):
        """
        Process phase array into PLM electrode bitmap.

        Combines quantization and electrode mapping into optimized pipeline.
        Data stays on GPU if available for maximum performance - ScreenMirrored
        will handle GPU→CPU transfer only when needed for display.

        Parameters
        ----------
        phase : numpy.ndarray or cupy.ndarray
            Phase data in any range (wrapping to [0, 2π) is handled internally
            by :meth:`_quantize`).
        replicate_bits : bool, optional
            Multiply final bitplane by 255 to display same CGH for full frame.
            Defaults to True.

        Returns
        -------
        numpy.ndarray or cupy.ndarray (uint8)
            Electrode-mapped bitmap ready for display.
            Returns GPU array if ``gpu`` backend is active, otherwise CPU array.

        Raises
        ------
        ValueError
            If enforce_shape=True and phase shape doesn't match model shape
        """
        xp = self.xp

        # Shape validation
        if len(phase.shape) < 2 or phase.shape[-2:] != self.shape:
            raise ValueError(
                f"Phase map shape {phase.shape} does not match "
                f"model shape {self.shape}"
            )

        # Coerce input to match backend (e.g. numpy→cupy if gpu=True)
        phase = xp.asarray(phase)

        # Quantize phase to discrete states (handles [0, 2π) wrapping internally)
        phase_state_idx = self._quantize(phase)

        return self._gray2display(phase_state_idx, replicate_bits=replicate_bits)

    def _gray2display(self, gray, replicate_bits=True):
        xp = self.xp
        # Map to electrode pattern
        result = self._electrode_map(gray)

        # Write into self.display in-place to avoid per-frame allocations
        # (mirrors how _phase2gray writes to self.display in slm.py).
        if replicate_bits:
            xp.multiply(result, 255, out=self.display, casting="unsafe")
        else:
            xp.copyto(self.display, result, casting="unsafe")

        return self.display

    @staticmethod
    def bitpack(bitmaps):
        """
        Combine multiple binary CGHs into single 8-bit or 24-bit image.

        Stacks the MSB of 8 or 24 bitmaps into a single multi-bit image.
        Supports GPU acceleration if :mod:`cupy` is available and input is on GPU.

        Parameters
        ----------
        bitmaps : list or tuple of ndarray
            List of 8 or 24 binary bitmaps (uint8) of same shape

        Returns
        -------
        ndarray (uint8)
            Packed image with shape (1, rows, cols) for 8 bitmaps
            or (3, rows, cols) for 24 bitmaps (RGB channels)

        Raises
        ------
        ValueError
            If number of bitmaps is not 8 or 24
        """
        # Determine backend from input arrays
        from slmsuite.hardware.slms.slm import _xp
        xp = _xp(bitmaps[0]) if bitmaps else np

        # Ensure all bitmaps are on same device
        bitmaps = [xp.asarray(bm) for bm in bitmaps]

        if len(bitmaps) == 8:
            # Single channel output
            stacked = xp.stack(bitmaps) & 1  # Isolate LSB
            shifts = xp.arange(8)[:, None, None]  # Shape (8, 1, 1) for broadcasting
            shifted = xp.left_shift(stacked.astype(xp.uint8), shifts.astype(xp.uint8))
            result = xp.sum(shifted, axis=0)[None, ...]  # Add channel dimension

        elif len(bitmaps) == 24:
            # RGB output (3 channels, 8 bits each)
            rgb = []
            for n in range(3):
                channel_bitmaps = bitmaps[n*8:(n+1)*8]
                stacked = xp.stack(channel_bitmaps) & 1
                shifts = xp.arange(8)[:, None, None]
                shifted = xp.left_shift(stacked.astype(xp.uint8), shifts.astype(xp.uint8))
                rgb.append(xp.sum(shifted, axis=0))
            result = xp.stack(rgb)

        else:
            raise ValueError(
                f"Bitpack requires 8 or 24 bitmaps, got {len(bitmaps)}"
            )

        # Convert back to NumPy if input was on GPU
        if xp is not np:
            result = np.asarray(result)

        return result

    @staticmethod
    def get_model_list():
        """
        Get list of available PLM models from database.

        Returns
        -------
        list of str
            Model identifiers available in texas_instruments.yaml
        """
        with open(MODEL_DB_PATH, 'r') as f:
            model_db = yaml.safe_load(f)

        return list(model_db.keys())



class DLPC900:
    """
    USB HID interface for the DLPC900 evaluation module.

    Implements the DLPC900 USB commands needed to configure the EVM for video
    pattern mode, eliminating the need for TI's GUI software. Uses the native
    OS HID driver via ``hidapi`` — no driver replacement (Zadig) required.

    The DLPC900 communicates via 64-byte HID reports with a 6-byte header::

        [flag, seq, len_lo, len_hi, cmd_lo, cmd_hi, ...payload...]

    See :class:`DLPC900Command` for the implemented command codes and their
    DLPU018J section references.
    """

    def __init__(self, vendor_id=None, product_id=None, device_number=0):
        """
        Initialize the DLPC900 USB interface.

        Parameters
        ----------
        vendor_id : int or None
            USB vendor ID. Defaults to ``0x0451`` (Texas Instruments).
        product_id : int or None
            USB product ID. Defaults to ``0xC900`` (DLPC900 EVM).
        device_number : int, optional
            Index of the device to open when multiple units are connected.
            Defaults to ``0``.

        Raises
        ------
        ImportError
            If the ``hidapi`` package is not installed.
        RuntimeError
            If the DLPC900 USB device is not found.
        """
        vid = vendor_id if vendor_id is not None else DLPC900_VENDOR_ID
        pid = product_id if product_id is not None else DLPC900_PRODUCT_ID

        devices = DLPC900._enumerate(vid, pid)

        if not devices:
            raise RuntimeError(
                f"No DLPC900 USB device found (VID=0x{vid:04X}, PID=0x{pid:04X}). "
                "Check that the EVM is powered on and connected via USB."
            )
        if device_number >= len(devices):
            raise RuntimeError(
                f"device_number={device_number} out of range; "
                f"{len(devices)} DLPC900 PLM(s) found."
            )
        self._device_info = devices[device_number]
        logger.debug(
            "DLPC900 device %s/%s: path=%s",
            device_number, len(devices), self._device_info["path"].decode()
        )
        self._dev = _hid.device()
        try:
            self._dev.open_path(self._device_info["path"])
        except OSError as e:
            raise RuntimeError(
                f"Failed to open DLPC900 device {device_number} "
                f"(VID=0x{vid:04X}, PID=0x{pid:04X})."
            ) from e

        self._seq = 0

    @staticmethod
    def _enumerate(vendor_id=None, product_id=None):
        """
        List the connected DLPC900 EVMs.

        Each EVM exposes two USB HID interfaces; only the one reporting the product string
        ``"DLPC900"`` accepts the commands implemented here.

        Parameters
        ----------
        vendor_id, product_id : int OR None
            USB identifiers. Default to Texas Instruments' DLPC900 EVM.

        Returns
        -------
        list of dict
            :mod:`hid` device dictionaries, in :mod:`hid` enumeration order.

        Raises
        ------
        ImportError
            If the ``hidapi`` package is not installed.
        """
        if not HID_AVAILABLE:
            raise ImportError(
                "hidapi is required for DLPC900 USB control. "
                "Install with: pip install hidapi"
            )

        vid = vendor_id if vendor_id is not None else DLPC900_VENDOR_ID
        pid = product_id if product_id is not None else DLPC900_PRODUCT_ID

        return [
            device for device in _hid.enumerate(vid, pid)
            if device.get("product_string") == "DLPC900"
        ]

    @staticmethod
    def info(verbose=True, vendor_id=None, product_id=None):
        """
        Get information about the connected DLPC900 EVMs and their device numbers.

        A device's USB path is tied to the physical USB port it is plugged into, so it is
        a stable way to tell several otherwise-identical PLMs apart.

        Parameters
        ----------
        verbose : bool
            Whether or not to print device information.
        vendor_id, product_id : int OR None
            USB identifiers. Default to Texas Instruments' DLPC900 EVM.

        Returns
        -------
        list of (int, str) tuples
            The device number and USB path of each EVM.
        """
        devices = DLPC900._enumerate(vendor_id, product_id)

        if verbose:
            print("DLPC900 Devices:")
            print("#,  Path")

        device_list = []

        for device_number, device in enumerate(devices):
            path = device["path"].decode()

            if verbose:
                print(f"{device_number},  {path}")

            device_list.append((device_number, path))

        return device_list

    def _send(self, mode, cmd, payload=None):
        """
        Send a command and optionally read the response.

        Parameters
        ----------
        mode : str
            ``'r'`` for read, ``'w'`` for write.
        cmd : DLPC900Command or int
            16-bit command code.
        payload : list of int or None
            Command data bytes.

        Returns
        -------
        list of int or None
            64-byte response for reads, None for writes.
        """
        if payload is None:
            payload = []

        self._seq = (self._seq + 1) & 0xFF
        cmd = int(cmd)
        length = len(payload) + 2

        # Build 64-byte packet: [flag, seq, len_lo, len_hi, cmd_lo, cmd_hi, ...data...]
        flag = 0xC0 if mode == 'r' else 0x00
        header = bytes([flag, self._seq]) + length.to_bytes(2, 'little') + cmd.to_bytes(2, 'little')
        buf = list(header) + payload[:58] + [0] * (58 - len(payload[:58]))

        # hidapi write: prepend report ID 0x00
        # print(" ".join(f"{b:02X}" for b in buf))
        self._dev.write([0x00] + buf)

        # Multi-packet payload (>58 bytes)
        remaining = payload[58:]
        while remaining:
            chunk = remaining[:64]
            remaining = remaining[64:]
            padded = chunk + [0x00] * (64 - len(chunk))
            self._dev.write([0x00] + padded)

        if mode == 'r':
            try:
                ret = self._dev.read(64, timeout_ms=1000)
                # print(" ".join(f"{b:02X}" for b in ret))
                return ret
            except Exception:
                logger.warning("Read command failed; ensure PLM GUI is closed.")

        # A bit of time for stability
        time.sleep(0.1)

        return None

    def _read_byte(self, cmd):
        """Read a single status byte (response byte 5) for a command."""
        ans = self._send('r', cmd)
        return ans[4] if ans else None

    @staticmethod
    def _poll_until(check_fn, timeout_s=10, interval_s=0.5, error_msg=""):
        """
        Poll ``check_fn`` until it returns truthy, or raise on timeout.

        Parameters
        ----------
        check_fn : callable
            Zero-argument callable that returns a truthy value on success.
        timeout_s : float
            Maximum time to wait in seconds.
        interval_s : float
            Sleep interval between polls in seconds.
        error_msg : str
            Message for the :class:`RuntimeError` raised on timeout.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            time.sleep(interval_s)
            if check_fn():
                return
        raise RuntimeError(error_msg)

    def close(self):
        """Release the USB HID device."""
        if self._dev is not None:
            self._dev.close()
            self._dev = None

    def get_hardware_status(self):
        """
        Read hardware status register.

        Returns
        -------
        dict
            Bool flags: ``init_done``, ``drc_error``, ``forced_swap``,
            ``sequencer_abort``, ``sequencer_error``.
        """
        b = self._read_byte(DLPC900Command.HW_STATUS)
        return {
            "init_done":       bool(b & 0x01),
            "drc_error":       bool(b & 0x04),
            "forced_swap":     bool(b & 0x08),
            "sequencer_abort": bool(b & 0x40),
            "sequencer_error": bool(b & 0x80),
        }

    def get_main_status(self):
        """
        Read main status register.

        Returns
        -------
        dict
            Bool flags: ``mirrors_parked``, ``sequencer_running``,
            ``video_frozen``, ``source_locked``, ``port1_syncs_valid``,
            ``port2_syncs_valid``.
        """
        b = self._read_byte(DLPC900Command.MAIN_STATUS)
        return {
            "mirrors_parked":    bool(b & 0x01),
            "sequencer_running": bool(b & 0x02),
            "video_frozen":      bool(b & 0x04),
            "source_locked":     bool(b & 0x08),
            "port1_syncs_valid": bool(b & 0x10),
            "port2_syncs_valid": bool(b & 0x20),
        }

    def get_firmware_version(self):
        """
        Read firmware version info.

        Returns
        -------
        dict
            Keys: ``app_version``, ``api_version``, ``sw_patch``,
            ``sw_minor``, ``sw_major``.
        """
        ans = self._send('r', DLPC900Command.VERSION)
        if not ans or len(ans) < 10:
            return {}
        return {
            "app_version": ans[6],
            "api_version": ans[7],
            "sw_patch":    ans[8],
            "sw_minor":    ans[9],
            "sw_major":    ans[10] if len(ans) > 10 else 0,
        }

    def set_input_source(self, source=0, bitdepth=0):
        """
        Set input source.

        Parameters
        ----------
        source : int
            0 = parallel (HDMI/DP), 1 = test, 2 = flash, 3 = curtain.
        bitdepth : int
            0 = 30-bit, 1 = 24-bit, 2 = 20-bit, 3 = 16-bit.
        """
        self._send('w', DLPC900Command.INPUT_SOURCE,
                   [source & 0x07 | (bitdepth & 0x03) << 3])

    def set_port_clock(self, data_port, px_clock=0, data_enable=0, vhsync=0):
        """
        Configure data port and clock routing.

        Parameters
        ----------
        data_port : int
            0 = port 1, 1 = port 2, 2 = dual (1-2), 3 = dual (2-1).
        px_clock : int
            0 = clock 1, 1 = clock 2, 2 = clock 3.
        data_enable : int
            0 = enable 1, 1 = enable 2.
        vhsync : int
            0 = P1 sync, 1 = P2 sync.
        """
        self._send('w', DLPC900Command.PORT_CLOCK, [
            data_port & 0x03
            | (px_clock & 0x03) << 2
            | (data_enable & 0x01) << 4
            | (vhsync & 0x01) << 5
        ])

    def set_display_mode(self, mode):
        """
        Set display mode.

        Parameters
        ----------
        mode : str or DisplayMode
            ``"video"``, ``"pattern"``, ``"video-pattern"``, or ``"otf"``.
            Must be in ``"video"`` mode with source locked before switching
            to ``"video-pattern"``.
        """
        if isinstance(mode, DisplayMode):
            self._send('w', DLPC900Command.DISPLAY_MODE, [int(mode)])
            return

        # Accept string with underscore or hyphen
        name = mode.upper().replace("-", "_")
        try:
            val = DisplayMode[name]
        except KeyError:
            valid = [m.name.lower().replace("_", "-") for m in DisplayMode]
            raise ValueError(
                f"Unknown mode '{mode}'. Valid: {valid}"
            ) from None
        self._send('w', DLPC900Command.DISPLAY_MODE, [int(val)])

    def get_display_mode(self):
        """
        Read current display mode.

        Returns
        -------
        DisplayMode
            The current display mode.
        """
        b = self._read_byte(DLPC900Command.DISPLAY_MODE)
        try:
            return DisplayMode(b)
        except ValueError:
            raise ValueError(f"Unknown display mode byte: {b}") from None

    def start_pattern(self):
        """Start the pattern display sequence."""
        self._send('w', DLPC900Command.PAT_STARTSTOP, [0x02])

    def stop_pattern(self):
        """Stop the pattern display sequence."""
        self._send('w', DLPC900Command.PAT_STARTSTOP, [0x00])

    def configure_pattern_lut(self, num_entries, num_repeats=0):
        """
        Configure the pattern LUT.

        Parameters
        ----------
        num_entries : int
            Number of LUT entries to display.
        num_repeats : int
            Repeat count (0 = infinite).
        """
        self._send(
            'w', DLPC900Command.PAT_LUT_CONFIG,
            list(num_entries.to_bytes(2, 'little'))
            + list(num_repeats.to_bytes(4, 'little'))
        )

    def define_pattern(
        self, index, bitdepth=1, color=7,
        clear_after_exposure=False, wait_for_trigger=False,
        dark_time_us=0, trigger_out2=False,
        image_index=0, bit_position=0,
    ):
        """
        Define a single pattern LUT entry.

        Uses the fixed exposure time :data:`DLPC900_EXPOSURE_US`.

        Parameters
        ----------
        index : int
            LUT index (0-399).
        bitdepth : int
            Bit depth (1-8).
        color : int
            Color channel (0-7; 7 = all RGB).
        clear_after_exposure : bool
            Clear pattern after exposure.
        wait_for_trigger : bool
            Wait for external trigger.
        dark_time_us : int
            Dark time after exposure (microseconds).
        trigger_out2 : bool
            Assert trigger output 2.
        image_index : int
            Source image/frame index.
        bit_position : int
            Bit position in image (0-23).
        """
        # Byte 5: [trigger_wait(7)][color(6:4)][depth-1(3:1)][clear(0)]
        options = (
            int(clear_after_exposure) & 0x01
            | ((bitdepth - 1) & 0x07) << 1
            | (color & 0x07) << 4
            | (int(wait_for_trigger) & 0x01) << 7
        )

        payload = (
            list(index.to_bytes(2, 'little'))
            + list(DLPC900_EXPOSURE_US.to_bytes(3, 'little'))
            + [options]
            + list(dark_time_us.to_bytes(3, 'little'))
            + [
                int(not trigger_out2) & 0x01,
                image_index & 0xFF,
                (image_index >> 8) & 0x07 | (bit_position & 0x1F) << 3,
            ]
        )
        self._send('w', DLPC900Command.PAT_LUT_DEFINE, payload)

    def set_it6535_power(self, mode):
        """
        Set IT6535 receiver power mode (0x1A01).

        Must be called before setting video mode.

        Parameters
        ----------
        mode : int or str
            0 or ``"off"`` = power-down (outputs tri-stated),
            1 or ``"hdmi"`` = power-up for HDMI input,
            2 or ``"displayport"`` = power-up for DisplayPort input.
        """
        modes = {"off": 0, "hdmi": 1, "displayport": 2}
        if isinstance(mode, str):
            mode = modes[mode.lower()]
        if mode not in modes.values():
            raise ValueError(f"Invalid IT6535 power mode: {mode}")
        else:
            self._send('w', DLPC900Command.IT6535_POWER, [mode & 0x03])

    def standby(self):
        """Put the IT6535 receiver into power-down mode."""
        self.set_it6535_power(0)

    def reset(self):
        """Reset the DLPC900."""
        self._send('w', DLPC900Command.POWER_MODE, [0x02])