"""
Hardware control for FLIR cameras via the :mod:`PySpin` interface to the Spinnaker SDK.
Install PySpin using the `provided instructions
<https://www.teledynevisionsolutions.com/support/support-center/technical-guidance/iis/installing-pyspin-for-the-spinnaker-sdk/>`_.

This implementation uses the QuickSpin API for simplified property access.
Refer to the PySpin documentation (included with installation) for
details on alternative approaches using the full Spinnaker API.

"""

import warnings
import numpy as np
from slmsuite.hardware.cameras.camera import Camera
from slmsuite._logging import make_logger

logger = make_logger(__name__)

try:
    import PySpin
except ImportError:
    PySpin = None
    warnings.warn("PySpin not installed. Install to use FLIR cameras.")

class FLIR(Camera):
    """
    FLIR camera subclass.

    Attributes
    ----------
    sdk : PySpin.System
        Spinnaker SDK singleton. Shared among instances of :class:`FLIR`.
    cam : PySpin.Camera
        Object to talk with the desired camera.
    camera_list : PySpin.CameraList
        List of available cameras for cleanup.
    """

    sdk = None

    ### Initialization and termination ###
    def __init__(self, serial="", bitdepth=None, pitch_um=None, **kwargs):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        serial : str
            Serial number of the camera to open.
            Use :meth:`.info()` to see detected options.
            If empty, defaults to the first camera in the list
            returned by :meth:`PySpin.System.GetCameras()`.
        bitdepth : int or None
            Desired ADC bit depth (8, 10, or 12). If ``None``, selects the
            highest available ADC bit depth.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        if PySpin is None:
            raise ImportError(
                "PySpin not installed. Install FLIR Spinnaker SDK and its Python "
                "bindings to use FLIR cameras."
            )

        # Initialize SDK singleton if needed
        if FLIR.sdk is None:
            logger.debug("PySpin initializing...")
            FLIR.sdk = PySpin.System.GetInstance()

        # Get camera list
        logger.debug("Looking for cameras...")
        self.camera_list = FLIR.sdk.GetCameras()

        # Build serial list and validate camera selection
        num_cameras = self.camera_list.GetSize()
        serial_list = []
        for i in range(num_cameras):
            cam_temp = self.camera_list.GetByIndex(i)
            nodemap_tldevice = cam_temp.GetTLDeviceNodeMap()
            node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsReadable(node_serial):
                serial_list.append(node_serial.GetValue())

        if serial == "":
            if num_cameras == 0:
                raise RuntimeError("No cameras found by PySpin.")
            if num_cameras > 1:
                logger.debug("No serial given; choosing first of %s", serial_list)
            self.cam = self.camera_list.GetByIndex(0)
            # Get actual serial for naming
            nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
            node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsReadable(node_serial):
                serial = node_serial.GetValue()
        else:
            if serial in serial_list:
                self.cam = self.camera_list.GetBySerial(serial)
            else:
                raise RuntimeError(
                    f"Serial {serial} not found by PySpin. Available: {serial_list}"
                )

        # Initialize camera
        logger.debug("PySpin sn '%s' initializing...", serial)

        try:
            self.cam.Init()
        except PySpin.SpinnakerException as ex:
            raise RuntimeError(f"Failed to initialize camera: {ex}")

        # If the camera was left streaming from a previous crashed session,
        # PixelFormat becomes RO while streaming, preventing format changes.
        try:
            if self.cam.IsStreaming():
                self.cam.EndAcquisition()
        except PySpin.SpinnakerException:
            pass

        # Configure camera properties
        self._trigger_is_software = False
        try:
            # Turn off automatic modes for manual control
            if self.cam.GainAuto.GetAccessMode() == PySpin.RW:
                self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
            else:
                logger.warning("GainAuto is not writable; could not set to Off.")
            if self.cam.Gain.GetAccessMode() == PySpin.RW:
                self.cam.Gain.SetValue(0.0)
            else:
                logger.warning("Gain is not writable; could not set to 0.0 dB.")
            if self.cam.ExposureAuto.GetAccessMode() == PySpin.RW:
                self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            else:
                logger.warning("ExposureAuto is not writable; could not set to Off.")
            if self.cam.ExposureMode.GetAccessMode() == PySpin.RW:
                self.cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
            else:
                logger.warning("ExposureMode is not writable; could not set to Timed.")

            # Black level: set to 0 for clean scientific imaging
            try:
                if self.cam.BlackLevelSelector.GetAccessMode() == PySpin.RW:
                    self.cam.BlackLevelSelector.SetValue(PySpin.BlackLevelSelector_All)
                else:
                    logger.warning("BlackLevelSelector is not writable; could not set to All.")
                if self.cam.BlackLevel.GetAccessMode() == PySpin.RW:
                    self.cam.BlackLevel.SetValue(0.0)
                else:
                    logger.warning("BlackLevel is not writable; could not set to 0.0.")
            except PySpin.SpinnakerException as ex:
                logger.warning(f"BlackLevel configuration failed: {ex}")

            # Gamma: disable for linear sensor response
            try:
                if self.cam.GammaEnable.GetAccessMode() == PySpin.RW:
                    self.cam.GammaEnable.SetValue(False)
                else:
                    logger.warning("GammaEnable is not writable; could not disable.")
            except PySpin.SpinnakerException:
                try:
                    if self.cam.Gamma.GetAccessMode() == PySpin.RW:
                        self.cam.Gamma.SetValue(1.0)
                    else:
                        logger.warning("Gamma is not writable; could not set to 1.0.")
                except PySpin.SpinnakerException as ex:
                    logger.warning(f"Gamma configuration failed: {ex}")

            # Configure pixel format
            bitdepth = self._configure_adc_depth(bitdepth=bitdepth)

            # Set a reasonable default exposure so _get_dtype's test capture
            # doesn't time out waiting for the camera's power-on default
            # (which can be as long as 30 s on some models).
            if self.cam.ExposureTime.GetAccessMode() == PySpin.RW:
                self.cam.ExposureTime.SetValue(self.cam.ExposureTime.GetMin())
            else:
                logger.warning("ExposureTime is not writable; could not set to minimum.")

            # Configure software trigger
            if self.cam.TriggerMode.GetAccessMode() == PySpin.RW:
                self.cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
            else:
                logger.warning("TriggerMode is not writable; could not set to On.")
            if self.cam.TriggerSource.GetAccessMode() == PySpin.RW:
                self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
            else:
                logger.warning("TriggerSource is not writable; could not set to Software.")
            if self.cam.TriggerSelector.GetAccessMode() == PySpin.RW:
                self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
            else:
                logger.warning("TriggerSelector is not writable; could not set to FrameStart.")

            # Cache the trigger configuration to avoid per-frame access.
            self._trigger_is_software = self._read_trigger_is_software()

            # Configure frame rate. This has to follow the trigger configuration
            # above, because the rate limiter is only meaningful when free-running.
            self._configure_frame_rate()

        except PySpin.SpinnakerException as ex:
            logger.warning(f"Failed to configure camera: {ex}")

        # Begin acquisition
        try:
            self.cam.BeginAcquisition()
        except PySpin.SpinnakerException as ex:
            raise RuntimeError(f"Failed to begin acquisition: {ex}")

        super().__init__(
            (self.cam.WidthMax.GetValue(), self.cam.HeightMax.GetValue()),
            bitdepth=bitdepth,
            pitch_um=pitch_um,
            name=serial,
            **kwargs
        )

        # Cache exposure bounds from hardware, unless the user supplied them.
        if self.exposure_bounds_s is None:
            try:
                self.exposure_bounds_s = (
                    self.cam.ExposureTime.GetMin() / 1e6,
                    self.cam.ExposureTime.GetMax() / 1e6,
                )
            except PySpin.SpinnakerException:
                pass

        self.logger.debug("Successfully initialized FLIR cam %s.", serial)

    def close(self):
        """See :meth:`.Camera.close`."""
        try:
            self.cam.EndAcquisition()
        except Exception:
            pass

        try:
            self.cam.DeInit()
        except Exception:
            pass

        # Clean up camera list
        if hasattr(self, 'camera_list'):
            try:
                self.camera_list.Clear()
            except Exception:
                pass
            del self.camera_list

        if hasattr(self, 'cam'):
            del self.cam

    @staticmethod
    def info(verbose=True):
        """
        Discovers all FLIR cameras.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of str
            List of FLIR serial numbers.
        """
        if PySpin is None:
            raise ImportError("PySpin not installed. Install to use FLIR cameras.")

        # Note: We don't close the SDK in info() because PySpin holds references
        # to cameras that prevent clean shutdown. The SDK will be cleaned up when
        # the last camera instance calls close_sdk() or at program exit.
        if FLIR.sdk is None:
            FLIR.sdk = PySpin.System.GetInstance()

        try:
            camera_list = FLIR.sdk.GetCameras()
            num_cameras = camera_list.GetSize()
            serial_list = []

            for i in range(num_cameras):
                cam = camera_list.GetByIndex(i)
                nodemap_tldevice = cam.GetTLDeviceNodeMap()
                node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
                node_model = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceModelName'))
                sn = node_serial.GetValue() if PySpin.IsReadable(node_serial) else f"cam_{i}"
                model = node_model.GetValue() if PySpin.IsReadable(node_model) else "unknown"
                serial_list.append(sn)
                if verbose:
                    print(f"  {i}: {sn} ({model})")
                # Don't hold references to individual cameras
                del cam

            if verbose and not serial_list:
                print("  No cameras found.")

            # Clear camera list
            camera_list.Clear()
            del camera_list

        except PySpin.SpinnakerException as ex:
            raise RuntimeError(f"Failed to enumerate cameras: {ex}")

        return serial_list

    @classmethod
    def close_sdk(cls):
        """
        Close the PySpin System instance.
        """
        if cls.sdk is not None:
            cls.sdk.ReleaseInstance()
            cls.sdk = None

    ### Internal Configuration Helpers ###

    def _configure_adc_depth(self, bitdepth=None):
        """
        Configure ADC bit depth and corresponding pixel format.

        Parameters
        ----------
        bitdepth : int or None
            Desired ADC bit depth (8, 10, or 12). If ``None``, selects the
            highest available ADC bit depth.

        Returns
        -------
        int
            The selected ADC bit depth (always 8, 10, or 12).
        """
        if bitdepth is not None and bitdepth not in (8, 10, 12):
            raise ValueError(f"Unsupported bitdepth {bitdepth}. Choose from 8, 10, or 12.")

        def _read_adc_depth():
            try:
                adc_str = self.cam.AdcBitDepth.ToString()  # e.g. "Bit12"
                adc_val = int(adc_str.replace("Bit", ""))
                if adc_val in (8, 10, 12):
                    return adc_val
            except Exception:
                pass
            return 8

        if self.cam.PixelFormat.GetAccessMode() != PySpin.RW:
            # Can't change format; infer ADC depth from current setting
            return _read_adc_depth()

        # Supported formats in descending bit depth order.
        # Only formats whose GetNDArray() returns a direct numpy array are listed;
        # packed formats (Mono12p, Mono10p) require ImageProcessor conversion and
        # are omitted in favor of Mono16 with the matching ADC depth.
        # Mono16 stores the ADC value left-shifted into the upper bits, so
        # _get_image_hw right-shifts the data back to the true ADC range.
        all_candidates = [
            (PySpin.PixelFormat_Mono16, PySpin.AdcBitDepth_Bit12, 12, "Mono16"),
            (PySpin.PixelFormat_Mono16, PySpin.AdcBitDepth_Bit10, 10, "Mono16"),
            (PySpin.PixelFormat_Mono8,  PySpin.AdcBitDepth_Bit8,   8, "Mono8"),
        ]

        if bitdepth is not None:
            # Filter to the requested ADC depth
            candidates = [(f, a, b, n) for f, a, b, n in all_candidates if b == bitdepth]
        else:
            candidates = all_candidates

        for pixel_fmt, adc_depth, bits, name in candidates:
            try:
                self.cam.PixelFormat.SetValue(pixel_fmt)
                # Set matching ADC bit depth if available
                try:
                    if self.cam.AdcBitDepth.GetAccessMode() == PySpin.RW:
                        self.cam.AdcBitDepth.SetValue(adc_depth)
                except PySpin.SpinnakerException:
                    pass
                logger.debug("PixelFormat set to %s (%s-bit)...", name, bits)
                return bits
            except PySpin.SpinnakerException:
                continue

        # Fallback
        logger.warning("Could not set preferred pixel format; using current setting.")
        return _read_adc_depth()

    def _read_trigger_is_software(self):
        """
        Read whether the camera is currently configured for software triggering.

        Cached into :attr:`_trigger_is_software` during initialization; call this
        again if the trigger configuration is changed outside this class. The cache
        exists because ``TriggerMode.GetValue()`` is a ~200 us device round-trip
        (unlike ``TriggerSource``, which GenICam serves from cache in under a
        microsecond), and :meth:`._get_image_hw` runs in the feedback loop.

        Returns
        -------
        bool
            ``True`` only if ``TriggerMode`` is on *and* ``TriggerSource`` is
            software. FLIR cameras report ``TriggerSource_Software`` even when
            ``TriggerMode`` is off, so the source alone is not sufficient.
        """
        try:
            return bool(
                self.cam.TriggerMode.GetValue() == PySpin.TriggerMode_On
                and self.cam.TriggerSource.GetValue() == PySpin.TriggerSource_Software
            )
        except PySpin.SpinnakerException:
            return False

    def _configure_frame_rate(self):
        """
        Configure the acquisition frame rate. Called during init and after WOI changes,
        since the maximum allowed frame rate depends on the current resolution.
        """
        # From the cache, not the device: __init__ reads the trigger configuration once
        # and this is also reached from set_woi() on every window change.
        triggered = self._trigger_is_software

        try:
            if self.cam.AcquisitionFrameRateEnable.GetAccessMode() == PySpin.RW:
                self.cam.AcquisitionFrameRateEnable.SetValue(not triggered)
            elif triggered and self.cam.AcquisitionFrameRateEnable.GetValue():
                # Read-only *and* still enabled: the limiter is stuck on and will
                # cost latency on every triggered capture. Worth surfacing, unlike
                # the common case where the node is simply not applicable.
                logger.warning(
                    "AcquisitionFrameRateEnable is enabled but not writable; the "
                    "frame rate limiter may add latency to each triggered capture."
                )
        except PySpin.SpinnakerException:
            pass  # Not all cameras have this node

        if triggered:
            logger.debug("Frame rate limiter disabled (software-triggered).")
            return

        try:
            if self.cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
                max_fps = self.cam.AcquisitionFrameRate.GetMax()
                self.cam.AcquisitionFrameRate.SetValue(max_fps)
                logger.debug("Frame rate set to %.1f Hz...", max_fps)
        except PySpin.SpinnakerException:
            pass  # Not all cameras support frame rate control

    ### Property Configuration ###

    def get_properties(self, verbose=True):
        """
        Print the list of camera properties using GenICam nodemap traversal.

        Parameters
        ----------
        verbose : bool
            Whether to print properties to console. If False, returns property dict.

        Returns
        -------
        dict or None
            Dictionary of {property_name: property_value} if verbose=False, else None.
        """
        properties = {}

        def traverse_category(category_node):
            """Recursively traverse category nodes to find actual properties."""
            try:
                features = category_node.GetFeatures()

                for feature in features:
                    # Skip if not readable
                    if not PySpin.IsReadable(feature):
                        continue

                    try:
                        # Check node type
                        node_type = feature.GetPrincipalInterfaceType()

                        # If it's a category, recurse into it
                        if node_type == PySpin.intfICategory:
                            category = PySpin.CCategoryPtr(feature)
                            print(f"\nCategory: {category.ToString()}\n")
                            traverse_category(category)
                        else:
                            # This is an actual property node - extract info
                            node = PySpin.CValuePtr(feature)
                            name = node.GetName()

                            # Try to get value as string
                            try:
                                value = node.ToString()
                            except Exception:
                                value = "N/A"

                            # Try to get unit
                            try:
                                unit = node.GetUnit()
                            except Exception:
                                unit = ""

                            # Try to get description
                            try:
                                description = node.GetToolTip()
                            except Exception:
                                description = ""

                            properties[name] = value

                            if verbose:
                                output = f"{name}\t{value}"
                                if unit:
                                    output += f"\t{unit}"
                                if description:
                                    output += f"\t{description}"
                                print(output)

                    except Exception:
                        continue

            except Exception:
                pass

        try:
            nodemap = self.cam.GetNodeMap()

            # Get root category
            root = PySpin.CCategoryPtr(nodemap.GetNode("Root"))
            if not PySpin.IsReadable(root):
                if verbose:
                    print("Unable to access camera properties")
                return properties if not verbose else None

            # Recursively traverse all categories to find properties
            traverse_category(root)

        except PySpin.SpinnakerException as ex:
            if verbose:
                print(f"Error accessing properties: {ex}")

        return properties if not verbose else None

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        return float(self.cam.ExposureTime.GetValue()) / 1e6

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        exposure_us = float(exposure_s * 1e6)

        if self.exposure_bounds_s is not None:
            min_us = self.exposure_bounds_s[0] * 1e6
            max_us = self.exposure_bounds_s[1] * 1e6
            exposure_us = max(min_us, min(max_us, exposure_us))

        self.cam.ExposureTime.SetValue(exposure_us)

    def _set_woi_hw(self, woi):
        """See :meth:`.Camera._set_woi_hw`. **(Untested)**"""
        # FLIR ROI coordinates (OffsetX/Width/Height) are in binned pixels when binning is active.
        # https://softwareservices.flir.com/BFS-U3-89S6/latest/Model/public/ImageFormatControl.html
        x, w, y, h = [int(v) for v in woi]

        def _snap(node, value):
            try:
                inc = node.GetInc()
                return (value // inc) * inc
            except Exception:
                return value

        x = _snap(self.cam.OffsetX, x)
        y = _snap(self.cam.OffsetY, y)
        w = _snap(self.cam.Width, w)
        h = _snap(self.cam.Height, h)

        acquisition_active = False
        try:
            if self.cam.IsStreaming():
                self.cam.EndAcquisition()
                acquisition_active = True
        except PySpin.SpinnakerException:
            pass

        try:
            if self.cam.OffsetX.GetAccessMode() == PySpin.RW:
                self.cam.OffsetX.SetValue(0)
            if self.cam.OffsetY.GetAccessMode() == PySpin.RW:
                self.cam.OffsetY.SetValue(0)
            if self.cam.Width.GetAccessMode() == PySpin.RW:
                self.cam.Width.SetValue(w)
            if self.cam.Height.GetAccessMode() == PySpin.RW:
                self.cam.Height.SetValue(h)
            if self.cam.OffsetX.GetAccessMode() == PySpin.RW:
                self.cam.OffsetX.SetValue(x)
            if self.cam.OffsetY.GetAccessMode() == PySpin.RW:
                self.cam.OffsetY.SetValue(y)
            self._configure_frame_rate()
        except PySpin.SpinnakerException as ex:
            raise RuntimeError(f"Failed to set WOI: {ex}")
        finally:
            if acquisition_active:
                try:
                    self.cam.BeginAcquisition()
                except PySpin.SpinnakerException as ex:
                    raise RuntimeError(f"Failed to restart acquisition after WOI change: {ex}")

    def _get_woi_hw(self):
        """See :meth:`.Camera._get_woi_hw`. **(Untested)**"""
        return (
            int(self.cam.OffsetX.GetValue()),
            int(self.cam.Width.GetValue()),
            int(self.cam.OffsetY.GetValue()),
            int(self.cam.Height.GetValue()),
        )

    def _set_binning_hw(self, binning):
        """See :meth:`.Camera._set_binning_hw`. **(Untested)**"""
        binx, biny = int(binning[0]), int(binning[1])
        acquisition_active = False
        try:
            if self.cam.IsStreaming():
                self.cam.EndAcquisition()
                acquisition_active = True
        except PySpin.SpinnakerException:
            pass
        try:
            nodemap = self.cam.GetNodeMap()
            bh = PySpin.CIntegerPtr(nodemap.GetNode("BinningHorizontal"))
            bv = PySpin.CIntegerPtr(nodemap.GetNode("BinningVertical"))
            if not PySpin.IsWritable(bh) or not PySpin.IsWritable(bv):
                raise NotImplementedError(f"Camera {self.name} does not support binning.")
            bh.SetValue(binx)
            bv.SetValue(biny)
        except PySpin.SpinnakerException as ex:
            raise NotImplementedError(f"Camera {self.name} does not support binning: {ex}")
        finally:
            if acquisition_active:
                try:
                    self.cam.BeginAcquisition()
                except PySpin.SpinnakerException:
                    pass

    def _get_binning_hw(self):
        """See :meth:`.Camera._get_binning_hw`."""
        try:
            nodemap = self.cam.GetNodeMap()
            bh = PySpin.CIntegerPtr(nodemap.GetNode("BinningHorizontal"))
            bv = PySpin.CIntegerPtr(nodemap.GetNode("BinningVertical"))
            return (int(bh.GetValue()), int(bv.GetValue()))
        except PySpin.SpinnakerException:
            return (1, 1)

    def _get_image_hw(self, timeout_s = 1.0):
        """
        See :meth:`.Camera._get_image_hw`.

        If the camera is in software trigger mode, executes a software trigger
        before capturing. Otherwise, waits for an externally triggered frame.

        Parameters
        ----------
        timeout_s : float
            Timeout in seconds.
        """

        try:
            # Only fire software trigger if in software trigger mode; an externally
            # triggered camera must not be force-triggered here. Read from the cache
            # rather than the device: the check needs TriggerMode, which costs ~200 us
            # per query, and this runs once per feedback iteration.
            if self._trigger_is_software:
                self.cam.TriggerSoftware.Execute()

            # Get image (software-triggered or externally triggered).
            frame = self.cam.GetNextImage(int(timeout_s * 1e3))

            # Check if image is incomplete
            if frame.IsIncomplete():
                status = frame.GetImageStatus()
                frame.Release()
                raise RuntimeError(f"Image incomplete with status {status}")

            # Get numpy array from image
            image_data = np.copy(frame.GetNDArray())

            # Release frame to free buffer
            frame.Release()

            # Mono16 stores ADC values left-shifted into the upper bits of the
            # 16-bit word.  Right-shift back so that values span [0, 2**bitdepth)
            # and normalization by bitresolution is correct.
            if image_data.dtype == np.uint16 and self.bitdepth < 16:
                image_data = np.right_shift(image_data, 16 - self.bitdepth)

            return image_data

        except PySpin.SpinnakerException as ex:
            raise RuntimeError(f"Camera acquisition failed: {ex}")
