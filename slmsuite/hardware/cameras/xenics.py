"""
Hardware control for Xenics camera via the :mod:`Xeneth` interface.

Python wrapper for Xenith C++ SDK 2.7 from Xenics; see xeneth-sdk.chm
there for additional documentation.
"""
import time
from ctypes import *

import numpy as np

from slmsuite.hardware.cameras.camera import Camera
from slmsuite._logging import make_logger

logger = make_logger(__name__)

CLKFREQ = 80e6

### Imaging Constants from XFilters.h ###

# pylint: disable=C0301
# fmt: off
#ErrorCodes
I_OK                    = 0        # Success.
I_DIRTY                 = 1        # Internal.
E_BUG                   = 10000    # Generic.
E_NOINIT                = 10001    # Camera was not successfully initialised.
E_LOGICLOADFAILED       = 10002    # Invalid logic file.
E_INTERFACE_ERROR       = 10003    # Command interface failure.
E_OUT_OF_RANGE          = 10004    # Provided value is incapable of being produced by the hardware.
E_NOT_SUPPORTED         = 10005    # Functionality not supported by this camera.
E_NOT_FOUND             = 10006    # File/Data not found.
E_FILTER_DONE           = 10007    # Filter has finished processing and will be removed.
E_NO_FRAME              = 10008    # A frame was requested by calling GetFrame but none was available.
E_SAVE_ERROR            = 10009    # Couldn't save to file.
E_MISMATCHED            = 10010    # Buffer size mismatch.
E_BUSY                  = 10011    # The API can not read a temperature because the camera is busy.
E_INVALID_HANDLE        = 10012    # An unknown handle was passed to the C API.
E_TIMEOUT               = 10013    # Operation timed out.
E_FRAMEGRABBER          = 10014    # Frame grabber error.
E_NO_CONVERSION         = 10015    # GetFrame could not convert the image data to the requested format.
E_FILTER_SKIP_FRAME     = 10016    # Filter indicates the frame should be skipped.
E_WRONG_VERSION         = 10017    # Version mismatch.
E_PACKET_ERROR          = 10018    # The requested frame cannot be provided because at least one packet has been lost.
E_WRONG_FORMAT          = 10019    # The emissivity map you tried to set should be a 16 bit greyscale PNG.
E_WRONG_SIZE            = 10020    # The emissivity map you tried to set has the wrong dimensions (width height).
E_CAPSTOP               = 10021    # Internal
E_OUT_OF_MEMORY         = 10022    # An allocation failed because the system ran out of memory.

#ColourMode
COLOURMODE_8            = 0    # Intensity only
COLOURMODE_16           = 1    # Alias
COLOURMODE_PROFILE      = 2    # Uses a colour profile bitmap. See #LoadColourProfile()
COLOURMODE_INVERT       = 256   # Set this flag if an inversion of the colour profile is desired. eg: #ColourMode_8 | #ColourMode_Invert

#BlitType
WINDOW          = 0    # Blit the contents of the last captured frame directly to a Windows client device context using a Window handle (HWND) */
DEVICECONTEXT   = 1    # Blit the contents of the last captured frame to a specified device context.
                       #        This can be any device context (HDC) like a memory DC paint DC or a handle to a DC associated with a Graphics-object (C#)   */

#FrameType
FT_UNKNOWN      = -1   # Unknown invalid frame type
FT_NATIVE       = 0    # The native frame type of this camera (can be FT_8..,FT_16..,FT32.. check GetFrameType())
FT_8_BPP_GRAY   = 1    # 8-bit greyscale
FT_16_BPP_GRAY  = 2    # 16-bit greyscale (default for most of the Xenics branded cameras)
FT_32_BPP_GRAY  = 3    # 32-bit greyscale
FT_32_BPP_RGBA  = 4    # 32-bit colour RGBA      [B,G,R,A] Available for output conversion.
FT_32_BPP_RGB   = 5    # 32-bit colour RGB       [B,G,R]   Available for output conversion.
FT_32_BPP_BGRA  = 6    # 32-bit colour BGRA      [R,G,B,A]
FT_32_BPP_BGR   = 7     # 32-bit colour BGR       [R,G,B]

#XEnumerationFlag
XEF_NETWORK         = 0x00000001   # Network
XEF_SERIAL          = 0x00000002   # Serial
XEF_CAMERALINK      = 0x00000004   # CameraLink
XEF_GIGEVISION      = 0x00000008   # GigEVision
XEF_COAXPRESS       = 0x00000010   # CoaXPress
XEF_USB             = 0x00000020   # USB
XEF_USB3VISION      = 0x00000040   # USB3Vision
XEF_GENCP           = 0x00000080   # CameraLink GenCP
XEF_ENABLEALL       = 0x0000FFFF   # Enable all protocols.
XEF_USECACHED       = 0x01000000   # Use cached devices on enumeration.
XEF_RELEASECACHE    = 0x02000000   # Release internally cached devices.

#XFilterMessage
XMSGINIT                = 0    # [API->Filter Event] Called when the filter is being installed  ( (!) calling thread context)         */
XMSGCLOSE               = 1    # [API->Filter Event] Called when the filter is being removed    ( (!) calling thread context)         */
XMSGFRAME               = 2    # [API->Filter Event] Called after every frame grab              ( (!) grabbing thread context)        */
XMSGGETNAME             = 3    # [App->Filter Event] Retrieve filter name: the filter should copy a friendly string to msgparm        */
XMSGGETVALUE            = 4    # [Obsolete]                                                                                           */
XMSGSAVE                = 5    # [Obsolete]                                                                                           */
XMSGGETSTATUS           = 6    # [API->Filter Event] Retrieves a general purpose status message from the image filter                 */
XMSGUPDATEVIEWPORT      = 7    # [API->Filter Event] Instructs an image correction filter to update it's view port
                               #                            This message is sent to a filter upon changing the window of interest or when
                               #                            flipping image horizontally or vertically                                        */
XMSGCANPROCEED          = 8    # Used by image filters in in interactive mode to indicate acceptable image conditions                 */
XMSGGETINFO             = 9    # [Internal]          Used to query filter 'registers'                                                 */
XMSGSELECT              = 10   # [Obsolete]                                                                                           */
XMSGPROCESSEDFRAME      = 11   # [API->Filter Event] Sent after other filters have done their processing. Do not modify the frame data
                               #                            in response to this event.                                                       */
XMSGTIMEOUT             = 13   # [API->Filter Event] A camera time-out event was generated                                             */
XMSGISBUSY              = 16   # [Thermography]      Is the temperature filter recalculating - Used to check if the thermal filter is
                               #                            still updating it's linearisation tables                                         */
XMSGSETTROI             = 17   # [Imaging/Thermo]    Set the adu/temperature span in percent (see #XMsgSetTROIParms)                 */
XMSGLOAD                = 18   # [Obsolete]                                                                                           */
XMSGUNLOAD              = 19   # [Obsolete]                                                                                           */
XMSGADUTOTEMP           = 12   # [Thermography]      Convert an ADU value to a temperature (see #XFltADUToTemperature)                 */
XMSGGETEN               = 14   # [Obsolete]          Get temperature correction parameters (see #XMsgGetRadiometricParms)              */
XMSGSETEN               = 15   # [Obsolete]          Set temperature correction parameters (see #XMsgGetRadiometricParms)              */
XMSGTEMPTOADU           = 20   # [Thermography]      Convert a temperature to an ADU value (see #XFltTemperatureToADU)                 */
XMSGGETTVALUE           = 21   # [Thermography]      Retrieve an emissivity corrected value from a coordinate                         */
XMSGGETRADIOMETRICPARMS = 22   # [Thermography]      Get temperature correction parameters (see #XMsgTempParms)                        */
XMSGSETRADIOMETRICPARMS = 23   # [Thermography]      Set temperature correction parameters (see #XMsgTempParms)                        */
XMSGSERIALISE           = 100  # [App->Filter event] Serialise internal parameter state (write xml structure) see #XFltSetParameter    */
XMSGDESERIALISE         = 101  # [App->Filter event] Deserialise parameter state (read xml structure) see #XFltSetParameter            */
XMSGGETPRIORITY         = 102  # [Filter Management] Write the current filter priority to the long * provided in v_pMsgParm           */
XMSGSETFILTERSTATE      = 104  # [Filter Management] Enable or disable an image filter temporarily by sending 0/1 in v_pMsgParm       */
XMSGISSERIALISEDIRTY    = 105  # [Internal]                                                                                           */
XMSGSTOREHANDLE         = 106  # [Internal]          Start tracking the module handle for plugin image filters                        */
XMSGUPDATETINT          = 107  # [API->Filter event] Integration time change notification                                             */
XMSGLINADUTOTEMP        = 109  # [Thermography]      Convert a Linearized ADU value to a temperature (see #XFltADUToTemperatureLin)    */
XMSGLINTEMPTOADU        = 110  # [Thermography]      Convert a temperature to a Linearized ADU value (see #XFltTemperatureToADULin)    */
XMSGUPDATESPAN          = 111  # [API->Filter event] Span change notification                                                         */
XMSGUPDATEPALETTE       = 112  # [API->Filter event] Colour profile change notification                                               */
XMSGFILTERQUEUED        = 113  # [API->Filter event] A filter is queued                                                               */
XMSGFILTERREMOVED       = 114  # [API->Filter event] A filter is removed                                                              */
XMSGDRAWOVERLAY         = 200  # [API->Filter event] Draw the RGBA frame overlay, v_pMsgParm is the pointer to the RGBA data
                               #                            structure                                                                        */
XMSGLINEARISEOUTPUT     = 201  # [Thermography]      When specifying a v_pMsgParm that is non zero, starts linearising adu output     */
XMSGSETEMIMAP           = 202  # [Thermography]      Streams the main emissivity map to the thermal filter (16 bit png, 65535 = 1.0)  */
XMSGSETEMIMAPUSER       = 203  # [Thermography]      Stream a user emissivity map to the thermal filter (16 bit png, 65535 = 1.0,
                                #                            0 values are replaced by the emissivity in the main map)                         */
XMSGGETEMIMAP           = 204  # [Thermography]      Stream out the combined emissivity map                                           */
XMSGCLREMIMAP           = 205  # [Thermography]      Clear emissivity map                                                             */
XMSGCLREMIMAPUSER       = 206  # [Thermography]      Clear emissivity map (user)                                                      */
XMSGPUSHRANGE           = 207  # [Thermography]      Push a new linearization range to the thermal filter                             */
XMSGTHMFILTERSTATE      = 208  # [Thermography]      Filter event indicating thermal filter queue/removal                             */
XMSGTHMADJUSTSET        = 209  # [Thermography]      Set global offset & gain adu adjustment (pre-thermal conversion)                 */
XMSGTHMADJUSTGET        = 210  # [Thermography]      (see #XMsgTempAdjustmentParms)                                                      */

XMSGLOG                 = 211  # [Plugin->API]       Fire a log event to the end user application\n
                               #                            Target filter id: 0xffffffff                                                */
XMSGGETDELTAT           = 212  # [Internal]                                                                                      */
XMSGGETTINTRANGE        = 213  # [Plugin->API]       Request the exposure time range
                               #                            Target filter id: 0xffffffff                                                 */
XMSGCORRECTIONDIRTY         = 214  # [Internal]          The onboard thermography parameters have changed                             */
XMSGHASRADIANCEINFO         = 215  # [Thermography]      Check if the radiance information is available. This is needed to for emissivity correction */
XMSGCORRECTIONDATACHANGED   = 216  # [Internal]          New correction data is loaded                             */
XMSGPOSTPROCESS             = 217  # [Internal]          A post processing step is introduced in the software correction filter */

XMSGZOOMLENSCONNECT     = 300  # [Zoom lens]         Connect to the zoom lens on the specified port.  */
XMSGZOOMLENSGETSTATE    = 301  # [Zoom lens]         Get the current zoom/focus state from the zoom lens filter.  */
XMSGZOOMLENSSETSTATE    = 302  # [Zoom lens]         Set the current zoom/focus state in the zoom lens filter.    */
XMSGZOOMLENSGETINFO     = 303  # [Zoom lens]         Get some descriptive information about the connected lens.   */

XMSGUSER                = 24200 # If you develop your own image filter plugins, please use this constant to offset your messages. */

#XStatusMessage
XSLOADLOGIC         = 1    # Passed when loading the camera's main logic file                           */
XSLOADVIDEOLOGIC    = 2    # Passed when loading the camera's video output firmware                     */
XSDATASTORAGE       = 3    # Passed when accessing persistent data on the camera                        */
XSCORRECTION        = 4    # Passed when uploading correction data to the camera                        */
XSSELFSTART         = 5    # Passed when a self starting camera is starting (instead of XSLoadLogic)    */
XSMESSAGE           = 6    # String event
                                #  This status message is used to relay critical errors and events originating
                                #  from within the API.
                                #  Cam|PropLimit|property=number - A filter notifies you your user interface should limit the value of 'property' to 'number'
                                #  Cam|TemperatureFilter|RangeUpdate       - The thermography filter uses this to notify you of a span update.
                                #  Cam|TemperatureFilter|Off               - The thermography filter suggests the application to dequeue the filter.
                                #  Cam|InterfaceUpdate           - Internal, do not handle, returning E_BUG here causes the API to stop unpacking 'abcd.package'.packages to %appdata%/xenics/interface
XSLOADGRABBER       = 7    # Passed when loading the framegrabber                                                                                */
XSDEVICEINFORMATION = 8    # Device information passed when connecting a device, ulP is the lower part of the address. When using 64-bit the higher part of the address is stored in ulT */

#XGetFrameFlags
XGF_BLOCKING    = 1    # In blocking-mode the method does not return immediately with the return codes #E_NO_FRAME / #I_OK.
                       #  Instead the method waits for a frame and only returns until a frame was captured, or a time-out period has elapsed.

XGF_NOCONVERSION= 2    # Prevents internal conversion to 8 bit, specifying this flag reduces computation time, but prevents #SaveData() and the #Blit() method from working.
XGF_FETCHPFF    = 4    # Retrieve the per frame footer with frame timing information. Call XCamera::GetFrameFooterLength() to determine the increase in frame size.
XGF_RFU_1       = 8
XGF_RFU_2       = 16
XGF_RFU_3       = 32

#XSaveDataFlags
XSD_FORCE16             = 1    # Forces 16-bit output independent of the current #ColourMode-setting (only possible for PNG's)
XSD_FORCE8              = 2    # Forces 8-bit output independent of the current #ColourMode
XSD_ALIGNLEFT           = 4    # Left aligns 16-bit output (#XSD_Force16 | #XSD_AlignLeft)
XSD_SAVETHERMALINFO     = 8    # Save thermal conversion structure (only available when saving 16-bit PNGs)
XSD_RFU_0               = 16   # Reserved
XSD_RFU_1               = 32
XSD_RFU_2               = 64
XSD_RFU_3               = 128

#XSaveSettingFlags
XSS_SAVECAMERAPROPS     = 1    # Define property sources to save settings from.
XSS_SAVEGRABBERPROPS    = 2    #
XSS_SAVEALLPROPS        = 4    # Also save properties marked 'No persist'.
XSS_SS_RFU_3            = 8    #

#XLoadSettingsFlags
XSS_IGNORENAIS        = 1    # Ignore properties which do not affect the image.
XSS_LS_RFU_1          = 2    #
XSS_LS_RFU_2          = 4    #
XSS_LS_RFU_3          = 8    #

#XLoadCalibrationFlags
XLC_STARTSOFTWARECORRECTION     = 1    # Starts the software correction filter after unpacking the calibration data
XLC_RFU_1                       = 2
XLC_RFU_2                       = 4
XLC_RFU_3                       = 8

#XPropType
XTYPE_NONE              = 0x00000000

XTYPE_BASE_MASK         = 0x000000ff   # Type mask
XTYPE_ATTR_MASK         = 0xffffff00   # Attribute mask
XTYPE_BASE_NUMBER       = 0x00000001   # A number (floating)
XTYPE_BASE_ENUM         = 0x00000002   # An enumerated type (a choice)
XTYPE_BASE_BOOL         = 0x00000004   # Boolean (true / false / 1 / 0)
XTYPE_BASE_BLOB         = 0x00000008   # Binary large object
XTYPE_BASE_STRING       = 0x00000010   # String
XTYPE_BASE_ACTION       = 0x00000020   # Action (button)
XTYPE_BASE_RFU1         = 0x00000040   # RFU
XTYPE_BASE_RFU2         = 0x00000080   # RFU

XTYPE_BASE_MINMAX       = 0x00002000   # The property accepts the strings 'min' and 'max' to set the best achievable extremities.
XTYPE_BASE_READONCE     = 0x00001000   # Property needs to be read at start-up only
XTYPE_BASE_NOPERSIST    = 0x00000800   # Property shouldn't be persisted (saved & restored)
XTYPE_BASE_NAI          = 0x00000400   # Property does not affect image intensity level ('Not Affecting Image')
XTYPE_BASE_RW           = 0x00000300   # Write and read back
XTYPE_BASE_WRITEABLE    = 0x00000200   # Writeable properties have this set in their high byte
XTYPE_BASE_READABLE     = 0x00000100   # Readable properties have this set in their high byte

XTYPE_NUMBER            = 0x00000201   # Write only number
XTYPE_ENUM              = 0x00000202   # Write only enumeration
XTYPE_BOOL              = 0x00000204   # Write only boolean
XTYPE_BLOB              = 0x00000208   # Write only binary large object
XTYPE_STRING            = 0x00000210   # Write only string
XTYPE_ACTION            = 0x00000220   # Action (button)

XTYPE_RO_NUMBER         = 0x00000101   # Read only number
XTYPE_RO_ENUM           = 0x00000102   # Read only enumeration
XTYPE_RO_BOOL           = 0x00000104   # Read only boolean
XTYPE_RO_BLOB           = 0x00000108   # Read only binary large object
XTYPE_RO_STRING         = 0x00000110   # Read only string

XTYPE_RW_NUMBER         = 0x00000301   # R/W number
XTYPE_RW_ENUM           = 0x00000302   # R/W enumeration
XTYPE_RW_BOOL           = 0x00000304   # R/W boolean
XTYPE_RW_BLOB           = 0x00000308   # R/W binary large object
XTYPE_RW_STRING         = 0x00000310   # R/W string

#XDirectories
XDIR_FILTERDATA         = 0x0000   # Filter data (%APPDATA%/XenICs/Data/&lt;sessionnumber&gt;)
XDIR_SCRIPTROOT         = 0x0001   # Script root (%APPDATA%/XenICs/Interface/&lt;PID-number&gt;)
XDIR_CALIBRATIONS       = 0x0002   # Calibration folder (%ProgramFiles%/Xeneth/Calibrations)
XDIR_INSTALLDIR         = 0x0003   # Installation folder (%CommonProgramFiles%/XenICs/Runtime)
XDIR_PLUGINS            = 0x0004   # Plugin folder (%CommonProgramFiles%/XenICs/Runtime/Plugins)
XDIR_CACHEPATH          = 0x0005   # Cache folder (%APPDATA%/XenICs/Cache)
XDIR_SDKRESOURCES       = 0x0006   # SDK resource folder (%CommonProgramFiles%/XenICs/Runtime/Resources)
XDIR_XENETH             = 0x0007   # Xeneth installation directory
XDIR_GRABBERSCRIPTROOT  = 0x0008   # Script root (%APPDATA%/XenICs/Interface/&lt;FrameGrabber&gt;)

#XDeviceStates
XDS_AVAILABLE = 0x0    # The device is available to establish a connection.
XDS_BUSY = 0x1         # The device is currently in use.
XDS_UNREACHABLE = 0x2  # The device was detected but is unreachable.
# pylint: enable=C0301
# fmt: on


class _XDeviceInformation(Structure):
    """Structure to be filled with camera data from the dll interface."""

    _fields_ = [
        ("size", c_int),
        ("name", (c_char * 64)),
        ("transport", (c_char * 64)),
        ("url", (c_char * 256)),
        ("address", (c_char * 64)),
        ("serial", c_uint),
        ("pid", c_uint),
        ("state", c_uint),
    ]


class Cheetah640(Camera):
    """
    Xeneth's Cheetah640 camera.

    Attributes
    ----------
    xeneth : windll
        Object to talk with the Xeneth SDK.
    cam : ptr
        Object to talk with the desired camera.
    profile : {'triggered', 'free'}
        Current pre-configured operation mode.

        ``'triggered'`` - Captures on external trigger \n
        ``'free'`` - Continuous capture

    frame_size : int
        Size of the frame in bytes for use in grabbing data for the buffer.
    frame_buffer : c_ushort array
        Buffer used to grab data from the camera and sdk interface.
    last_capture : numpy.ndarray
        Previous capture by get_image()
    last_tag : numpy.uint16
        Frame tag of last capture
    last_process_time : float
        Time to process last frame
    filters : dict
        Dictionary with enabled filters. See Xeneth documentation for more information.
    """

    ### Camera Interface ###

    def __init__(
            self,
            virtual=False,
            temperature=None,
            pitch_um=(20,20),
            **kwargs
        ):
        """
        Initialize camera. Default ``profile`` is ``'free'``.

        Parameters
        ----------
        virtual : bool
            Whether or not the camera is virtual.
        temperature : float or None
            Temperature in degrees Celsius to set on startup. ``None`` defaults to no cooling.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations. Defaults to 20 micron square pixels.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.

        Raises
        ------
        RuntimeError
            If the camera is not reachable.
        """
        # Load the SDK
        self.xeneth = windll.LoadLibrary("xeneth64.dll")

        # Enumerate devices
        dev_count = c_uint()
        _ = self.xeneth.XCD_EnumerateDevices(None, byref(dev_count), XEF_CAMERALINK)
        device = _XDeviceInformation()
        _ = self.xeneth.XCD_EnumerateDevices(
            byref(device), byref(dev_count), XEF_USECACHED
        )

        # Open the first avaliable camera
        if virtual:
            logger.debug("Opening connection to virtual camera")
            self.cam = self.xeneth.XC_OpenCamera(b"soft://0", 0, 0)
        else:
            if device.name.decode():
                logger.debug("Opening connection to %s", device.name.decode())
                self.cam = self.xeneth.XC_OpenCamera(device.url, 0, 0)
                name = device.name.decode()
            else:
                raise RuntimeError(
                    "Camera not reachable! Close Xeneth GUI or check connections."
                )

        # Check that the camera opened properly
        if self.xeneth.XC_IsInitialised(self.cam):
            logger.debug("Camera initialized, loading properties...")

            super().__init__(
                self.xeneth.XC_GetWidth(self.cam),
                self.xeneth.XC_GetHeight(self.cam),
                bitdepth=12,
                pitch_um=pitch_um,
                name=name,
                **kwargs
            )

            # Initialize a 16-bit buffer for a single frame; initialize to 0
            self.frame_size = self.xeneth.XC_Getframe_size(self.cam)
            self.frame_buffer = (c_ushort * int(self.frame_size / 2))(0)
            self.last_capture = np.empty(self.shape)
            self.last_tag = np.uint16(0)  # Frame tag of last capture
            self.last_process_time = 0  # Time to process last frame

            # Make a filter dictionary
            self.filters = {}

            # Setup base imaging configuration
            if not virtual:
                # Starts a circular buffer w/ 4k frames (anything larger crashes???)
                self.setup_grabber(frames=4000)
                # Max out the API buffer to
                self.set_buffer_api(frames=64)
                # Enable frame tags (frame number in first two pixels)
                self.enable_frametags(True)
                if temperature is None:
                    # Disable high fan speed for stability
                    self.enable_cooling(False)
                else:
                    self.enable_cooling()
                    # Set Peltier setpoint
                    self.set_temperature(temperature)
                    # Flip X, Flip Y readout by default (used in onboard cal)
                self.set_readout_orientation(True, True)
                # Enable output trigger to readout integration periods
                self.setup_output_trigger()
                # Set max frame rate to basically unlimited (exposure-limited)
                self.set_framerate(0)

                # Load visible setup by default
                self.setup("free")
        else:
            logger.error("Initialization failed.")

    def close(self):
        """See :meth:`.Camera.close`."""
        if self.xeneth.XC_IsInitialised(self.cam):
            self.logger.info("Closing connection to camera...")
            self.xeneth.XC_CloseCamera(self.cam)
        else:
            self.logger.warning("Camera not open!")

    ### Property Configuration ###

    def get_property_status(self, save_file_path=None, verbose=True):
        """
        List and get status of all addressable properties on camera.

        Parameters
        ----------
        save_file_path : str or None
            If not ``None``, the property status results will be saved to this path.
        verbose : bool
            Prints property results if ``True``.
        """
        if self.xeneth.XC_IsInitialised(self.cam):
            if verbose:
                print("Getting camera properties...")
            property_count = self.xeneth.XC_GetPropertyCount(self.cam)
            property_name = (c_char * 128)(0)
            property_category = (c_char * 128)(0)
            property_range = (c_char * 128)(0)
            property_unit = (c_char * 128)(0)
            property_type = c_int()  # XType_None
            lvalue = c_long(0)
            fvalue = c_double(0)
            cvalue = c_char(0)

            # Iterate over each property and output details such as name, type, value
            for x in range(property_count):
                self.xeneth.XC_Getproperty_name(self.cam, x, property_name, 128)
                self.xeneth.XC_Getproperty_category(
                    self.cam, property_name, property_category, 128
                )
                self.xeneth.XC_Getproperty_type(
                    self.cam, property_name, byref(property_type)
                )
                self.xeneth.XC_Getproperty_range(
                    self.cam, property_name, property_range, 128
                )
                self.xeneth.XC_Getproperty_unit(
                    self.cam, property_name, property_unit, 128
                )

                if verbose:
                    print(
                        "Property[%d]    Category: %s"
                        % (x, property_category.value.decode())
                    )
                    print(
                        "Property[%d]        Name: %s"
                        % (x, property_name.value.decode())
                    )
                    print(
                        "Property[%d]       Flags: %s"
                        % (
                            x,
                            (
                                "MinMax | "
                                if property_type.value & XTYPE_BASE_MINMAX
                                else ""
                            )
                            + (
                                "ReadOnce | "
                                if property_type.value & XTYPE_BASE_READONCE
                                else ""
                            )
                            + (
                                "NoPersist | "
                                if property_type.value & XTYPE_BASE_NOPERSIST
                                else ""
                            )
                            + ("NAI | " if property_type.value & XTYPE_BASE_NAI else "")
                            + (
                                "Writeable | "
                                if property_type.value & XTYPE_BASE_WRITEABLE
                                else ""
                            )
                            + (
                                "Readable | "
                                if property_type.value & XTYPE_BASE_READABLE
                                else ""
                            ),
                        )
                    )

                # The following output depends on the property type.
                type_num = property_type.value & XTYPE_BASE_MASK

                # Write-only number
                if type_num == XTYPE_BASE_NUMBER:
                    # Check camera doc for float vs. long
                    self.xeneth.XC_GetPropertyValueL(
                        self.cam, property_name, byref(lvalue)
                    )
                    self.xeneth.XC_GetPropertyValueF(
                        self.cam, property_name, byref(fvalue)
                    )
                    if verbose:
                        print("Property[%d]        Type: Number" % x)
                        print(
                            "Property[%d]       Range: %s"
                            % (x, property_range.value.decode())
                        )
                        print(
                            "Property[%d]        Unit: %s"
                            % (x, property_unit.value.decode())
                        )
                        print("Property[%d]  Long value: %lu" % (x, lvalue.value))
                        print("Property[%d] Float value: %f" % (x, fvalue.value))

                # Write-only enumeration
                elif type_num == XTYPE_BASE_ENUM:
                    self.xeneth.XC_GetPropertyValueL(
                        self.cam, property_name, byref(lvalue)
                    )
                    if verbose:
                        print("Property[%d]        Type: Enum" % x)
                        print(
                            "Property[%d]       Range: %s"
                            % (x, property_range.value.decode())
                        )
                        print("Property[%d]       Value: %lu" % (x, lvalue.value))

                # Boolean TF
                elif type_num == XTYPE_BASE_BOOL:
                    self.xeneth.XC_GetPropertyValueL(
                        self.cam, property_name, byref(lvalue)
                    )
                    if verbose:
                        print("Property[%d]        Type: Bool" % x)
                        print(
                            "Property[%d]       Value: %s"
                            % (x, "True" if lvalue.value == 1 else "False")
                        )

                # Binary large object (BLOB)
                elif type_num == XTYPE_BASE_BLOB:
                    if verbose:
                        print("Skipping BLOB information collection to save time...")

                # String
                elif type_num == XTYPE_BASE_STRING:
                    # Get the string.
                    cvalue = (c_char * 128)(0)
                    self.xeneth.XC_GetPropertyValue(
                        self.cam, property_name, cvalue, 128
                    )
                    if verbose:
                        print("Property[%d]        Type: String" % x)
                        print("Property[%d]       Value: %s" % (x, cvalue.value))

                if verbose:
                    print("")

                # Save current settings to file if desired
                if save_file_path is not None:
                    fname = save_file_path + ".xcf"
                    print('Saving settings to file "{}"...'.format(fname))
                    self.xeneth.XC_SaveSettings(self.cam, fname)
        else:
            print("Camera not open!")

    def configure(self, format_file):
        """
        Loads pre-stored imaging profile from XC_SaveSettings XCF file

        Parameters
        ----------
        format_file : str
            Path to XCF file containing format information
        """
        if self.xeneth.XC_IsInitialised(self.cam):
            self.logger.info("Loading settings from %s.xcf ...", format_file)
            self.xeneth.XC_LoadSettings(self.cam, format_file)
        else:
            self.logger.warning("Camera not open!")

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        exposure_old = c_double(0.0)
        err1 = self.xeneth.XC_GetPropertyValueF(
            self.cam, b"IntegrationTime", byref(exposure_old)
        )
        if err1:
            self.logger.warning("Error encountered! Error code: %d", err1)
        return exposure_old.value / 1e6

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        exposure_old = c_double(0.0)
        exposure = c_double(exposure_s * 1e6)  # us
        err1 = self.xeneth.XC_GetPropertyValueF(
            self.cam, b"IntegrationTime", byref(exposure_old)
        )
        err2 = self.xeneth.XC_SetPropertyValueF(
            self.cam, b"IntegrationTime", exposure, ""
        )
        err3 = self.xeneth.XC_GetPropertyValueF(
            self.cam, b"IntegrationTime", byref(exposure)
        )
        if err1 or err2 or err3:
            self.logger.warning(
                "Error encountered! Error codes: %d, %d, %d", err1, err2, err3
            )

    def set_framerate(self, framerate):
        """
        Set the camera framerate.

        Parameters
        ----------
        framerate : int
            The framerate in fps.
        """
        rate = c_long(int(framerate))
        self.logger.info("Setting frame rate to %d Hz...", rate.value)
        rate_old = c_long(0)
        err1 = False
        err1 = self.xeneth.XC_GetPropertyValueL(self.cam, b"FrameRate", byref(rate_old))
        err2 = self.xeneth.XC_SetPropertyValueL(self.cam, b"FrameRate", rate, "")
        err3 = self.xeneth.XC_GetPropertyValueL(self.cam, b"FrameRate", byref(rate))
        self.logger.info(
            "Previous frame rate: %d Hz; new frame rate: %d Hz", rate_old.value, rate.value
        )
        if err1 or err2 or err3:
            self.logger.warning(
                "Error encountered! Error codes: %d, %d, %d", err1, err2, err3
            )

    def get_frame_footer_length(self):
        """
        Get the length of software frame tags.

        Returns
        -------
        int
        """
        return self.xeneth.XC_GetFrameFooterLength(self.cam)

    def set_buffer_api(self, frames=64):
        """
        Set the number of API-facing buffer frames.

        Parameters
        ----------
        frames : int
            Number of buffer frames to allocate.
        """
        self.logger.info("Setting API buffer frame count to %d...", frames)
        frame_current = c_ulong(0)
        err1 = self.xeneth.XC_GetPropertyValueL(
            self.cam, b"_API_FPC_BFRNUM", byref(frame_current)
        )
        self.logger.info("Previous API buffer size: %d frames", frame_current.value)
        err2 = self.xeneth.XC_SetPropertyValueL(
            self.cam, b"_API_FPC_BFRNUM", c_long(frames), ""
        )
        err3 = self.xeneth.XC_GetPropertyValueL(
            self.cam, b"_API_FPC_BFRNUM", byref(frame_current)
        )
        self.logger.info("New API buffer size: %d frames", frame_current.value)
        if err1 or err2 or err3:
            self.logger.warning(
                "Error encountered! Error codes: %d, %d, %d", err1, err2, err3
            )

    def set_timeout_api(self, timeout_ms=10000):
        """
        Set the get_frame time allowed before issuing E_NOFRAME error.

        Warning
        ~~~~~~~~
        Implementation unfinished and untested.

        Parameters
        ----------
        timeout_ms : int
            Time in ms to wait for blocking frame capture.
        """
        self.logger.info("Setting API timeout to %d ms...", timeout_ms)
        timeout_current = c_ulong(0)
        err1 = self.xeneth.XC_GetPropertyValueL(
            self.cam, b"_API_GETFRAME_TIMEOUT", byref(timeout_current)
        )
        self.logger.info("Previous API timeout: %d ms", timeout_current.value)
        err2 = self.xeneth.XC_SetPropertyValueL(
            self.cam, b"_API_GETFRAME_TIMEOUT", c_long(timeout_ms), ""
        )
        err3 = self.xeneth.XC_GetPropertyValueL(
            self.cam, b"_API_GETFRAME_TIMEOUT", byref(timeout_current)
        )
        self.logger.info("New API timeout: %d ms", timeout_current.value)
        if err1 or err2 or err3:
            self.logger.warning(
                "Error encountered! Error codes: %d, %d, %d", err1, err2, err3
            )

    def set_temperature(self, temp_c):
        """
        Set the camera temperature setpoint.

        Parameters
        ----------
        temp_c : float
            Temperature in degrees Celsius to set on startup.
        """
        self.logger.info("Setting settle temperature to %1.2fC...", temp_c)
        temp_current = c_double(0)
        err1 = self.xeneth.XC_GetPropertyValueF(
            self.cam, b"SettleTemperature", byref(temp_current)
        )
        self.logger.info("Previous set temperature: %1.2f degC", temp_current.value - 273.15)
        err2 = self.xeneth.XC_SetPropertyValueF(
            self.cam, b"SettleTemperature", c_double(temp_c + 273.15), ""
        )
        err3 = self.xeneth.XC_GetPropertyValueF(
            self.cam, b"SettleTemperature", byref(temp_current)
        )
        self.logger.info("New set temperature: %1.2f degC", temp_current.value - 273.15)
        if err1 or err2 or err3:
            self.logger.warning(
                "Error encountered! Error codes: %d, %d, %d", err1, err2, err3
            )

    def get_temperature(self):
        """
        Get the camera temperature setpoint.

        Returns
        -------
        float
            Temperature in degrees Celsius to set on startup. -1 if the temperature
            could not be read.
        """
        temp = -1.0
        temp_current = c_double(0)
        err = self.xeneth.XC_GetPropertyValueF(
            self.cam, b"Temperature", byref(temp_current)
        )

        if err:
            self.logger.warning("Error while reading sensor temperature; code: %s", err)
        else:
            temp = temp_current.value - 273.15

        return temp

    def set_readout_orientation(self, flip_x=True, flip_y=True):
        """
        Sets direction of pixel readout from focal plane.

        Parameters
        ----------
        flip_x, flip_y : tuple(bool)
            Whether a flip will be applied in the X or Y directions.
        """
        errs = []
        flip_x = c_long(int(flip_x))
        flip_y = c_long(int(flip_y))
        errs.append(
            self.xeneth.XC_SetPropertyValueL(self.cam, b"ReadoutFlipX", flip_x, "")
        )
        errs.append(
            self.xeneth.XC_SetPropertyValueL(self.cam, b"ReadoutFlipY", flip_y, "")
        )
        errs.append(
            self.xeneth.XC_GetPropertyValueL(self.cam, b"ReadoutFlipX", byref(flip_x))
        )
        errs.append(
            self.xeneth.XC_GetPropertyValueL(self.cam, b"ReadoutFlipX", byref(flip_x))
        )
        self.logger.info(
            "Readout orientation set to (flip_x,flip_y) = (%s,%s)",
            bool(flip_x.value), bool(flip_y.value),
        )

        if any(errs):
            self.logger.warning("Errors detected in trigger setup: %s", errs)

    def enable_frametags(self, enable=False):
        """
        Adds captured frame number to first two pixels of an image.
        Disabled by default to prevent issues with autoexposure.

        Parameters
        ----------
        enable : bool
            Whether to enable or disable.
        """
        if enable:
            err = self.xeneth.XC_SetPropertyValueE(self.cam, b"FrameMarker", b"Enabled")
        else:
            err = self.xeneth.XC_SetPropertyValueE(
                self.cam, b"FrameMarker", b"Disabled"
            )
        if err:
            self.logger.warning("Error setting frame tags! Code: %s", err)
        else:
            self.logger.info("Frame tags set...")

    def setup_input_trigger(
        self,
        mode=0,
        delay=0,
        source=0,
        skip=0,
        fpt=1,
    ):
        """
        Configure capture control via triggering.

        Parameters
        ----------
        mode : int
            Trigger properties.\n
            0 means free running.\n
            1 means level.\n
            2 means rising edge.\n
            3 means falling edge.
        delay : float
            Trigger delay in microseconds.
        source : int
            Source of the input trigger.\n
            0 means trigger in.\n
            1 means software.\n
            2 means CameraLink CC1.
        skip : int
            Number of frames to skip after trigger.
        fpt : int
            Frames per trigger.
        """
        trigger_modes = {
            0: b"Free running",
            1: b"Level",
            2: b"Rising edge",
            3: b"Falling edge",
        }
        trigger_sources = {0: b"Trigger in", 1: b"Software", 2: b"CameraLink CC1"}
        errs = []

        # Get current trigger setup
        mode_old = (c_char * 128)(0)
        errs.append(
            self.xeneth.XC_GetPropertyValueE(
                self.cam, b"TriggerMode", byref(mode_old), 128
            )
        )
        delay_old = c_double(0)
        errs.append(
            self.xeneth.XC_GetPropertyValueF(
                self.cam, b"TriggerInputDelay", byref(delay_old)
            )
        )
        source_old = (c_char * 128)(0)
        errs.append(
            self.xeneth.XC_GetPropertyValueE(
                self.cam, b"TriggerSource", byref(source_old), 128
            )
        )
        skip_old = c_long(0)
        errs.append(
            self.xeneth.XC_GetPropertyValueL(self.cam, b"TriggerSkip", byref(skip_old))
        )
        fpt_old = c_long(0)
        errs.append(
            self.xeneth.XC_GetPropertyValueL(self.cam, b"NrOfFrames", byref(fpt_old))
        )
        self.logger.debug(
            "Original trigger setup: Mode - %s | Delay - %1.2fus "
            "| Source - %s | NSkip - %d | FPT - %d",
            mode_old.value.decode(),
            delay_old.value,
            source_old.value.decode(),
            skip_old.value,
            fpt_old.value,
        )

        # Set desired trigger setup
        errs.append(
            self.xeneth.XC_SetPropertyValueL(self.cam, b"NrOfFrames", c_long(fpt), "")
        )
        errs.append(
            self.xeneth.XC_SetPropertyValueE(
                self.cam, b"TriggerMode", trigger_modes[mode]
            )
        )
        errs.append(
            self.xeneth.XC_SetPropertyValueF(
                self.cam, b"TriggerInputDelay", c_double(delay), ""
            )
        )
        errs.append(
            self.xeneth.XC_SetPropertyValueE(
                self.cam, b"TriggerSource", trigger_sources[source]
            )
        )
        errs.append(
            self.xeneth.XC_SetPropertyValueL(self.cam, b"TriggerSkip", c_long(skip), "")
        )

        # Get final trigger setup
        errs.append(
            self.xeneth.XC_GetPropertyValueE(
                self.cam, b"TriggerMode", byref(mode_old), 128
            )
        )
        errs.append(
            self.xeneth.XC_GetPropertyValueF(
                self.cam, b"TriggerInputDelay", byref(delay_old)
            )
        )
        errs.append(
            self.xeneth.XC_GetPropertyValueE(
                self.cam, b"TriggerSource", byref(source_old), 128
            )
        )
        errs.append(
            self.xeneth.XC_GetPropertyValueL(self.cam, b"TriggerSkip", byref(skip_old))
        )
        errs.append(
            self.xeneth.XC_GetPropertyValueL(self.cam, b"NrOfFrames", byref(fpt_old))
        )
        self.logger.debug(
            "New trigger setup: Mode - %s | Delay - %1.2fus "
            "| Source - %s | NSkip - %d | FPT - %d",
            mode_old.value.decode(),
            delay_old.value,
            source_old.value.decode(),
            skip_old.value,
            fpt_old.value,
        )

        if any(errs):
            self.logger.warning("Errors detected in trigger setup: %s", errs)

    def setup_output_trigger(
        self,
        enable=1,
        mode=1,
        source=2,
        delay=0,
        width=10,
    ):
        """
        Configures output trigger.

        Parameters
        ----------
        enable : bool
            Whether to enable the output trigger.
        mode : int
            Trigger properties.\n
            0 means active low.\n
            1 means active high.
        source : int
            Source of the input trigger.\n
            0 means integration start.\n
            1 means trigger input.\n
            2 means integration period.
        delay : float
            Trigger delay in microseconds.
        width : float
            Number of frames to skip after trigger.
        """

        trigger_status = {0: b"Off", 1: b"On"}
        trigger_modes = {0: b"Active low", 1: b"Active high"}
        trigger_sources = {
            0: b"Integration start",
            1: b"Trigger input",
            2: b"Integration period",
        }
        errs = []

        # Get current trigger setup
        status_old = (c_char * 128)(0)
        errs.append(
            self.xeneth.XC_GetPropertyValueE(
                self.cam, b"TriggerOutEnable", byref(status_old), 128
            )
        )
        mode_old = (c_char * 128)(0)
        errs.append(
            self.xeneth.XC_GetPropertyValueE(
                self.cam, b"TriggerOutMode", byref(mode_old), 128
            )
        )
        delay_old = c_double(0)
        errs.append(
            self.xeneth.XC_GetPropertyValueF(
                self.cam, b"TriggerOutDelay", byref(delay_old)
            )
        )
        source_old = (c_char * 128)(0)
        errs.append(
            self.xeneth.XC_GetPropertyValueE(
                self.cam, b"TriggerOutSource", byref(source_old), 128
            )
        )
        width_old = c_long(0)
        errs.append(
            self.xeneth.XC_GetPropertyValueL(
                self.cam, b"TriggerOutWidth", byref(width_old)
            )
        )
        self.logger.debug(
            "Original trigger setup: Status - %s | Mode - %s | Delay - %1.2fus "
            "| Source - %s | Width - %d us",
            status_old.value.decode(),
            mode_old.value.decode(),
            delay_old.value,
            source_old.value.decode(),
            width_old.value,
        )

        # Set desired trigger setup
        errs.append(
            self.xeneth.XC_SetPropertyValueE(
                self.cam, b"TriggerOutEnable", trigger_status[enable]
            )
        )
        errs.append(
            self.xeneth.XC_SetPropertyValueE(
                self.cam, b"TriggerOutMode", trigger_modes[mode]
            )
        )
        errs.append(
            self.xeneth.XC_SetPropertyValueF(
                self.cam, b"TriggerOutDelay", c_double(delay), ""
            )
        )
        errs.append(
            self.xeneth.XC_SetPropertyValueE(
                self.cam, b"TriggerOutSource", trigger_sources[source]
            )
        )
        errs.append(
            self.xeneth.XC_SetPropertyValueF(
                self.cam, b"TriggerOutWidth", c_double(width), ""
            )
        )

        # Get final trigger setup
        errs.append(
            self.xeneth.XC_GetPropertyValueE(
                self.cam, b"TriggerOutEnable", byref(status_old), 128
            )
        )
        errs.append(
            self.xeneth.XC_GetPropertyValueE(
                self.cam, b"TriggerOutMode", byref(mode_old), 128
            )
        )
        errs.append(
            self.xeneth.XC_GetPropertyValueF(
                self.cam, b"TriggerOutDelay", byref(delay_old)
            )
        )
        errs.append(
            self.xeneth.XC_GetPropertyValueE(
                self.cam, b"TriggerOutSource", byref(source_old), 128
            )
        )
        errs.append(
            self.xeneth.XC_GetPropertyValueL(
                self.cam, b"TriggerOutWidth", byref(width_old)
            )
        )
        self.logger.debug(
            "New trigger setup: Status - %s | Mode - %s | Delay - %1.2fus "
            "| Source - %s | Width - %d us",
            status_old.value.decode(),
            mode_old.value.decode(),
            delay_old.value,
            source_old.value.decode(),
            width_old.value,
        )

        if any(errs):
            self.logger.warning("Errors detected in trigger setup: %s", errs)

    def setup_grabber(self, mode=0, frames=4000):
        """
        Setup frame grabber capture mode.

        Parameters
        ----------
        mode : int
            Capture mode.\n
            0 means circular buffer.\n
            1 means waits till all frames grabbed before restarting capture.\n
            2 means non-circular (Stops grabbing when buffer filled).
        frames : int
            Number of frames to capture.
        """
        errs = []
        modes = {
            0: b"Preview",  # Circular buffer
            1: b"Synchronous bursts",  # Waits till all frames grabbed before restarting capture
            2: b"Synchronous burst",  # Non-circular (Stops grabbing when buffer filled)
        }
        mode_old = (c_char * 128)(0)
        errs.append(
            self.xeneth.XC_GetPropertyValueE(self.cam, b"Mode", byref(mode_old), 128)
        )
        self.logger.info("Previous capture mode: %s", mode_old.value.decode())
        errs.append(self.xeneth.XC_SetPropertyValueE(self.cam, b"Mode", modes[mode]))
        errs.append(
            self.xeneth.XC_GetPropertyValueE(self.cam, b"Mode", byref(mode_old), 128)
        )
        self.logger.info("New capture mode: %s", mode_old.value.decode())

        # Set the buffer size
        errs.append(
            self.xeneth.XC_SetPropertyValueL(
                self.cam, b"FrameCount", c_long(frames), ""
            )
        )
        frames = c_long(0)
        errs.append(
            self.xeneth.XC_GetPropertyValueL(
                self.cam, b"FrameCount", byref(frames), 128
            )
        )
        self.logger.info("Buffer frame count set to %d frames", frames.value)

        if any(errs):
            self.logger.warning("Error(s) encountered: %s", errs)

    def set_woi(self, woi=None):
        """See :meth:`.Camera.set_woi`"""
        if woi is None:
            woi = (0, self.default_shape[1], 0, self.default_shape[0])

        woi = (woi[0], woi[1] - woi[0], woi[2], woi[3] - woi[2])

        # If collecting, stop
        if self.isCapturing():
            self.stopCapture()

        # Get current WOI setup
        report_str = "Original WOI setup: "
        woi_prop = [b"WoiSX(0)", b"WoiEX(0)", b"WoiSY(0)", b"WoiEY(0)"]
        prop_val = c_long(0)
        errs = []
        for prop in woi_prop:
            errs.append(
                self.xeneth.XC_GetPropertyValueL(self.cam, prop, byref(prop_val))
            )
            report_str += "%s: %d | " % (prop.decode(), prop_val.value)
        self.logger.debug("%s", report_str)

        # Conservatively round inputs (make wider than requested) based on cam. reqts.
        min_w_factor = 16
        min_h_factor = 4
        if (woi[0]) % min_w_factor:
            woi[0] = max([woi[0] - woi[0] % min_w_factor, 0])
        if (woi[1] - woi[0] + 1) % min_w_factor:
            woi[1] = (
                woi[1] + min_w_factor - (woi[1] - woi[0]) % min_w_factor
            )
        if (woi[2]) % min_h_factor:
            woi[2] = max([woi[2] - woi[2] % min_h_factor, 0])
        if (woi[3] - woi[2] + 1) % min_h_factor:
            woi[3] = (
                woi[3] + min_h_factor - (woi[3] - woi[2]) % min_h_factor
            )

        # Set new WOI
        for i, prop in enumerate(woi_prop):
            errs.append(
                self.xeneth.XC_SetPropertyValueL(self.cam, prop, c_long(woi[i]), "")
            )

        # Report new WOI
        report_str = "     New WOI setup: "
        for i, prop in enumerate(woi_prop):
            errs.append(
                self.xeneth.XC_GetPropertyValueL(self.cam, prop, byref(prop_val))
            )
            report_str += "%s: %d | " % (prop.decode(), prop_val.value)
            woi[i] = prop_val.value
        self.logger.debug("%s", report_str)

        # Reconfigure buffer based on new WOI
        self.shape = (woi[3] - woi[2] + 1, woi[1] - woi[0] + 1)
        self.woi = (woi[0], self.shape[0], woi[2], self.shape[1])
        self.frame_size = self.xeneth.XC_Getframe_size(self.cam)
        self.frame_buffer = (c_ushort * int(self.frame_size / 2))(0)
        self.last_capture = np.empty(self.shape)

        if any(errs):
            self.logger.warning("Error(s) encountered: %s", errs)

        return self.woi

    def set_low_gain(self, enable=True):
        """
        Enables or disables low gain mode.

        Parameters
        ----------
        enable : bool
            Whether to enable or disable.
        """
        gain_current = c_long(0)
        self.xeneth.XC_GetPropertyValueL(self.cam, b"LowGain", byref(gain_current))
        gain_current = bool(gain_current.value)
        if enable and not gain_current:
            self.logger.info("Enabling low gain mode...")
            self.xeneth.XC_SetPropertyValueL(self.cam, b"LowGain", c_long(1), "")
        elif (not enable) and gain_current:
            self.logger.info("Disabling low gain mode...")
            self.xeneth.XC_SetPropertyValueL(self.cam, b"LowGain", c_long(0), "")

    def enable_cooling(self, enable=True):
        """
        Enables/disables TEC and high fan speed.

        Parameters
        ----------
        enable : bool
            Whether to enable or disable.
        """
        fan = c_long(0)
        self.xeneth.XC_GetPropertyValueL(self.cam, b"Fan", byref(fan))
        fan_current = bool(fan.value)
        if enable and not fan_current:
            self.logger.info("Enabling cooling/high fan speed...")
            self.xeneth.XC_SetPropertyValueL(self.cam, b"Fan", c_long(1), "")
        elif (not enable) and fan_current:
            self.logger.info("Disabling cooling/high fan speed...")
            self.xeneth.XC_SetPropertyValueL(self.cam, b"Fan", c_long(0), "")

    ### Image Grabbing ###

    def setup(self, profile, fpt=1):
        """
        Sample pre-configured imaging profiles.

        Likely to be reconfigured by end-user for various imaging tasks.

        Parameters
        ----------
        profile
            See :attr:`profile`.
        fpt : int
            Frames per trigger for intput trigger.
        """
        if profile == "triggered":
            # 0.1 ms exposure (allows 1kHz imaging w/ 350x350 WOI w/ 400Hz Cheetah 640)
            self.set_exposure(100e-6)
            # Rising edge trigger on camera; 1 frame each trigger
            self.setup_input_trigger(mode=2, source=0, fpt=1)
            # Need 2x avoid pre-trigger
            self.setup_input_trigger(mode=2, source=0, fpt=fpt)
            # Enable low gain
            self.set_low_gain(False)
        elif profile == "free":
            # Free running, software trigger
            self.setup_input_trigger()
            # 30 ms fixed exposure
            self.set_exposure(7e-3)
            # Start free-running capture
            self.start_capture()
        else:
            self.logger.warning("Profile '%s' not found! Returning...", profile)

    def snap(self, conversion=False):
        """
        Start capture, grab image, stop capture.

        Parameters
        ----------
        conversion : bool
            Makes an internal 8 bit buffer for :meth:`xeneth.SaveData` and :meth:`xeneth.Blit`
        """
        self.logger.info("Starting capture...")
        err = self.xeneth.XC_StartCapture(self.cam)
        if err != I_OK:
            self.logger.warning("Could not start capturing, errorCode: %d", err)
        elif self.xeneth.XC_IsCapturing(self.cam):
            self.logger.info("Grabbing a frame...")
            if conversion:
                err = self.xeneth.XC_GetFrame(
                    self.cam,
                    FT_NATIVE,
                    XGF_BLOCKING,
                    self.frame_buffer,
                    self.frame_size,
                )
            else:  # Max performance by skipping 8 bit buffer gen
                err = self.xeneth.XC_GetFrame(
                    self.cam,
                    FT_NATIVE,
                    XGF_BLOCKING | XGF_NOCONVERSION,
                    self.frame_buffer,
                    self.frame_size,
                )
            if err != I_OK:
                self.logger.warning("Problem while fetching frame, errorCode %d", err)
            else:
                im = np.frombuffer(self.frame_buffer, c_ushort).reshape(self.shape)
                self.logger.info("Stopping capture...")
                err = self.xeneth.XC_StopCapture(self.cam)
                if err != I_OK:
                    self.logger.warning("Could not stop capturing, errorCode: %d", err)
                else:
                    return im
        return -1

    def _get_image_hw(self, timeout_s=None, frame_type=FT_NATIVE, block=True, convert=True):
        """
        Main grabbing function; captures latest image into single frame buffer.

        Warning
        ~~~~~~~~
        ``timeout_s`` parameter is currently untested; setting it may lead to unintended behavior.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait for frames to catch up with triggers.
        frame_type : FT_NATIVE, 0
            Sets type of frame conversion.
        block : bool
            Blocking read; waits up to ``timeout_s`` for frame.
        convert : bool
            Makes internal 8 bit buffer, set false for max performance.

        Returns
        -------
        int, numpy.ndarray
            Error code in the event of an error, otherwise the current frame.
        """

        # Update the timeout time (in ms) if different than API default
        # Note: To be renovated to only set timeout if different than current camera value...
        if block:
            self.set_timeout_api(int(1000 * timeout_s))

        # Set flag based on input options
        flag = XGF_BLOCKING if block else 0
        if not convert:
            flag = flag | XGF_NOCONVERSION

        ret = err = self.xeneth.XC_GetFrame(
            self.cam, frame_type, flag, self.frame_buffer, self.frame_size
        )
        if err == I_OK:
            t = time.perf_counter()
            self.last_capture = np.frombuffer(self.frame_buffer, c_ushort)
            self.last_tag = np.uint16(
                int(
                    format(self.last_capture[1], "012b")[:8]
                    + format(self.last_capture[0], "012b")[:8],
                    2,
                )
            )
            # Delete frame tag to avoid issues w/ autfocus/exposure.
            self.last_capture[:2] = 0
            self.last_capture = self.last_capture.reshape(self.shape)
            self.last_process_time = time.perf_counter() - t
            ret = self.last_capture

        return ret

    def get_frame_number(self):
        """
        Get number of captured frames since :meth:`start_capture()`.

        Returns
        ----------
        int
            Number of frames.
        """
        return self.xeneth.XC_GetFrameCount(self.cam)

    def start_capture(self):
        """Initiates the current capture run"""
        self.logger.info("Starting capture...")
        err = self.xeneth.XC_StartCapture(self.cam)
        if err != I_OK:
            self.logger.warning("Could not start capturing, errorCode: %d", err)
        while not self.isCapturing():
            self.logger.debug("Waiting for capture start...")
            time.sleep(0.1)

    def stop_capture(self):
        """Terminates the current capture run"""
        self.logger.info("Stopping capture...")
        err = self.xeneth.XC_StopCapture(self.cam)
        if err != I_OK:
            self.logger.warning("Could not stop capturing, errorCode: %d", err)

    def abort_capture(self):
        """Cancels any long, live frame captures"""
        self.logger.info("Aborting capture...")
        err = self.xeneth.XC_SetPropertyValueE(self.cam, b"AbortExposure", b"Abort")
        if err != I_OK:
            self.logger.warning("Could not abort capture, errorCode: %d", err)

    def flush(self, timeout_s=1):
        """See :meth:`.Camera.flush`"""
        time.sleep(0.1)  # Allow some time to grab free-running images
        err = self._get_image_hw(block=False, convert=False, return_img=False)
        while not err:
            err = self._get_image_hw(block=False, convert=False, return_img=False)

    def is_capturing(self):
        """
        Checks if currently capturing

        Returns
        -------
        bool
            ``True`` if capturing, ``False`` otherwise.
        """
        return self.xeneth.XC_IsCapturing(self.cam)

    ### Filters ###

    def autogain(self, enable=True):
        """
        Adds autogain and offset to current filter stack. Makes use of full dynamic range.

        Parameters
        ----------
        enable : bool
            Enables autogain if True.
        """
        if enable and "autogain" not in self.filters.keys():
            self.logger.info("Enabling autogain...")
            tag = self.xeneth.XC_FLT_Queue(self.cam, b"AutoOffsetAndGain", "")
            self.filters["autogain"] = tag
        elif (not enable) and "autogain" in self.filters.keys():
            self.logger.info("Disabling autogain...")
            self.xeneth.XC_RemImageFilter(self.cam, self.filters["autogain"])
            self.filters.pop("autogain")

    def autoexpose_xenics(self, enable=True, t_settle=0):
        """
        Adds Xenics autogain and offset filters to current filter stack.

        Makes use of the camera's full dynamic range.

        Parameters
        ----------
        enable : bool
            Enables autogain if True.
        t_settle : float
            Time to allow autoexposure to settle.
        """
        if enable and "autoexposure" not in self.filters.keys():
            self.logger.info("Enabling autoexposure...")
            tag = self.xeneth.XC_FLT_Queue(self.cam, b"AutoExposure", "")
            self.xeneth.XC_FLT_SetParameter(self.cam, tag, b"Target", b"50")
            self.logger.debug(
                "Outliers parameter set: %s",
                self.xeneth.XC_FLT_SetParameter(self.cam, tag, b"Outliers", b"0.0"),
            )
            self.filters["autoexposure"] = tag
            t_start = time.perf_counter()
            while time.perf_counter() - t_start < t_settle:
                self._get_image_hw()

        elif (not enable) and "autoexposure" in self.filters.keys():
            self.logger.info("Disabling autoexposure...")
            self.xeneth.XC_RemImageFilter(self.cam, self.filters["autoexposure"])
            self.filters.pop("autoexposure")

    def close_filters(self):
        """Deletes all current *tracked* filters in the stack."""
        errs = []
        for filter_key in self.filters:
            self.logger.info("Closing %s filter...", filter_key)
            errs.append(
                self.xeneth.XC_RemImageFilter(self.cam, self.filters[filter_key])
            )
        self.filters = {}
        if any(errs):
            self.logger.warning("Errors when closing filters! Codes: %s", errs)
