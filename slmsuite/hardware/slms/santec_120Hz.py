"""
Hardware control for Santec SLMs.
Tested with Santec LCoS SLM-200, SLM-210, and SLM-300.

Note
~~~~
:class:`.Santec_120` requires dynamically linked libraries from Santec to be present in the
runtime directory:

- SLMFunc.dll
- FTD3XX.dll

These files should be copied in before use.

Note
~~~~
Santec provides base wavefront correction accounting for the curvature of the SLM surface.
Consider loading these files via :meth:`.SLM.load_vendor_phase_correction()`
"""
import os
import ctypes
import numpy as np
import cv2
import warnings

from .slm import SLM

try:  # Load Santec's header file.
    from . import _slm_win as slm_funcs
except BaseException as e:  # Provide an informative error should something go wrong.
    warnings.warn(
        "Santec DLLs not installed. Install these to use Santec SLMs."
        "  Dynamically linked libraries from Santec (usually provided via USB) "
        "must be present in the runtime directory:\n"
        "  - SLMFunc.dll\n  - FTD3XX.dll\n"
        "  Check that these files are present and are error-free.\n"
        "Original error: {}".format(e)
    )
    slm_funcs = None


class Santec_120(SLM):
    """
    Interfaces with Santec SLMs (120Hz version).

    Attributes
    ----------
    slm_number : int
        USB port number assigned by Santec SDK.
    display_number : int
        Display number assigned by Santec SDK.
    optionboard_id : str
        ID of the option board
    driveboard_id : str
        ID of the drive board
    product_code_id : str
        Product code of the device
    """

    def __init__(
            self,
            slm_number=1,
            display_number=2,
            bitdepth=10,
            wav_um=1,
            pitch_um=(8,8),
            verbose=True,
            **kwargs
        ):
        r"""
        Initializes an instance of a Santec SLM (120Hz version).

        Arguments
        ------
        slm_number
            See :attr:`slm_number`.
        display_number
            See :attr:`display_number`.
        bitdepth : int
            Depth of SLM pixel well in bits. Defaults to 10.
        wav_um : float
            Wavelength of operation in microns. Defaults to 1 um.
        pitch_um : (float, float)
            Pixel pitch in microns. Defaults to 8 micron square pixels.
        verbose : bool
            Whether to print extra information.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.

        Note
        ----
        Santec SLMs can reconfigure their phase table: the correspondence between
        grayscale values and applied voltages. This is configured based upon the wavelength
        supplied to :attr:`.SLM.wav_design_um`. This allows :attr:`.SLM.phase_scaling`
        to be one if desired, and make use of optimized routines (see :meth`.set_phase()`).
        However, sometimes setting the phase table runs into issues, where the maximum value
        doesn't correspond to exactly :math:`2\pi` at the target wavelength. This is noted
        in the initialization, and the user should update :attr:`.SLM.wav_design_um` or otherwise
        to avoid undesired behavior.
        :attr:`wav_design_um` defaults to :attr:`wav_um` if it is not given.

        Caution
        ~~~~~~~
        :class:`.Santec_120` defaults to 8 micron square pixel size :attr:`.SLM.pitch_um`
        and 10-bit :attr:`.SLM.bitdepth`.
        This is valid for SLM-200, SLM-210, and SLM-300,
        but may not be valid for future Santec models.
        """
        if slm_funcs is None:
            raise RuntimeError(
                "Santec DLLs not installed. Install these to use Santec SLMs."
                "  Dynamically linked libraries from Santec (usually provided via USB) "
                "must be present in the runtime directory:\n"
                "  - SLMFunc.dll\n  - FTD3XX.dll\n"
                "  Check that these files are present and are error-free."
            )

        # Default max phase. Maybe this should be opened to the user in the future.
        # Otherwise, wav_um and wav_design_um have the same functionality.
        max_phase = 2 * np.pi

        self.slm_number = int(slm_number)
        self.display_number = int(display_number)

        # By default, target wavelength is the design wavelength
        wav_design_um = kwargs.pop("wav_design_um", None)
        if wav_design_um is None:
            wav_design_um = wav_um

        if verbose:
            print("Santec_120 slm_number={} initializing... ".format(self.slm_number), end="")
        Santec_120._parse_status(slm_funcs.SLM_Ctrl_Open(self.slm_number))

        try:
            # Wait for the SLM to no longer be busy.
            while True:
                status = slm_funcs.SLM_Ctrl_ReadSU(self.slm_number)

                if status == 0:
                    break  # SLM_OK (proceed)
                elif status == 2:
                    continue  # SLM_BS (busy)
                else:
                    Santec_120._parse_status(status)

            # Check to see if the device or option boards have an error.
            self.get_error(raise_error=True)

            # Right now, only DVI mode is supported.
            Santec_120._parse_status(slm_funcs.SLM_Ctrl_WriteVI(self.slm_number, 1))  # 0:Memory 1:DVI
            if verbose:
                print("success")

            # Update wavelength if needed
            wav_current_nm = ctypes.c_uint32(0)
            phase_current = ctypes.c_ulong(0)
            Santec_120._parse_status(
                slm_funcs.SLM_Ctrl_ReadWL(self.slm_number, wav_current_nm, phase_current)
            )

            wav_desired_nm = int(1e3 * wav_design_um)
            phase_desired = int(np.floor(max_phase * 100 / np.pi))

            # Update the phase table if necessary. This sometimes fails for unknown
            # reasons, so we do multiple attempts. Reasons for failure might include overheating
            # while the energy-intensive process of updating the table is underway.
            attempt = 1
            while wav_current_nm.value != wav_desired_nm and attempt < 5:
                if verbose:
                    if attempt > 1:
                        print("(attempt {})".format(attempt))
                    else:
                        print(
                            "Current phase table: wav = {0} nm, maxphase = {1:.2f}pi".format(
                                wav_current_nm.value, phase_current.value / 100.0
                            )
                        )
                        print(
                            "Desired phase table: wav = {0} nm, maxphase = {1:.2f}pi".format(
                                wav_desired_nm, phase_desired / 100.0
                            )
                        )
                    print("     ...Updating phase table (this may take 40 seconds)...")

                # Set wavelength (nm) and maximum phase (100 * [float pi])
                Santec_120._parse_status(
                    slm_funcs.SLM_Ctrl_WriteWL(
                        self.slm_number, ctypes.c_uint32(wav_desired_nm), phase_desired
                    )
                )

                # Save wavelength
                Santec_120._parse_status(slm_funcs.SLM_Ctrl_WriteAW(self.slm_number))

                # Verify wavelength
                Santec_120._parse_status(
                    slm_funcs.SLM_Ctrl_ReadWL(self.slm_number, wav_current_nm, phase_current)
                )
                if verbose:
                    print(
                        "Updated phase table: wav = {0} nm, maxphase = {1:.2f}pi".format(
                            wav_current_nm.value, phase_current.value / 100.0
                        )
                    )

                attempt += 1

            # Raise an error if we failed.
            if wav_current_nm.value != wav_desired_nm or abs(phase_current.value - 200) > 100:
                raise RuntimeError("Failed to update Santec phase table.")

            # Note phase table issues if they are present
            if verbose and abs(phase_current.value - 200) > 4:
                wav_design_fixed_um = wav_design_um * (phase_current.value / 200.0)
                print(
                    "  Warning: the Santec phase table maximum deviates significantly (>2%) from 2pi ({0:.2f}pi).".format(
                        phase_current.value / 100.0
                    )
                )
                print(
                    "    This is likely due to internal checks avoiding 'abnormal' phase table results."
                )
                print(
                    "    To compensate for this, wav_design_um is noted to equal {} instead of the desired {}.".format(
                        wav_design_fixed_um, wav_design_um
                    )
                )
                if wav_um / wav_design_fixed_um != 1:
                    print(
                        "    This results in phase_scaling={0:.4f} != 1, which has negative speed implications (see .set_phase()).".format(
                            wav_um / wav_design_fixed_um
                        )
                    )
                print(
                    "    If this behavior is undesired, play with wav_design_um to find a better regime."
                )
                wav_design_um = wav_design_fixed_um

            # Check for the SLM parameters and save them
            width = ctypes.c_ushort(0)
            height = ctypes.c_ushort(0)
            display_name = ctypes.create_string_buffer(128)

            if verbose:
                print("Looking for display_number={}... ".format(self.display_number), end="")
            Santec_120._parse_status(
                slm_funcs.SLM_Disp_Info2(self.display_number, width, height, display_name)
            )

            # For instance, "LCOS-SLM,SOC,8001,2018021001"
            # Format is "UserFriendlyName,ManufacterName,ProductCodeID,SerialNumberID"
            name = display_name.value.decode("mbcs")
            if verbose:
                print("success")

            names = name.split(",")

            # If the target display_number is not found, then print some debug:
            if names[0] != "LCOS-SLM":
                # Don't parse status around this one...
                raise ValueError(
                    "SLM not found at display_number={}. "
                    "Use .info() to find the correct display!".format(self.display_number)
                )

            # Populate some info
            driveboard_id = ctypes.create_string_buffer(16)
            optionboard_id = ctypes.create_string_buffer(16)
            Santec_120._parse_status(
                slm_funcs.SLM_Ctrl_ReadSDO(self.slm_number, driveboard_id, optionboard_id)
            )

            self.driveboard_id = driveboard_id.value.decode("mbcs")
            self.optionboard_id = optionboard_id.value.decode("mbcs")
            self.product_code_id = names[2]

            # Open SLM
            if verbose:
                print("Opening {}... ".format(name), end="")
            Santec_120._parse_status(slm_funcs.SLM_Disp_Open(self.display_number))
            if verbose:
                print("success")

            super().__init__(
                (int(width.value), int(height.value)),
                bitdepth=bitdepth,
                name=kwargs.pop("name", names[-1]), # SerialNumberID
                wav_um=wav_um,
                wav_design_um=wav_design_um,
                pitch_um=pitch_um,
                **kwargs,
            )

            self.set_phase(None)
        except Exception as init_error:
            try:
                Santec_120._parse_status(slm_funcs.SLM_Ctrl_Close(self.slm_number))
            except Exception as close_error:
                print(
                    "Could not close attempt to open Santec_120 slm_number={}: {}".format(
                        slm_number, str(close_error)
                    )
                )

            raise init_error

    @staticmethod
    def info(verbose=True):
        """
        Discovers the names of all the displays.
        Checks all 8 possible supported by Santec's SDK.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of (int, str) tuples
            The number and name of each potential display.
        """
        if slm_funcs is None:
            raise RuntimeError(
                "Santec DLLs not installed. Install these to use Santec SLMs."
                "  Dynamically linked libraries from Santec (usually provided via USB) "
                "must be present in the runtime directory:\n"
                "  - SLMFunc.dll\n  - FTD3XX.dll\n"
                "  Check that these files are present and are error-free."
            )

        # Check for the SLM parameters and save them
        display_list = []

        if verbose:
            print("Displays detected by Santec_120")
            print("display_number, display_name:")

        for display_number in range(1, 9):
            width = ctypes.c_ushort(0)
            height = ctypes.c_ushort(0)
            display_name = ctypes.create_string_buffer(128)

            status = slm_funcs.SLM_Disp_Info2(display_number, width, height, display_name)

            if not (status == 0 or status == -1):
                Santec_120._parse_status(status)

            name = display_name.value.decode("mbcs")
            if len(name) > 0:
                if verbose:
                    print("{},  {}".format(display_number, name))

                display_list.append((display_number, name))

        return display_list

    def load_vendor_phase_correction(self, file_path, smooth=False, overwrite=True):
        """
        Load phase correction provided by Santec from file,
        setting ``"phase"`` in :attr:`~slmsuite.hardware.slms.slm.SLM.source`.

        Parameters
        ----------
        file_path : str
            File path for the vendor-provided phase correction.
        smooth : bool
            Whether to apply a Gaussian blur to smooth the data.
        overwrite : bool
            Whether to overwrite the previous phase in
            :attr:`~slmsuite.hardware.slms.slm.SLM.source`.

        Note
        ~~~~
        This correction is only fully valid at the wavelength at which it was collected.

        Returns
        ----------
        numpy.ndarray
            :attr:`~slmsuite.hardware.slms.slm.SLM.source` ``["phase"]``,
            the Santec-provided phase correction.
        """
        try:
            # Load from .csv, skipping the first row and column
            # (corresponding to X and Y coordinates).
            map = np.loadtxt(file_path, skiprows=1, dtype=int, delimiter=",")[:, 1:]
            phase = (-2 * np.pi / self.bitresolution) * map.astype(float)

            # Smooth the map
            if smooth:
                real = np.cos(phase)
                imag = np.sin(phase)
                size_blur = 15  # The user should have access to this eventually

                real = cv2.GaussianBlur(real, (size_blur, size_blur), 0)
                imag = cv2.GaussianBlur(imag, (size_blur, size_blur), 0)

                # Recombine the components
                phase = np.arctan2(imag, real) + np.pi

            if overwrite:
                self.source["phase"] = phase

            return phase
        except BaseException as e:
            warnings.warn("Error while loading phase correction.\n{}".format(e))
            return self.source["phase"]

    def close(self):
        """See :meth:`.SLM.close`."""
        slm_funcs.SLM_Disp_Close(self.display_number)
        slm_funcs.SLM_Ctrl_Close(self.slm_number)

    def _set_phase_hw(self, phase):
        """See :meth:`.SLM._set_phase_hw`."""
        matrix = phase.astype(slm_funcs.USHORT)
        n_h, n_w = self.shape

        # Write to SLM
        c = matrix.ctypes.data_as(ctypes.POINTER((slm_funcs.USHORT * n_h) * n_w)).contents
        Santec_120._parse_status(
            slm_funcs.SLM_Disp_Data(self.display_number, n_w, n_h, 0x20000000, c), raise_error=False
        )

    ### Additional Santec-specific functionality

    def get_temperature(self):
        """
        Read the drive board and option board temperatures.

        Returns
        -------
        (float, float)
            Temperature in Celsius of the drive and option board
        """
        # Note that the Santec documentation suggests using signed
        # integers, but the header requests unsigned integers.
        drive_temp = ctypes.c_uint32(0)
        option_temp = ctypes.c_uint32(0)

        Santec_120._parse_status(slm_funcs.SLM_Ctrl_ReadT(self.slm_number, drive_temp, option_temp))

        return (drive_temp.value / 10.0, option_temp.value / 10.0)

    def get_error(self, raise_error=True, return_codes=False):
        """
        Read the drive board and option board errors.

        Parameters
        ----------
        raise_error : bool
            Whether to raise an error (if True) or a warning (if False) if error(s) are detected.
        return_codes : bool
            Whether to return an error string or integer error codes
            (in ``(drive_error, option_error)`` form).

        Returns
        -------
        list of str OR (int, int)
            List of errors.
        """
        drive_error = ctypes.c_uint32(0)
        option_error = ctypes.c_uint32(0)

        Santec_120._parse_status(
            slm_funcs.SLM_Ctrl_ReadEDO(self.slm_number, drive_error, option_error),
            raise_error=raise_error,
        )

        # Check the resulting bitstrings for errors (0 ==> all good).
        errors = []

        for drive_error_bit in slm_funcs.SLM_DRIVEBOARD_ERROR.keys():
            if drive_error.value & drive_error_bit:
                errors.append(slm_funcs.SLM_DRIVEBOARD_ERROR[drive_error_bit])

        for option_error_bit in slm_funcs.SLM_OPTIONBOARD_ERROR.keys():
            if option_error.value & option_error_bit:
                errors.append(slm_funcs.SLM_OPTIONBOARD_ERROR[option_error_bit])

        if raise_error and len(errors) > 0:
            error = "Santec_120 error: " + ", ".join(["'" + err + "'" for err in errors])
            if raise_error:
                raise RuntimeError(error)
            else:
                warnings.warn(error)

        if return_codes:
            return (drive_error.value, option_error.value)
        else:
            return errors

    def get_status(self, raise_error=True):
        """
        Gets ``SLM_STATUS`` return from a Santec SLM and parses the result.

        Parameters
        ----------
        raise_error : bool
            Whether to raise an error (if True) or a warning (if False) when status is
            not ``SLM_OK``.

        Returns
        -------
        (int, str, str)
            Status in ``(num, name, note)`` form.
        """
        return Santec_120._parse_status(slm_funcs.SLM_Ctrl_ReadSU(self.slm_number), raise_error)

    @staticmethod
    def _parse_status(status, raise_error=True):
        """
        Parses the meaning of a ``SLM_STATUS`` return from a Santec SLM.

        Parameters
        ----------
        status : int
            ``SLM_STATUS`` return.
        raise_error : bool
            Whether to raise an error (if True) or a warning (if False) when status is not ``SLM_OK``.

        Returns
        -------
        (int, str, str)
            Status in ``(name, note)`` form.
        """
        # Parse status
        status = int(status)

        if not status in slm_funcs.SLM_STATUS_DICT.keys():
            raise ValueError("SLM status '{}' not recognized.".format(status))

        # Recover the meaning of status
        (name, note) = slm_funcs.SLM_STATUS_DICT[status]

        status_str = "Santec_120 error {}; '{}'".format(name, note)

        if status != 0:
            if raise_error:
                raise RuntimeError(status_str)
            else:
                warnings.warn(status_str)

        return (status, name, note)

    def load_csv(self, filename):
        """
        Write the phase image contained in a .csv file to the SLM.
        This image should have the size of the SLM.

        Parameters
        ----------
        filename : str
            Path to the .csv file.
        """
        Santec_120._parse_status(slm_funcs.SLM_Disp_ReadCSV(self.display_number, 0x20000000, filename))#wf250708

    ### Low priority trigger and memory features.
    ### Feel free to finish, or poke regarding implementation.

    # def software_trigger(self):
    #     slm_funcs.SLM_Ctrl_WriteTS(self.slm_number)

    # def set_intput_trigger(self, enabled=True):
    #     slm_funcs.SLM_Ctrl_WriteTI(self.slm_number, int(enabled))

    # def set_intput_trigger_direction(self, ascending=True):
    #     slm_funcs.SLM_Ctrl_WriteTC(self.slm_number, int(ascending))

    # def set_output_trigger(self, enabled=True):
    #     slm_funcs.SLM_Ctrl_WriteTM(self.slm_number, int(enabled))

    # def set_memory_framerate(self, framerate_hz):
    #     # Not sure if it's actually Hz
    #     assert framerate_hz >= 0
    #     assert framerate_hz <= 120
    #     slm_funcs.SLM_Ctrl_WriteMW(self.slm_number, int(framerate_hz))

    # def get_memory_framerate(self):
    #     framerate_hz = slm_funcs.DWORD(0)

    #     slm_funcs.SLM_Ctrl_ReadMW(self.slm_number, framerate_hz)

    #     return int(framerate_hz.value)

    # def start_memory_continuous(self, ascending=True):
    #     slm_funcs.SLM_Ctrl_WriteDR(self.slm_number, int(ascending))

    # def stop_memory_continuous(self):
    #     slm_funcs.SLM_Ctrl_WriteDB(self.slm_number)

    # def set_memory_table(self, mapping):
    #     assert len(mapping) == 128

    #     for entry, map in zip(range(1, 129), mapping):
    #         Santec._parse_status(
    #             slm_funcs.SLM_Ctrl_WriteMT(self.slm_number, entry, int(map))
    #         )

    # def get_memory_table(self):
    #     table = []

    #     ptr = slm_funcs.DWORD(0)

    #     for entry in range(1, 129):
    #         Santec._parse_status(
    #             slm_funcs.SLM_Ctrl_ReadMS(self.slm_number, entry, ptr)
    #         )
    #         table.append(int(ptr.value))

    #     return table

    # def set_memory_table_position(self, position):
    #     slm_funcs.SLM_Ctrl_WriteMP(self.slm_number, int(position))

    # def get_memory_table_position(self, position):
    #     # Function does not exist?
    #     slm_funcs.SLM_Ctrl_ReadMP(self.slm_number, int(position))

    # def set_memory_table_range(self, start, end):
    #     # end is one above the true end, according to docs.
    #     slm_funcs.SLM_Ctrl_WriteMR(self.slm_number, int(start), int(end))

    # def get_memory_table_range(self):
    #     start = slm_funcs.DWORD(0)
    #     end = slm_funcs.DWORD(0)

    #     slm_funcs.SLM_Ctrl_ReadMR(self.slm_number, start, end)

    #     return (start.value, end.value)

    # def set_phase_memory(self, phase):
    #     # Needs to be combined with set_phase(), probably.
    #     slm_funcs.SLM_Ctrl_WriteMI()
    #     raise NotImplementedError()
