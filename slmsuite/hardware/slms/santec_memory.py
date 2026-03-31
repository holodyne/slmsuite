"""
Hardware control for Santec SLMs in memory mode.
"""

import os
import ctypes
import numpy as np
import cv2
import warnings
from PIL import Image
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import ctypes, time

# Define SLM_OK and SLM_BS if not present in DLL
SLM_OK = 0
SLM_BS = 2
import matplotlib.pyplot as plt

from .slm import SLM

try:  # Load Santec's header file.
    from . import _slm_win as slm_funcs
except BaseException as e:  # Provide an informative error should something go wrong.
    warnings.warn(
        "Santec DLLs not installed. Install these to use Santec  SLMs."
        "  Dynamically linked libraries from Santec (usually provided via USB) "
        "must be present in the runtime directory:\n"
        "  - SLMFunc.dll\n  - FTD3XX.dll\n"
        "  Check that these files are present and are error-free.\n"
        "Original error: {}".format(e)
    )
    slm_funcs = None

class Santec_memory(SLM):
    """
    Interfaces with Santec SLMs in memory mode.
    
    Attributes
    ----------
    slm_number : int
        USB port number assigned by Santec SDK.
    memory_id : int
        Base memory block ID for memory mode operations.
    memory_buffer_size : int
        Number of memory blocks used as buffer.
    current_display_memory : int
        Currently displayed memory block ID.
    next_write_memory : int
        Next memory block ID to write to.
    driveboard_id : str
        ID of the drive board.
    optionboard_id : str
        ID of the option board.
    product_code_id : str
        Product code of the device.
    """

    def __init__(
            self,
            slm_number=1,
            display_number=2,  # Display number, defaults to 2
            memory_id=1,
            memory_buffer_size=2,  # Memory buffer size
            bitdepth=10,
            wav_um=1,
            pitch_um=(8,8),
            wavelength_set=532,   # Laser wavelength in nm
            phase_set=200,        # Maximum phase range (200=2pi)
            verbose=True,
            **kwargs
        ):
        r"""
        Initializes an instance of a Santec SLM in memory mode.

        Parameters
        ----------
        slm_number : int
            USB port number assigned by Santec SDK.
        memory_id : int
            Memory block ID to use for memory mode (1~128).
        memory_buffer_size : int
            Number of memory blocks to use as buffer. Defaults to 2.
            Helps avoid overwriting currently displayed memory.
        bitdepth : int
            Depth of SLM pixel well in bits. Defaults to 10.
        wav_um : float
            Wavelength of operation in microns. Defaults to 1 um.
        pitch_um : (float, float)
            Pixel pitch in microns. Defaults to 8 micron square pixels.
        wavelength_set : int
            Laser wavelength in nm, used for phase table.
        phase_set : int
            Max phase range (e.g. 200 means 2*pi).
        verbose : bool
            Whether to print extra information.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.

        Note
        ----
        This class is for Santec SLMs operated in memory mode.
        """
        if slm_funcs is None:
            raise RuntimeError(
                "Santec DLLs not installed. Install these to use Santec SLMs."
                "  Dynamically linked libraries from Santec (usually provided via USB) "
                "must be present in the runtime directory:\n"
                "  - SLMFunc.dll\n  - FTD3XX.dll\n"
                "  Check that these files are present and are error-free."
            )

        self.slm_number = int(slm_number)
        self.display_number = int(display_number)
        self.memory_id = int(memory_id)
        self.memory_buffer_size = int(memory_buffer_size)
        self.bitdepth = int(bitdepth)
        self.wav_um = wav_um
        self.pitch_um = pitch_um
        self.verbose = verbose
        self.wavelength_set = int(wavelength_set)
        self.phase_set = int(phase_set)

        if verbose:
            print(f"Santec_memory slm_number={self.slm_number} initializing in memory mode... ", end="")
        # Open SLM

        status = slm_funcs.SLM_Ctrl_Open(self.slm_number)
        if status != SLM_OK:
            raise RuntimeError(f"Failed to open SLM device {self.slm_number}, status={status}")
        else:
            print(f"SLM device {self.slm_number} opened successfully.")

        # Wait for SLM ready
        while True:
            su = slm_funcs.SLM_Ctrl_ReadSU(self.slm_number)
            if su == SLM_OK:
                break
            elif su == SLM_BS:
                continue
            else:
                raise RuntimeError(f"SLM status error: {su}")

        # Read current wavelength and maximum phase
        current_wl = ctypes.c_uint32(0)
        current_phase = ctypes.c_uint32(0)
        status = slm_funcs.SLM_Ctrl_ReadWL(self.slm_number, current_wl, current_phase)
        if status != SLM_OK:
            raise RuntimeError("Failed to read current wavelength/phase.")
        
        current_wl_val = current_wl.value
        current_phase_val = current_phase.value
        
        if (current_wl_val != self.wavelength_set):
            # Only write if inconsistent
            status = slm_funcs.SLM_Ctrl_WriteWL(self.slm_number, self.wavelength_set, self.phase_set)
            if status != SLM_OK:
                raise RuntimeError("Failed to set wavelength/phase table.")
            status = slm_funcs.SLM_Ctrl_WriteAW(self.slm_number)
            if status != SLM_OK:
                raise RuntimeError("Failed to save wavelength/phase table.")
            if self.verbose:
                print(f"Set wavelength to {self.wavelength_set} nm, phase_set to {self.phase_set}.")
        else:
            if self.verbose:
                print(f"Wavelength and phase_set already match ({current_wl_val} nm, {current_phase_val}). No update needed.")
        
        # Switch to memory mode
        status = slm_funcs.SLM_Ctrl_WriteVI(self.slm_number, 0)  # 0: memory mode
        if status != SLM_OK:
            raise RuntimeError("Failed to set SLM to memory mode.")
        if self.verbose:
            print("SLM set to memory mode.")
        
        # Query resolution and name
        width = ctypes.c_ushort(0)
        height = ctypes.c_ushort(0)
        display_name = ctypes.create_string_buffer(128)
        status = slm_funcs.SLM_Disp_Info2(self.display_number, width, height, display_name)
        if status != SLM_OK and status != -1:
            raise RuntimeError(f"SLM_Disp_Info2 failed, status={status}")
        
        # print(width)  # Debug output
        # print(height)  # Debug output

        # Ensure correct (height, width) order - should be (1200, 1920)
        self.shape = (int(width.value), int(height.value))
        self.name = display_name.value.decode("mbcs")

        # print(self.shape)  # Debug output
        
        if self.verbose:
            # print(f"DEBUG: width={width.value}, height={height.value}")  # Debug output
            # print(f"DEBUG: self.shape set to (height, width) = {self.shape}")  # Debug output
            pass
        
        # Read board IDs, product codes, etc.
        driveboard_id = ctypes.create_string_buffer(16)
        optionboard_id = ctypes.create_string_buffer(16)
        slm_funcs.SLM_Ctrl_ReadSDO(self.slm_number, driveboard_id, optionboard_id)
        self.driveboard_id = driveboard_id.value.decode("mbcs")
        self.optionboard_id = optionboard_id.value.decode("mbcs")
        names = self.name.split(",")
        self.product_code_id = names[2] if len(names) > 2 else ""

        # print(f"before super_init{self.shape}")  # Debug output
        super().__init__(
            self.shape,  # Use properly set self.shape (height, width)
            bitdepth=bitdepth,
            name=kwargs.pop("name", names[-1] if len(names) > 3 else f"SLM-{self.slm_number}"),  # SerialNumberID or default name
            wav_um=wav_um,
            wav_design_um=kwargs.pop("wav_design_um", wav_um),  # Default design wavelength equals working wavelength
            pitch_um=pitch_um,
            **kwargs  # settle_time_s and other parameters passed through kwargs
        )
        # print(f"super_init{self.shape}")  # Debug output
        # Add status tracking variables
        self.is_open = True
        self.current_display_memory = self.memory_id  # Currently displayed memory block
        self.next_write_memory = self._get_next_memory_id(self.memory_id)  # Next memory block to write to
        self.last_phase = None
        
        if self.verbose:
            print(f"SLM resolution: {int(height.value)}x{int(width.value)}")
            print(f"Product info: {self.name}")
            print(f"Drive board ID: {self.driveboard_id}")
            print(f"Option board ID: {self.optionboard_id}")

        # Initialize display with zero phase
        if self.verbose:
            print("Initializing SLM with zero phase...")
        self._set_phase_hw(None)
        
        if self.verbose:
            print("Santec memory mode SLM initialization complete.")

    def _get_next_memory_id(self, current_memory):
        """
        Get the next available memory block ID, avoiding overwriting the currently displayed memory.
        Uses circular buffer strategy.
        
        Parameters
        ----------
        current_memory : int
            Current memory block ID
            
        Returns
        -------
        int
            Next available memory block ID
        """
        next_id = current_memory + 1
        if next_id > 128:
            next_id = 1
        return next_id

    def close(self):
        """Close the SLM and delete related objects."""
        if hasattr(self, 'is_open') and self.is_open:
            status = slm_funcs.SLM_Ctrl_Close(self.slm_number)
            if status == SLM_OK:
                if self.verbose:
                    print(f"SLM device {self.slm_number} closed successfully.")
            else:
                if self.verbose:
                    print(f"Warning: Failed to close SLM device {self.slm_number}, status={status}")
            self.is_open = False
        else:
            if self.verbose:
                print("SLM already closed or not opened.")

    @staticmethod
    def info(verbose=True):
        """
        Discovers all SLMs detected by the Santec SDK.
        
        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of dict
            List containing information about detected SLMs.
        """
        if slm_funcs is None:
            if verbose:
                print("Santec DLLs not available.")
            return []
        
        slm_list = []
        for slm_num in range(1, 8):  # Check SLM 1-7
            try:
                # Try to open SLM
                status = slm_funcs.SLM_Ctrl_Open(slm_num)
                if status == SLM_OK:
                    # Read device information
                    width = ctypes.c_ushort(0)
                    height = ctypes.c_ushort(0)
                    display_name = ctypes.create_string_buffer(128)
                    slm_funcs.SLM_Disp_Info2(slm_num, width, height, display_name)
                    
                    slm_info = {
                        'slm_number': slm_num,
                        'name': display_name.value.decode("mbcs"),
                        'resolution': (int(width.value), int(height.value))
                    }
                    slm_list.append(slm_info)
                    
                    if verbose:
                        print(f"Found SLM {slm_num}: {slm_info['name']} "
                              f"({slm_info['resolution'][1]}x{slm_info['resolution'][0]})")
                    
                    # Close SLM
                    slm_funcs.SLM_Ctrl_Close(slm_num)
            except:
                continue
        
        if verbose and not slm_list:
            print("No Santec SLMs detected.")
            
        return slm_list
    def _set_phase_hw(self, phase):
        """
        Low-level hardware interface to set_phase ``phase`` data onto the SLM in memory mode.
        When the user calls the :meth:`.SLM.write` method of
        :class:`.SLM`, ``phase`` is error checked before calling
        :meth:`_set_phase_hw()`. See :meth:`.SLM._set_phase_hw` for further detail.
        
        Parameters
        ----------
        phase : numpy.ndarray or None
            Integer phase data to be written to the SLM memory.
            Should be of type uint16 with values in range [0, 1023] for 10-bit SLM.
            If None, displays uniform zero grayscale.
        """

        # print(f"set_phase_hw {self.shape}")  # Debug output

        # Handle None case: directly display zero grayscale
        if phase is None:
            if self.verbose:
                print("Setting SLM to zero grayscale (gs=0)...")
            
            status = slm_funcs.SLM_Ctrl_WriteGS(self.slm_number, 0)
            if status != SLM_OK:
                raise RuntimeError(f"Failed to set SLM to zero grayscale, status={status}")
            
            if self.verbose:
                print("Successfully set SLM to zero grayscale.")
            
            # Clear tracking state since we're not displaying from memory now
            self.current_display_memory = None
            self.last_phase = None
            return
        
        # Handle normal phase data: write to memory and display
        # Output phase shape information
        # print(f"Original Phase shape: {phase.shape}")  # Debug output
        # print(f"Expected SLM shape (h, w): {self.shape}")  # Debug output
        
        # Check if transpose is needed
        if phase.shape != self.shape:
            if phase.shape == (self.shape[1], self.shape[0]):
                # print(f"Phase shape {phase.shape} doesn't match SLM shape {self.shape}, transposing...")  # Debug output
                phase = phase.T
                # print(f"After transpose, phase shape: {phase.shape}")  # Debug output
            else:
                if self.verbose:
                    print(f"Warning: Phase shape {phase.shape} doesn't match SLM shape {self.shape} even after considering transpose!")
        # else:
            # print(f"Phase shape matches SLM shape: {phase.shape}")  # Debug output
        
        matrix = phase.astype(np.ushort)
        n_h, n_w = self.shape
        
        # Correct method: use .contents (exactly following the example program)
        c_matrix = matrix.ctypes.data_as(ctypes.POINTER((ctypes.c_ushort * n_h) * n_w)).contents
        
        # Use the next available memory block to avoid overwriting currently displayed memory
        target_memory_id = self.next_write_memory
        
        # ===== Debug information: detailed output of all data types and parameters =====
        if self.verbose and False:  # Debug code, disabled
            print("=" * 60)
            print("DEBUG: WriteMI debug information")
            print("=" * 60)
            
            # Original input data
            print(f"Original phase type: {type(phase)}")
            print(f"Original phase shape: {phase.shape}")
            print(f"Original phase dtype: {phase.dtype}")
            print(f"Original phase value range: [{phase.min()}, {phase.max()}]")
            print(f"Original phase memory flags: {phase.flags}")
            print()
            
            # Converted matrix data
            print(f"Converted matrix type: {type(matrix)}")
            print(f"Converted matrix shape: {matrix.shape}")
            print(f"Converted matrix dtype: {matrix.dtype}")
            print(f"Converted matrix value range: [{matrix.min()}, {matrix.max()}]")
            print(f"Converted matrix memory flags: {matrix.flags}")
            print()
            
            # SLM parameters
            print(f"SLM shape (n_h, n_w): ({n_h}, {n_w})")
            print(f"SLM device number: {self.slm_number}")
            print(f"Target memory block ID: {target_memory_id}")
            print()
            
            # ctypes conversion related
            pointer_obj = matrix.ctypes.data_as(ctypes.POINTER((ctypes.c_ushort * n_h) * n_w))
            print(f"ctypes pointer type: {type(pointer_obj)}")
            print(f"ctypes .contents type: {type(c_matrix)}")
            print(f"ctypes .contents value: {c_matrix}")
            print()
            
            # Check memory contiguity
            print(f"matrix C contiguous: {matrix.flags['C_CONTIGUOUS']}")
            print(f"matrix F contiguous: {matrix.flags['F_CONTIGUOUS']}")
            print(f"matrix writable: {matrix.flags['WRITEABLE']}")
            print()
            
            # WriteMI function parameters
            print("SLM_Ctrl_WriteMI parameters:")
            print(f"  Param1 (SLM number): {self.slm_number} (type: {type(self.slm_number)})")
            print(f"  Param2 (memory ID): {target_memory_id} (type: {type(target_memory_id)})")
            print(f"  Param3 (width): {n_w} (type: {type(n_w)})")
            print(f"  Param4 (height): {n_h} (type: {type(n_h)})")
            print(f"  Param5 (flags): 0 (type: {type(0)})")
            print(f"  Param6 (data): {c_matrix} (type: {type(c_matrix)})")
            print("=" * 60)
        
        # Step 1: Write phase data to SLM memory
        status = slm_funcs.SLM_Ctrl_WriteMI(
            self.slm_number,     # SLM 设备号
            target_memory_id,    # 目标内存块 ID
            n_w,                 # 宽度
            n_h,                 # 高度
            0,                   # flags (通常为0)
            c_matrix             # Pass .contents (example program's approach)
        )
        
        # Output WriteMI result
        if self.verbose and False:  # Debug code, disabled
            print(f"SLM_Ctrl_WriteMI return status: {status}")
            if status == 0:
                print("✓ WriteMI success")
            else:
                print(f"✗ WriteMI failed, status code: {status}")
        
        if status != SLM_OK:
            raise RuntimeError(f"Failed to write phase data to SLM memory {target_memory_id}, status={status}")
        
        # Step 2: Display data from specified memory
        status = slm_funcs.SLM_Ctrl_WriteDS(self.slm_number, target_memory_id)
        
        # Output WriteDS result
        if self.verbose and False:  # Debug code, disabled
            print(f"SLM_Ctrl_WriteDS return status: {status}")
            if status == 0:
                print("✓ WriteDS success")
            else:
                print(f"✗ WriteDS failed, status code: {status}")
        
        if status != SLM_OK:
            raise RuntimeError(f"Failed to display memory {target_memory_id}, status={status}")
        
        # Update status tracking
        self.current_display_memory = target_memory_id
        self.next_write_memory = self._get_next_memory_id(target_memory_id)
        self.last_phase = matrix.copy()
        
        if self.verbose and False:  # Debug code, disabled
            print(f"Successfully displayed memory {target_memory_id}. Next write will use memory {self.next_write_memory}.")
            print("=" * 60)
    # def _set_phase_hw(self, phase):
    #     """
    #     Low-level hardware interface to set_phase ``phase`` data onto the SLM in memory mode.
    #     When the user calls the :meth:`.SLM.write` method of
    #     :class:`.SLM`, ``phase`` is error checked before calling
    #     :meth:`_set_phase_hw()`. See :meth:`.SLM._set_phase_hw` for further detail.
        
    #     Parameters
    #     ----------
    #     phase : numpy.ndarray
    #         Integer phase data to be written to the SLM memory.
    #         Should be of type uint16 with values in range [0, 1023] for 10-bit SLM.
    #     """
    #     # 确保数据类型正确
    #     matrix = phase.astype(np.ushort)
    #     n_h, n_w = self.shape
        
    #     # 正确的方法：使用 .contents（完全按照示例程序）
    #     c_matrix = matrix.ctypes.data_as(ctypes.POINTER((ctypes.c_ushort * n_h) * n_w)).contents
        
    #     # 使用下一个可用的内存块，避免覆写当前显示的内存
    #     target_memory_id = self.next_write_memory
        
    #     if self.verbose:
    #         print(f"Writing phase data to memory block {target_memory_id}...")
    #         print(f"DEBUG: SLM shape = {self.shape}")
    #         print(f"DEBUG: n_h = {n_h}, n_w = {n_w}")
    #         print(f"DEBUG: matrix shape = {matrix.shape}")
        
    #     # 步骤1: 将相位数据写入 SLM 内存
    #     status = slm_funcs.SLM_Ctrl_WriteMI(
    #         self.slm_number,     # SLM 设备号
    #         target_memory_id,    # 目标内存块 ID
    #         n_w,                 # 宽度
    #         n_h,                 # 高度
    #         0,                   # flags (通常为0)
    #         c_matrix             # 传递 .contents（示例程序的做法）
    #     )
    #     if status != 0:  # SLM_OK = 0
    #         raise RuntimeError(f"Failed to write phase data to SLM memory {target_memory_id}, status={status}")
        
    #     # 步骤2: 显示指定内存中的数据
    #     status = slm_funcs.SLM_Ctrl_WriteDS(self.slm_number, target_memory_id)
    #     if status != 0:  # SLM_OK = 0
    #         raise RuntimeError(f"Failed to display memory {target_memory_id}, status={status}")
        
    #     # 更新状态跟踪
    #     self.current_display_memory = target_memory_id
    #     self.next_write_memory = self._get_next_memory_id(target_memory_id)
    #     self.last_phase = matrix.copy()
        
    #     if self.verbose:
    #         print(f"Successfully displayed memory {target_memory_id}. Next write will use memory {self.next_write_memory}.")

    def get_temperature(self):
        """
        Read the drive board and option board temperatures.

        Returns
        -------
        (float, float)
            Temperature in Celsius of the drive and option board
        """
        drive_temp = ctypes.c_uint32(0)
        option_temp = ctypes.c_uint32(0)

        status = slm_funcs.SLM_Ctrl_ReadT(self.slm_number, drive_temp, option_temp)
        if status != SLM_OK:
            raise RuntimeError(f"Failed to read temperature, status={status}")

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

        status = slm_funcs.SLM_Ctrl_ReadEDO(self.slm_number, drive_error, option_error)
        if status != SLM_OK and raise_error:
            raise RuntimeError(f"Failed to read error status, status={status}")

        # Simple error checking - just return the raw error codes
        errors = []
        if drive_error.value != 0:
            errors.append(f"Drive board error: {drive_error.value}")
        if option_error.value != 0:
            errors.append(f"Option board error: {option_error.value}")

        if raise_error and len(errors) > 0:
            error_msg = "Santec_memory error: " + ", ".join(errors)
            if raise_error:
                raise RuntimeError(error_msg)
            else:
                warnings.warn(error_msg)

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
        status = slm_funcs.SLM_Ctrl_ReadSU(self.slm_number)
        # Simple status mapping
        status_names = {
            SLM_OK: ("SLM_OK", "Success"),
            1: ("SLM_NG", "General error"),
            SLM_BS: ("SLM_BS", "Busy"),
            3: ("SLM_ER", "Error")
        }
        name, note = status_names.get(status, ("UNKNOWN", f"Unknown status code: {status}"))
        if status != SLM_OK and raise_error:
            raise RuntimeError(f"Santec_memory status error {status}: {name} - {note}")
        return (status, name, note)

    # 删除不存在的 _parse_status 静态方法，因为引用的字典不存在

    def get_memory_info(self):
        """
        Get current memory usage information.
        
        Returns
        -------
        dict
            Dictionary containing memory usage information
        """
        return {
            'current_display_memory': self.current_display_memory,
            'next_write_memory': self.next_write_memory,
            'memory_buffer_size': self.memory_buffer_size,
            'base_memory_id': self.memory_id,
            'available_memory_range': (1, 128)
        }

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
