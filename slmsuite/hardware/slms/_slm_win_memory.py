import sys
from ctypes import *


STRING = c_char_p
from ctypes.wintypes import DWORD
from ctypes.wintypes import ULONG
from ctypes.wintypes import WORD
from ctypes.wintypes import BYTE
from ctypes.wintypes import BOOL
from ctypes.wintypes import BOOLEAN
from ctypes.wintypes import LPCSTR
from ctypes.wintypes import HANDLE
from ctypes.wintypes import LONG
from ctypes.wintypes import UINT
from ctypes.wintypes import LPSTR
from ctypes.wintypes import LPCSTR
from ctypes.wintypes import LPCWSTR
from ctypes.wintypes import FILETIME
import os
import platform

pf = platform.system()
    
    
FLAGS_COLOR_R    = 0x00000001
FLAGS_COLOR_G    = 0x00000002
FLAGS_COLOR_B    = 0x00000004
FLAGS_COLOR_GRAY = 0x00000008
FLAGS_RATE120    = 0x20000000

SLM_OK = 0
SLM_NG = 1
SLM_BS = 2
SLM_ER = 3

if (sys.version_info.major*100 + sys.version_info.minor) >= 308:
    os.add_dll_directory(os.getcwd())



USHORT = c_ushort
SHORT = c_short
UCHAR = c_ubyte
WCHAR = c_wchar
LPBYTE = POINTER(c_ubyte)
LPUSHORT = POINTER(c_ushort)
CHAR = c_char
LPBOOL = POINTER(c_int)
PUCHAR = POINTER(c_ubyte)
PCHAR = STRING
PVOID = c_void_p
INT = c_int
LPTSTR = STRING
LPDWORD = POINTER(c_uint32)
LPWORD = POINTER(WORD)
PULONG = POINTER(ULONG)
LPVOID = PVOID
VOID = None
ULONGLONG = c_ulonglong
SLM_STATUS = c_int32


_libraries = {}

if pf == 'Darwin':
    _libname = 'SLMFunc.dylib'
    _libraries[_libname] = cdll.LoadLibrary(_libname)
    _FILENAME = STRING
elif pf == 'Windows':
    _libname = 'SLMFunc.dll'
    path = os.path.abspath(_libname)
    _libraries[_libname] = WinDLL(_libname)
    _FILENAME = LPCWSTR
    _FILENAME_A = LPCSTR

SLM_Disp_Info = _libraries[_libname].SLM_Disp_Info
SLM_Disp_Info.restype = SLM_STATUS
SLM_Disp_Info.argtypes = [DWORD, LPUSHORT, LPUSHORT]
SLM_Disp_Info.__doc__ = \
"""SLM_Disp_Info(DWORD DisplayNumber, USHORT *width, USHORT *height)"""

SLM_Disp_Info2 = _libraries[_libname].SLM_Disp_Info2
SLM_Disp_Info2.restype = SLM_STATUS
SLM_Disp_Info2.argtypes = [DWORD, LPUSHORT, LPUSHORT, LPSTR]
SLM_Disp_Info2.__doc__ = \
"""SLM_Disp_Info2(DWORD DisplayNumber, USHORT *width, USHORT *height, LPSTR DisplayName )"""

SLM_Disp_Open = _libraries[_libname].SLM_Disp_Open
SLM_Disp_Open.restype = SLM_STATUS
SLM_Disp_Open.argtypes = [DWORD]
SLM_Disp_Open.__doc__ = \
"""SLM_Disp_Open(DWORD DisplayNumber)"""

SLM_Disp_Close = _libraries[_libname].SLM_Disp_Close
SLM_Disp_Close.restype = SLM_STATUS
SLM_Disp_Close.argtypes = [DWORD]
SLM_Disp_Close.__doc__ = \
"""SLM_Disp_Close(DWORD DisplayNumber)"""

SLM_Disp_GrayScale = _libraries[_libname].SLM_Disp_GrayScale
SLM_Disp_GrayScale.restype = SLM_STATUS
SLM_Disp_GrayScale.argtypes = [DWORD, DWORD, USHORT]
SLM_Disp_GrayScale.__doc__ = \
"""SLM_Disp_GrayScale(DWORD DisplayNumber, DWORD Flags, USHORT GrayScale)"""

#SLM_Disp_BMP = _libraries[_libname].SLM_Disp_BMP
#SLM_Disp_BMP.restype = SLM_STATUS
#SLM_Disp_BMP.argtypes = [DWORD, DWORD, HBITMAP]
#SLM_Disp_BMP.__doc__ = \
#"""SLM_Disp_BMP(DWORD DisplayNumber, DWORD Flags, HBITMAP bmp)"""

SLM_Disp_Data = _libraries[_libname].SLM_Disp_Data
SLM_Disp_Data.restype = SLM_STATUS
SLM_Disp_Data.argtypes = [DWORD, USHORT, USHORT, DWORD, c_void_p]
SLM_Disp_Data.__doc__ = \
"""SLM_Disp_Data(DWORD DisplayNumber, USHORT width, USHORT height, DWORD Flags, USHORT* data)"""

SLM_Disp_ReadBMP = _libraries[_libname].SLM_Disp_ReadBMP
SLM_Disp_ReadBMP.restype = SLM_STATUS
SLM_Disp_ReadBMP.argtypes = [DWORD, DWORD, _FILENAME]
SLM_Disp_ReadBMP.__doc__ = \
"""SLM_Disp_ReadBMP(DWORD DisplayNumber, DWORD Flags, LPCWSTR FileName)"""

SLM_Disp_ReadCSV = _libraries[_libname].SLM_Disp_ReadCSV
SLM_Disp_ReadCSV.restype = SLM_STATUS
SLM_Disp_ReadCSV.argtypes = [DWORD, DWORD, _FILENAME]
SLM_Disp_ReadCSV.__doc__ = \
"""SLM_Disp_ReadCSV(DWORD DisplayNumber, DWORD Flags, LPCWSTR FileName)"""

if pf == 'Windows':
    SLM_Disp_ReadBMP_A = _libraries[_libname].SLM_Disp_ReadBMP_A
    SLM_Disp_ReadBMP_A.restype = SLM_STATUS
    SLM_Disp_ReadBMP_A.argtypes = [DWORD, DWORD, _FILENAME_A]
    SLM_Disp_ReadBMP_A.__doc__ = \
    """SLM_Disp_ReadBMP_A(DWORD DisplayNumber, DWORD Flags, LPCSTR FileName)"""
    
    SLM_Disp_ReadCSV_A = _libraries[_libname].SLM_Disp_ReadCSV_A
    SLM_Disp_ReadCSV_A.restype = SLM_STATUS
    SLM_Disp_ReadCSV_A.argtypes = [DWORD, DWORD, _FILENAME]
    SLM_Disp_ReadCSV_A.__doc__ = \
    """SLM_Disp_ReadCSV_A(DWORD DisplayNumber, DWORD Flags, LPCSTR FileName)"""

SLM_Ctrl_Open = _libraries[_libname].SLM_Ctrl_Open
SLM_Ctrl_Open.restype = SLM_STATUS
SLM_Ctrl_Open.argtypes = [DWORD]
SLM_Ctrl_Open.__doc__ = \
"""SLM_Ctrl_Open(DWORD SLMNumber)"""

SLM_Ctrl_Close = _libraries[_libname].SLM_Ctrl_Close
SLM_Ctrl_Close.restype = SLM_STATUS
SLM_Ctrl_Close.argtypes = [DWORD]
SLM_Ctrl_Close.__doc__ = \
"""SLM_Ctrl_Close(DWORD SLMNumber)"""

if pf == 'Windows':
    SLM_Ctrl_Read = _libraries[_libname].SLM_Ctrl_Read
    SLM_Ctrl_Read.restype = SLM_STATUS
    SLM_Ctrl_Read.argtypes = [DWORD, LPBYTE, LPUSHORT]
    SLM_Ctrl_Read.__doc__ = \
    """SLM_Ctrl_Read(DWORD SLMNumber, BYTE* recv, USHORT* recv_len)"""

SLM_Ctrl_WriteVI = _libraries[_libname].SLM_Ctrl_WriteVI
SLM_Ctrl_WriteVI.restype = SLM_STATUS
SLM_Ctrl_WriteVI.argtypes = [DWORD, DWORD]
SLM_Ctrl_WriteVI.__doc__ = \
"""SLM_Ctrl_WriteVI(DWORD SLMNumber, DWORD mode)"""

SLM_Ctrl_ReadVI = _libraries[_libname].SLM_Ctrl_ReadVI
SLM_Ctrl_ReadVI.restype = SLM_STATUS
SLM_Ctrl_ReadVI.argtypes = [DWORD, LPDWORD]
SLM_Ctrl_ReadVI.__doc__ = \
"""SLM_Ctrl_ReadVI(DWORD SLMNumber, DWORD *mode)"""

SLM_Ctrl_WriteWL = _libraries[_libname].SLM_Ctrl_WriteWL
SLM_Ctrl_WriteWL.restype = SLM_STATUS
SLM_Ctrl_WriteWL.argtypes = [DWORD, DWORD, DWORD]
SLM_Ctrl_WriteWL.__doc__ = \
"""SLM_Ctrl_WriteWL(DWORD SLMNumber, DWORD wavelength, float phase)"""

SLM_Ctrl_ReadWL = _libraries[_libname].SLM_Ctrl_ReadWL
SLM_Ctrl_ReadWL.restype = SLM_STATUS
SLM_Ctrl_ReadWL.argtypes = [DWORD, LPDWORD, LPDWORD]
SLM_Ctrl_ReadWL.__doc__ = \
"""SLM_Ctrl_ReadWL(DWORD SLMNumber, DWORD *wavelength, float *phase)"""

SLM_Ctrl_WriteAW = _libraries[_libname].SLM_Ctrl_WriteAW
SLM_Ctrl_WriteAW.restype = SLM_STATUS
SLM_Ctrl_WriteAW.argtypes = [DWORD]
SLM_Ctrl_WriteAW.__doc__ = \
"""SLM_Ctrl_WriteAW(DWORD SLMNumber)"""

SLM_Ctrl_WriteTI = _libraries[_libname].SLM_Ctrl_WriteTI
SLM_Ctrl_WriteTI.restype = SLM_STATUS
SLM_Ctrl_WriteTI.argtypes = [DWORD, DWORD]
SLM_Ctrl_WriteTI.__doc__ = \
"""SLM_Ctrl_WriteTI(DWORD SLMNumber, DWORD onoff)"""

SLM_Ctrl_ReadTI = _libraries[_libname].SLM_Ctrl_ReadTI
SLM_Ctrl_ReadTI.restype = SLM_STATUS
SLM_Ctrl_ReadTI.argtypes = [DWORD, LPDWORD]
SLM_Ctrl_ReadTI.__doc__ = \
"""SLM_Ctrl_ReadTI(DWORD SLMNumber, DWORD *onoff)"""

SLM_Ctrl_WriteTM = _libraries[_libname].SLM_Ctrl_WriteTM
SLM_Ctrl_WriteTM.restype = SLM_STATUS
SLM_Ctrl_WriteTM.argtypes = [DWORD, DWORD]
SLM_Ctrl_WriteTM.__doc__ = \
"""SLM_Ctrl_WriteTM(DWORD SLMNumber, DWORD onoff)"""

SLM_Ctrl_ReadTM = _libraries[_libname].SLM_Ctrl_ReadTM
SLM_Ctrl_ReadTM.restype = SLM_STATUS
SLM_Ctrl_ReadTM.argtypes = [DWORD, LPDWORD]
SLM_Ctrl_ReadTM.__doc__ = \
"""SLM_Ctrl_ReadTM(DWORD SLMNumber, DWORD *onoff)"""

SLM_Ctrl_ReadTM = _libraries[_libname].SLM_Ctrl_ReadTM
SLM_Ctrl_ReadTM.restype = SLM_STATUS
SLM_Ctrl_ReadTM.argtypes = [DWORD, DWORD]
SLM_Ctrl_ReadTM.__doc__ = \
"""SLM_Ctrl_WriteTC(DWORD SLMNumber, DWORD order)"""

SLM_Ctrl_ReadTC = _libraries[_libname].SLM_Ctrl_ReadTC
SLM_Ctrl_ReadTC.restype = SLM_STATUS
SLM_Ctrl_ReadTC.argtypes = [DWORD, LPDWORD]
SLM_Ctrl_ReadTC.__doc__ = \
"""SLM_Ctrl_ReadTC(DWORD SLMNumber, DWORD *order)"""

SLM_Ctrl_WriteTS = _libraries[_libname].SLM_Ctrl_WriteTS
SLM_Ctrl_WriteTS.restype = SLM_STATUS
SLM_Ctrl_WriteTS.argtypes = [DWORD]
SLM_Ctrl_WriteTS.__doc__ = \
"""SLM_Ctrl_WriteTS(DWORD SLMNumber)"""

SLM_Ctrl_WriteMC = _libraries[_libname].SLM_Ctrl_WriteMC
SLM_Ctrl_WriteMC.restype = SLM_STATUS
SLM_Ctrl_WriteMC.argtypes = [DWORD, DWORD]
SLM_Ctrl_WriteMC.__doc__ = \
"""SLM_Ctrl_WriteMC(DWORD SLMNumber, DWORD MemoryNumber)"""

SLM_Ctrl_WriteMI = _libraries[_libname].SLM_Ctrl_WriteMI
SLM_Ctrl_WriteMI.restype = SLM_STATUS
SLM_Ctrl_WriteMI.argtypes = [DWORD, DWORD, USHORT, USHORT, DWORD, c_void_p]
SLM_Ctrl_WriteMI.__doc__ = \
"""SLM_Ctrl_WriteMI(DWORD SLMNumber, DWORD MemoryNumber, USHORT width, USHORT height, DWORD Flags, USHORT* data)"""

SLM_Ctrl_WriteMI_BMP = _libraries[_libname].SLM_Ctrl_WriteMI_BMP
SLM_Ctrl_WriteMI_BMP.restype = SLM_STATUS
SLM_Ctrl_WriteMI_BMP.argtypes = [DWORD, DWORD, DWORD, _FILENAME]
SLM_Ctrl_WriteMI_BMP.__doc__ = \
"""SLM_Ctrl_WriteMI_BMP(DWORD SLMNumber, DWORD MemoryNumber, DWORD Flags, LPCWSTR FileName)"""

SLM_Ctrl_WriteMI_CSV = _libraries[_libname].SLM_Ctrl_WriteMI_CSV
SLM_Ctrl_WriteMI_CSV.restype = SLM_STATUS
SLM_Ctrl_WriteMI_CSV.argtypes = [DWORD, DWORD, DWORD, _FILENAME]
SLM_Ctrl_WriteMI_CSV.__doc__ = \
"""SLM_Ctrl_WriteMI_CSV(DWORD SLMNumber, DWORD MemoryNumber, DWORD Flags, LPCWSTR FileName)"""


if pf == 'Windows':
    SLM_Ctrl_WriteMI_BMP_A = _libraries[_libname].SLM_Ctrl_WriteMI_BMP_A
    SLM_Ctrl_WriteMI_BMP_A.restype = SLM_STATUS
    SLM_Ctrl_WriteMI_BMP_A.argtypes = [DWORD, DWORD, DWORD, _FILENAME_A]
    SLM_Ctrl_WriteMI_BMP_A.__doc__ = \
    """SLM_Ctrl_WriteMI_BMP_A(DWORD SLMNumber, DWORD MemoryNumber, DWORD Flags, LPCSTR FileName)"""

    SLM_Ctrl_WriteMI_CSV_A = _libraries[_libname].SLM_Ctrl_WriteMI_CSV_A
    SLM_Ctrl_WriteMI_CSV_A.restype = SLM_STATUS
    SLM_Ctrl_WriteMI_CSV_A.argtypes = [DWORD, DWORD, DWORD, _FILENAME_A]
    SLM_Ctrl_WriteMI_CSV_A.__doc__ = \
    """SLM_Ctrl_WriteMI_CSV_A(DWORD SLMNumber, DWORD MemoryNumber, DWORD Flags, LPCSTR FileName)"""

SLM_Ctrl_WriteME = _libraries[_libname].SLM_Ctrl_WriteME
SLM_Ctrl_WriteME.restype = SLM_STATUS
SLM_Ctrl_WriteME.argtypes = [DWORD, DWORD]
SLM_Ctrl_WriteME.__doc__ = \
"""SLM_Ctrl_WriteME(DWORD SLMNumber, DWORD MemoryNumber)"""

SLM_Ctrl_WriteMT = _libraries[_libname].SLM_Ctrl_WriteMT
SLM_Ctrl_WriteMT.restype = SLM_STATUS
SLM_Ctrl_WriteMT.argtypes = [DWORD, DWORD, DWORD]
SLM_Ctrl_WriteMT.__doc__ = \
"""SLM_Ctrl_WriteMT(DWORD SLMNumber, DWORD TableNumber, DWORD MemoryNumber)"""

SLM_Ctrl_ReadMS = _libraries[_libname].SLM_Ctrl_ReadMS
SLM_Ctrl_ReadMS.restype = SLM_STATUS
SLM_Ctrl_ReadMS.argtypes = [DWORD, DWORD, c_void_p]
SLM_Ctrl_ReadMS.__doc__ = \
"""SLM_Ctrl_ReadMS(DWORD SLMNumber, DWORD TableNumber, DWORD *MemoryNumber)"""

SLM_Ctrl_WriteMR = _libraries[_libname].SLM_Ctrl_WriteMR
SLM_Ctrl_WriteMR.restype = SLM_STATUS
SLM_Ctrl_WriteMR.argtypes = [DWORD, DWORD, DWORD]
SLM_Ctrl_WriteMR.__doc__ = \
"""SLM_Ctrl_WriteMR(DWORD SLMNumber, DWORD TableNumber1, DWORD TableNumber2)"""

SLM_Ctrl_ReadMR = _libraries[_libname].SLM_Ctrl_ReadMR
SLM_Ctrl_ReadMR.restype = SLM_STATUS
SLM_Ctrl_ReadMR.argtypes = [DWORD, c_void_p, c_void_p]
SLM_Ctrl_ReadMR.__doc__ = \
"""SLM_Ctrl_ReadMR(DWORD SLMNumber, DWORD *TableNumber1, DWORD *TableNumber2)"""

SLM_Ctrl_WriteMP = _libraries[_libname].SLM_Ctrl_WriteMP
SLM_Ctrl_WriteMP.restype = SLM_STATUS
SLM_Ctrl_WriteMP.argtypes = [DWORD, DWORD]
SLM_Ctrl_WriteMP.__doc__ = \
"""SLM_Ctrl_WriteMP(DWORD SLMNumber, DWORD TableNumber)"""

SLM_Ctrl_WriteMZ = _libraries[_libname].SLM_Ctrl_WriteMZ
SLM_Ctrl_WriteMZ.restype = SLM_STATUS
SLM_Ctrl_WriteMZ.argtypes = [DWORD]
SLM_Ctrl_WriteMZ.__doc__ = \
"""SLM_Ctrl_WriteMZ(DWORD SLMNumber)"""

SLM_Ctrl_WriteMW = _libraries[_libname].SLM_Ctrl_WriteMW
SLM_Ctrl_WriteMW.restype = SLM_STATUS
SLM_Ctrl_WriteMW.argtypes = [DWORD,DWORD]
SLM_Ctrl_WriteMW.__doc__ = \
"""SLM_Ctrl_WriteMW(DWORD SLMNumber, DWORD frames)"""

SLM_Ctrl_ReadMW = _libraries[_libname].SLM_Ctrl_ReadMW
SLM_Ctrl_ReadMW.restype = SLM_STATUS
SLM_Ctrl_ReadMW.argtypes = [DWORD, c_void_p]
SLM_Ctrl_ReadMW.__doc__ = \
"""SLM_Ctrl_ReadMW(DWORD SLMNumber, DWORD *frames)"""

SLM_Ctrl_WriteDS = _libraries[_libname].SLM_Ctrl_WriteDS
SLM_Ctrl_WriteDS.restype = SLM_STATUS
SLM_Ctrl_WriteDS.argtypes = [DWORD, DWORD]
SLM_Ctrl_WriteDS.__doc__ = \
"""SLM_Ctrl_WriteDS(DWORD SLMNumber, DWORD MemoryNumber)"""

SLM_Ctrl_ReadDS = _libraries[_libname].SLM_Ctrl_ReadDS
SLM_Ctrl_ReadDS.restype = SLM_STATUS
SLM_Ctrl_ReadDS.argtypes = [DWORD, c_void_p]
SLM_Ctrl_ReadDS.__doc__ = \
"""SLM_Ctrl_ReadDS(DWORD SLMNumber, DWORD *MemoryNumber)"""

SLM_Ctrl_WriteDR = _libraries[_libname].SLM_Ctrl_WriteDR
SLM_Ctrl_WriteDR.restype = SLM_STATUS
SLM_Ctrl_WriteDR.argtypes = [DWORD, DWORD]
SLM_Ctrl_WriteDR.__doc__ = \
"""SLM_Ctrl_WriteDR(DWORD SLMNumber, DWORD order)"""

SLM_Ctrl_WriteDB = _libraries[_libname].SLM_Ctrl_WriteDB
SLM_Ctrl_WriteDB.restype = SLM_STATUS
SLM_Ctrl_WriteDB.argtypes = [DWORD]
SLM_Ctrl_WriteDB.__doc__ = \
"""SLM_Ctrl_WriteDB(DWORD SLMNumber)"""

SLM_Ctrl_WriteGS = _libraries[_libname].SLM_Ctrl_WriteGS
SLM_Ctrl_WriteGS.restype = SLM_STATUS
SLM_Ctrl_WriteGS.argtypes = [DWORD, c_void_p]
SLM_Ctrl_WriteGS.__doc__ = \
"""SLM_Ctrl_WriteGS(DWORD SLMNumber, USHORT GrayScale)"""

SLM_Ctrl_ReadGS = _libraries[_libname].SLM_Ctrl_ReadGS
SLM_Ctrl_ReadGS.restype = SLM_STATUS
SLM_Ctrl_ReadGS.argtypes = [DWORD, LPUSHORT]
SLM_Ctrl_ReadGS.__doc__ = \
"""SLM_Ctrl_ReadGS(DWORD SLMNumber, USHORT *GrayScale)"""

SLM_Ctrl_ReadT = _libraries[_libname].SLM_Ctrl_ReadT
SLM_Ctrl_ReadT.restype = SLM_STATUS
SLM_Ctrl_ReadT.argtypes = [DWORD, LPDWORD, LPDWORD]
SLM_Ctrl_ReadT.__doc__ = \
"""SLM_Ctrl_ReadT(DWORD SLMNumber, float *deviceTemp, float *optionTemp)"""

SLM_Ctrl_ReadTD = _libraries[_libname].SLM_Ctrl_ReadTD
SLM_Ctrl_ReadTD.restype = SLM_STATUS
SLM_Ctrl_ReadTD.argtypes = [DWORD, LPDWORD]
SLM_Ctrl_ReadTD.__doc__ = \
"""SLM_Ctrl_ReadTD(DWORD SLMNumber, float *deviceTemp)"""

SLM_Ctrl_ReadTO = _libraries[_libname].SLM_Ctrl_ReadTO
SLM_Ctrl_ReadTO.restype = SLM_STATUS
SLM_Ctrl_ReadTO.argtypes = [DWORD, LPDWORD]
SLM_Ctrl_ReadTO.__doc__ = \
"""SLM_Ctrl_ReadTO(DWORD SLMNumber, float *optionTemp)"""

SLM_Ctrl_ReadEDO = _libraries[_libname].SLM_Ctrl_ReadEDO
SLM_Ctrl_ReadEDO.restype = SLM_STATUS
SLM_Ctrl_ReadEDO.argtypes = [DWORD, LPDWORD, LPDWORD]
SLM_Ctrl_ReadEDO.__doc__ = \
"""SLM_Ctrl_ReadEDO(DWORD SLMNumber, DWORD *deviceError, DWORD *optionError)"""

SLM_Ctrl_ReadED = _libraries[_libname].SLM_Ctrl_ReadED
SLM_Ctrl_ReadED.restype = SLM_STATUS
SLM_Ctrl_ReadED.argtypes = [DWORD, LPDWORD]
SLM_Ctrl_ReadED.__doc__ = \
"""SLM_Ctrl_ReadED(DWORD SLMNumber, DWORD *deviceError)"""

SLM_Ctrl_ReadEO = _libraries[_libname].SLM_Ctrl_ReadEO
SLM_Ctrl_ReadEO.restype = SLM_STATUS
SLM_Ctrl_ReadEO.argtypes = [DWORD, LPDWORD]
SLM_Ctrl_ReadEO.__doc__ = \
"""SLM_Ctrl_ReadEO(DWORD SLMNumber, DWORD *optionError)"""

SLM_Ctrl_ReadSU = _libraries[_libname].SLM_Ctrl_ReadSU
SLM_Ctrl_ReadSU.restype = SLM_STATUS
SLM_Ctrl_ReadSU.argtypes = [DWORD]
SLM_Ctrl_ReadSU.__doc__ = \
"""SLM_Ctrl_ReadSU(DWORD SLMNumber)"""

SLM_Ctrl_ReadSDO = _libraries[_libname].SLM_Ctrl_ReadSDO
SLM_Ctrl_ReadSDO.restype = SLM_STATUS
SLM_Ctrl_ReadSDO.argtypes = [DWORD, LPSTR, LPSTR]
SLM_Ctrl_ReadSDO.__doc__ = \
"""SLM_Ctrl_ReadSDO(DWORD SLMNumber, LPSTR deviceID, LPSTR optionID)"""

SLM_Ctrl_ReadSD = _libraries[_libname].SLM_Ctrl_ReadSD
SLM_Ctrl_ReadSD.restype = SLM_STATUS
SLM_Ctrl_ReadSD.argtypes = [DWORD, LPSTR]
SLM_Ctrl_ReadSD.__doc__ = \
"""SLM_Ctrl_ReadSD(DWORD SLMNumber, LPSTR deviceID)"""

SLM_Ctrl_ReadSO = _libraries[_libname].SLM_Ctrl_ReadSO
SLM_Ctrl_ReadSO.restype = SLM_STATUS
SLM_Ctrl_ReadSO.argtypes = [DWORD, LPSTR]
SLM_Ctrl_ReadSO.__doc__ = \
"""SLM_Ctrl_ReadSO(DWORD SLMNumber, LPSTR optionID)"""

SLM_Ctrl_ReadPS = _libraries[_libname].SLM_Ctrl_ReadPS
SLM_Ctrl_ReadPS.restype = SLM_STATUS
SLM_Ctrl_ReadPS.argtypes = [DWORD, DWORD, LPSTR]
SLM_Ctrl_ReadPS.__doc__ = \
"""SLM_Ctrl_ReadPS(DWORD SLMNumber, DWOER BoardNo LPSTR ProductID)"""

SLM_Ctrl_ReadLS = _libraries[_libname].SLM_Ctrl_ReadLS
SLM_Ctrl_ReadLS.restype = SLM_STATUS
SLM_Ctrl_ReadLS.argtypes = [DWORD, DWORD, LPSTR]
SLM_Ctrl_ReadLS.__doc__ = \
"""SLM_Ctrl_ReadLS(DWORD SLMNumber, DWOER BoardNo LPSTR LCOSID)"""

SLM_Ctrl_ReadVR = _libraries[_libname].SLM_Ctrl_ReadVR
SLM_Ctrl_ReadVR.restype = SLM_STATUS
SLM_Ctrl_ReadVR.argtypes = [DWORD, LPSTR]
SLM_Ctrl_ReadVR.__doc__ = \
"""SLM_Ctrl_ReadVR(DWORD SLMNumber, LPSTR DLL_DRIVE_BOARD_FPGA_ver)"""

SLM_Ctrl_Reboot = _libraries[_libname].SLM_Ctrl_Reboot
SLM_Ctrl_Reboot.restype = SLM_STATUS
SLM_Ctrl_Reboot.argtypes = [DWORD]
SLM_Ctrl_Reboot.__doc__ = \
"""SLM_Ctrl_Reset(DWORD SLMNumber)"""


SLM_Ctrl_ReadPN = _libraries[_libname].SLM_Ctrl_ReadPN
SLM_Ctrl_ReadPN.restype = SLM_STATUS
SLM_Ctrl_ReadPN.argtypes = [DWORD, LPSTR]
SLM_Ctrl_ReadPN.__doc__ = \
"""SLM_Ctrl_ReadPN(DWORD SLMNumber, LPSTR DisplayProductID)"""

SLM_Ctrl_WritePN = _libraries[_libname].SLM_Ctrl_WritePN
SLM_Ctrl_WritePN.restype = SLM_STATUS
SLM_Ctrl_WritePN.argtypes = [DWORD, _FILENAME_A]
SLM_Ctrl_WritePN.__doc__ = \
"""SLM_Ctrl_Write(DWORD SLMNumber, LPCSTR DisplayProductID)"""
