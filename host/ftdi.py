"""Bounded UART access to interface B of an explicitly named FT2232 device."""

import ctypes
from pathlib import Path
import sys
import time


DEFAULT_LIBRARY = (
    Path(__file__).resolve().parents[1]
    / ".tools/tang-toolchain/oss-cad-suite/lib/libftdi1.2.5.0.dylib"
)


class LibraryVersion(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_int), ("minor", ctypes.c_int), ("micro", ctypes.c_int),
        ("version_str", ctypes.c_char_p), ("snapshot_str", ctypes.c_char_p),
    ]


class ContextPrefix(ctypes.Structure):
    # libftdi 1.5 src/ftdi.h: context's public timeout fields follow two pointers.
    _fields_ = [
        ("usb_ctx", ctypes.c_void_p), ("usb_dev", ctypes.c_void_p),
        ("usb_read_timeout", ctypes.c_int), ("usb_write_timeout", ctypes.c_int),
    ]


class Ftdi:
    def __init__(self, serial_number, library=None, *, requested_baudrate=460800):
        if type(requested_baudrate) is not int or not 0 < requested_baudrate <= 2147483647:
            raise ValueError('Requested baudrate must be a positive signed-32-bit integer')
        self.requested_baudrate = requested_baudrate
        self.serial_number = serial_number
        self.context = None
        self.opened = False
        path = Path(library) if library else DEFAULT_LIBRARY
        try:
            self.library = ctypes.CDLL(str(path))
        except OSError as error:
            raise OSError(
                f"Cannot load libftdi at {path}: {error}. On macOS set "
                f"DYLD_FALLBACK_LIBRARY_PATH={path.parent} before starting Python."
            ) from error
        version_function = self.library.ftdi_get_library_version
        version_function.argtypes = []
        version_function.restype = LibraryVersion
        version = version_function()
        if (version.major, version.minor) != (1, 5):
            raise OSError(f"Unsupported libftdi ABI {version.major}.{version.minor}; expected 1.5")
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        byte_pointer = ctypes.POINTER(ctypes.c_ubyte)
        signatures = {
            "ftdi_new": ([], pointer),
            "ftdi_free": ([pointer], None),
            "ftdi_get_error_string": ([pointer], ctypes.c_char_p),
            "ftdi_usb_close": ([pointer], integer),
            "ftdi_set_interface": ([pointer, integer], integer),
            "ftdi_usb_open_desc": ([pointer, integer, integer, ctypes.c_char_p, ctypes.c_char_p], integer),
            "ftdi_set_bitmode": ([pointer, ctypes.c_ubyte, ctypes.c_ubyte], integer),
            "ftdi_set_baudrate": ([pointer, integer], integer),
            "ftdi_set_line_property": ([pointer, integer, integer, integer], integer),
            "ftdi_setflowctrl": ([pointer, integer], integer),
            "ftdi_set_latency_timer": ([pointer, ctypes.c_ubyte], integer),
            "ftdi_read_data": ([pointer, byte_pointer, integer], integer),
            "ftdi_write_data": ([pointer, byte_pointer, integer], integer),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = result

    def checked(self, name, *arguments):
        result = getattr(self.library, name)(self.context, *arguments)
        if result < 0:
            detail = self.library.ftdi_get_error_string(self.context)
            message = detail.decode(errors="replace") if detail else "unknown libftdi error"
            raise OSError(f"{name} failed ({result}): {message}")
        return result

    def __enter__(self):
        self.context = self.library.ftdi_new()
        if not self.context:
            raise OSError("ftdi_new failed to allocate a context")
        try:
            prefix = ctypes.cast(self.context, ctypes.POINTER(ContextPrefix)).contents
            prefix.usb_read_timeout = 100
            prefix.usb_write_timeout = 100
            self.checked("ftdi_set_interface", 2)
            self.checked("ftdi_usb_open_desc", 0x0403, 0x6010, None, self.serial_number.encode())
            self.opened = True
            self.checked("ftdi_set_bitmode", 0, 0)
            self.checked("ftdi_set_baudrate", self.requested_baudrate)
            self.checked("ftdi_set_line_property", 8, 0, 0)
            self.checked("ftdi_setflowctrl", 0)
            self.checked("ftdi_set_latency_timer", 2)
            self.drain()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exception_type, exception, traceback):
        try:
            if self.opened:
                status = self.library.ftdi_usb_close(self.context)
                if status < 0 and exception_type is None:
                    raise OSError(f"ftdi_usb_close failed ({status})")
        finally:
            if self.context:
                self.library.ftdi_free(self.context)
            self.context = None
            self.opened = False

    def drain(self):
        deadline = time.monotonic() + 1.0
        buffer = (ctypes.c_ubyte * 512)()
        quiet_since = None
        while time.monotonic() < deadline:
            if self.checked("ftdi_read_data", buffer, len(buffer)) == 0:
                now = time.monotonic()
                if quiet_since is None:
                    quiet_since = now
                if now - quiet_since >= 0.020:
                    return
                time.sleep(0.001)
            else:
                quiet_since = None
        raise OSError("libftdi startup drain did not become quiet within one second")

    def write(self, data):
        deadline = time.monotonic() + 1.0
        written = 0
        while written < len(data) and time.monotonic() < deadline:
            remaining = data[written:]
            buffer = (ctypes.c_ubyte * len(remaining)).from_buffer_copy(remaining)
            count = self.checked("ftdi_write_data", buffer, len(buffer))
            written += count
            if count == 0:
                time.sleep(0.001)
        return written

    def read(self, size):
        deadline = time.monotonic() + 1.0
        received = bytearray()
        while len(received) < size and time.monotonic() < deadline:
            buffer = (ctypes.c_ubyte * (size - len(received)))()
            count = self.checked("ftdi_read_data", buffer, len(buffer))
            received.extend(buffer[:count])
            if count == 0:
                time.sleep(0.001)
        return bytes(received)
