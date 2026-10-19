"""Binary frames shared by the host and UART probe."""
import functools
import operator
import struct


def build_request(seq, a, b, addr):
    frame = struct.pack('<2sHhhB', b'\xa5\x5a', seq, a, b, addr)
    return frame + bytes([functools.reduce(operator.xor, frame)])


def parse_response(frame):
    if len(frame) != 9 or frame[:2] != b'\x5a\xa5':
        raise ValueError('Invalid response frame')
    if functools.reduce(operator.xor, frame) != 0:
        raise ValueError('Invalid response checksum')
    return struct.unpack('<Hi', frame[2:8])
