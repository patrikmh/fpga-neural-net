"""Binary request and complete-state response frames for the CfC core."""

import functools
import operator
import struct


def build_request(seq, x, command=1):
    frame = struct.pack('<2sHBh', b'\xa5\x5a', seq, command, x)
    return frame + bytes([functools.reduce(operator.xor, frame)])


def parse_response(frame):
    if len(frame) != 28 or frame[:2] != b'\x5a\xa5':
        raise ValueError('Invalid CfC response frame')
    if functools.reduce(operator.xor, frame) != 0:
        raise ValueError('Invalid CfC response checksum')
    seq, status, y, *remaining = struct.unpack('<HBh8hI', frame[2:27])
    if status != 0:
        raise ValueError(f'CfC response status {status}')
    return seq, y, tuple(remaining[:8]), remaining[8]
