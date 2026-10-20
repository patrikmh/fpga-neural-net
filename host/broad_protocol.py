"""CRC16-CCITT-FALSE frames for the BROAD recurrent core."""

import struct


def crc16(data):
    crc = 0xffff
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ (0x1021 if crc & 0x8000 else 0)) & 0xffff
    return crc


def build_request(seq, inputs, command=1):
    if type(seq) is not int or not 0 <= seq <= 65535 or command not in (1, 2):
        raise ValueError('Invalid BROAD sequence or command')
    if len(inputs) != 9 or any(type(value) is not int or not -32768 <= value <= 32767 for value in inputs):
        raise ValueError('Expected nine signed-16 integer inputs')
    payload = struct.pack('<BBH9h', 0xa6, command, seq, *inputs)
    return payload + struct.pack('<H', crc16(payload))


def parse_response(frame, hidden_size=96):
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError('Invalid hidden size')
    if len(frame) != 18 + 2 * hidden_size or frame[0] != 0xa6:
        raise ValueError('Invalid BROAD response frame')
    if crc16(frame[:-2]) != int.from_bytes(frame[-2:], 'little'):
        raise ValueError('Invalid BROAD response CRC')
    if frame[1] != 0:
        raise ValueError(f'BROAD response status {frame[1]}')
    seq, *values = struct.unpack(f'<H{hidden_size + 4}hI', frame[2:-2])
    return seq, tuple(values[:hidden_size]), tuple(values[hidden_size:-1]), values[-1]
