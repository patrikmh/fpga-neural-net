"""Strict parsing of independently CRC-framed BROAD replies."""

import struct
import unittest

from host.broad_protocol import parse_response


def independent_frame(payload):
    crc=0xffff
    for byte in payload:
        crc^=byte<<8
        for _ in range(8):
            crc=((crc<<1)^0x1021)&0xffff if crc&0x8000 else (crc<<1)&0xffff
    return payload+struct.pack("<H",crc)


class BroadProtocolRejectionTests(unittest.TestCase):
    def test_malformed_length_magic_crc_or_nonzero_status_is_rejected(self):
        payload=struct.pack("<BBH3h4hI",0xa6,0,123,-1,2,-3,4,-5,6,-7,542)
        valid=independent_frame(payload)
        self.assertEqual(parse_response(valid,hidden_size=3),(123,(-1,2,-3),(4,-5,6,-7),542))
        magic=bytearray(payload)
        magic[0]=0
        status=bytearray(payload)
        status[1]=1
        cases={"short":valid[:-1],"long":valid+b"\x00","magic":independent_frame(bytes(magic)),
               "crc":valid[:-1]+bytes([valid[-1]^1]),"status":independent_frame(bytes(status))}
        for cause,reply in cases.items():
            with self.subTest(cause=cause),self.assertRaises(ValueError):
                parse_response(reply,hidden_size=3)


if __name__=="__main__":
    unittest.main()
