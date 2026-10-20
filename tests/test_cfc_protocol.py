"""Host rejects malformed or unsuccessful complete-state replies."""

# Behavior: invalid reply length/header/checksum/status cannot become a result.
import functools
import operator
import struct
import unittest

from host.cfc_protocol import parse_response


def framed(payload):
    return payload+bytes([functools.reduce(operator.xor,payload)])


class CfcProtocolRejectionTests(unittest.TestCase):
    def test_parser_rejects_malformed_length_header_checksum_and_failure_status(self):
        payload=b"\x5a\xa5"+struct.pack("<HBh8hI",123,0,-17,*range(8),1052)
        valid=framed(payload)
        nonzero_status=bytearray(payload)
        nonzero_status[4]=1
        bad_header=bytearray(payload)
        bad_header[0]=0
        cases={"short":valid[:-1],"long":valid+b"\x00",
               "header":framed(bytes(bad_header)),
               "checksum":valid[:-1]+bytes([valid[-1]^1]),
               "status":framed(bytes(nonzero_status))}
        for cause,frame in cases.items():
            with self.subTest(cause=cause),self.assertRaises(ValueError):
                parse_response(frame)


if __name__=="__main__":
    unittest.main()
