# UART and arithmetic probe

This probe verifies the computer-to-FPGA path before adding the neural model.
An input frame supplies two signed integers and a ROM address. The FPGA computes
their product plus the signed ROM entry, writes that value into RAM, reads it
back synchronously, and returns the result. Simulation must exercise the actual
UART pins; synthesis must separately prove that ROM and RAM use block memory.

## Serial link

115200 baud, eight data bits, no parity, one stop bit. The FPGA clock is 50 MHz.
Only one request is in flight. All multi-byte fields are little-endian.

## Request: ten bytes

| Offset | Bytes | Meaning |
| --- | --- | --- |
| 0 | 2 | Magic `A5 5A` |
| 2 | 2 | Unsigned sequence number |
| 4 | 2 | Signed operand `a` |
| 6 | 2 | Signed operand `b` |
| 8 | 1 | ROM/RAM address, 0–255 |
| 9 | 1 | XOR of bytes 0–8 |

## Response: nine bytes

| Offset | Bytes | Meaning |
| --- | --- | --- |
| 0 | 2 | Magic `5A A5` |
| 2 | 2 | Echoed sequence number |
| 4 | 4 | Signed result, read back from RAM |
| 8 | 1 | XOR of bytes 0–7 |

The deterministic signed 16-bit ROM contents are:

```text
rom[i] = ((i * 4051 + 17389) & 65535) - 32768
result = a * b + rom[address]
```

The product plus ROM value fits in signed 32 bits for every allowed operand.
The XOR check detects many transmission errors but does not detect every
possible corruption. Sequence checking and an exact arithmetic comparison are
also required during the physical-board verification run.

## Reset and startup

The physical reset input is active high. It uses the official example's H11 pin
and pulldown setting. The design must also initialize itself after SRAM loading;
RAM contents need not be reset because every accepted request writes before reading.

Do not send a second request until the complete response to the first has arrived.
A malformed checksum must not update the arithmetic state. Reset discards a
partial frame. Recovery and numeric extremes are tested after the first tracer.

## First tracer

Sequence `0x1234`, `a=-123`, `b=17`, address `17` gives ROM value `-12048`
and expected result `-14139`. The expected value is calculated in Python from
the documented rule, independently of the DUT's committed ROM initialization file.
