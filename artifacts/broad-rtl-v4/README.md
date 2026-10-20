# H96 V4 replay — PASS

[The completed report](report.json) records **10,000 exact continuous steps**
plus RESET replay for core
`a9accd3ae37c2b9a97f76061756acc89bfd643e51e410147028e353932f74885`.
All 96 hidden values and four raw outputs matched at every step, with
**124,046 cycles** per step (2.48092 ms at 50 MHz). Simulation took 144.717 s.
The vectors are the old validation-02 prefix with **zero movement samples**;
this establishes numerical equivalence for that sequence, not motion accuracy.

The staging audit records 46 files byte-identical to V3: the same frozen
10,000 old validation-02 vectors, 43 ROM files, stage manifest and C++ harness.
The published report, freeze, logs and publication audit preserve this run.
Reproduction requires fresh output paths; the completed local directory
cannot be rerun in place.
The runner requires all result paths to be absent, hashes the current core
at runtime, verifies the staged inputs and frozen integer export, and binds
the parent V3 report SHA-256
`75e3b34c53d9107271ffe7519a1972c0f4c79809150ad791c15c9d0877d49a76`.
It requires 10,000 exact steps, RESET replay and V3's 124,046-cycle schedule
before writing a report. Verilator and a C++ compiler must be installed.
Source and parent report hashes are checked again after simulation.

To stage in a fresh checkout, first reproduce the integer study and ensure
`build/broad-rtl-v4/` does not exist:

```sh
.venv/bin/python artifacts/broad-rtl-v4/stage-replay.py --export-dir build/broad-integer-v1/export --vectors-file build/broad-integer-v1/validation/validation/02.hex --output-dir build/broad-rtl-v4
cp artifacts/broad-rtl-v4/replay.cpp build/broad-rtl-v4/replay.cpp
cp artifacts/broad-rtl-v4/run-replay.py build/broad-rtl-v4/run-replay.py
.venv/bin/python build/broad-rtl-v4/run-replay.py
```

Simulation alone does not establish physical correctness or FPGA timing.
The separate motion fixture in `artifacts/broad-validation-motion-v1/`
contains 1,000 old validation samples starting at index 11,449, initialized
from zero state. It is for numerical USB equivalence, with no orientation
accuracy claim and no deployment receipt implied.
