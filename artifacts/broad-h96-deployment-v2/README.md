# H96 implementation repair and deployment evidence

This is an equivalent RTL implementation repair after the
[original study selection](../broad-h96-final-v1/selection.json) was frozen.
That selection and its [CPU results](../broad-h96-final-v1/report.json) remain
immutable. Checkpoint, normalization, numeric format and accuracy evaluation
were not retuned. The deployment runner binds the repaired build to those
parent hashes and permits only the recorded core source change.
The unchanged CPU mathematics did not require another heldout accuracy run;
this is not a new accuracy-based model selection.
The [deployment receipt](receipt.json) binds the published evidence to the
source commit and the immutable study parent.

The original loaded design failed physical equivalence: the first trial-04
step differed by +512 at hidden lane 9. Separate validation-02 diagnostics
also retained failures. [Historical evidence](history/original-physical-diagnostic/manifest.json)
belongs to that original core and bitstream, not the repaired design. Those
observations did not establish the physical root cause.

The repair replaces magnitude-based rounding with quotient/remainder
rounding: arithmetic quotient is rounded upward at a half tie for positive
values and strictly above a half for negative values, preserving half-away-
from-zero results. The blend uses the same quotient and rounding increment,
combining three addends through XOR/majority carry-save arithmetic before
the final addition and saturation. Model mathematics and the cycle schedule
are unchanged. [V4 simulation](../broad-rtl-v4/report.json) matched all 96
hidden values and four raw outputs for 10,000 continuous old validation
steps plus RESET replay, at 124,046 cycles per step.

## Build and verification status

The [seed-3 build](build/report.json) achieved **50.556114 MHz Fmax** against
the strict 50 MHz requirement. Actual operating clock remains **50 MHz**.
It uses **11,361 LUT4, 4,218 DFF and 48/56 BSRAM**. Its bitstream SHA-256 is
`3564b2c2ac90d1a38b2d505df4df7945ac665dc17db042ba477274322e463988`.
The pinned capacity regression reproduced the identical bitstream and passed
in 79.390 s. [History](history.json) retains failed implementation variants
and seeds 1/2; seed 4 was not run. Builder default seed 1 is unchanged, so
reproducing this passing build requires explicit seed 3.

The [quick physical diagnostic](quick-physical/report.json) passed RESET
and 16 continuous old validation-prefix steps on the repaired bitstream.
The [full physical verification](physical/report.json) also **passed all
15,000 steps**, with six RESETs: 10,000 old validation-prefix steps, 1,000
old validation movement-window steps starting with RESET at sample 11,449,
and four separate RESET + 1,000 new test-prefix sequences (04/08/12/31).
Every step matched all 96 hidden values and four raw outputs, with valid
CRC, sequence and the expected 124,046 cycles. Core compute time is
**2.48092 ms at the actual 50 MHz clock**; USB round-trip time is measured
separately in the published sequence reports. The validation prefix and all
new test prefixes contain no movement samples. This establishes numerical
equivalence over those sequences, not physical full-trial orientation accuracy.

[Software verification](verification.json) passed **94 unique tests** in two
separate runs: one capacity regression and 93 remaining tests
([log](remaining-tests.log)). Compileall and strict H3/H96 Verilator lint
also passed. Repeated focused tests are not added to that unique total.

The earlier scalar 600-second USB endurance gate remains open. CPU results
also show better inclination accuracy from the causal gyro/accelerometer
baseline; this implementation repair makes no new accuracy claim.

## Reproduce

Use the repository root and a fresh build directory with installed FPGA tools:

```sh
PATH="$PWD/.tools/tang-toolchain/oss-cad-suite/bin:$PATH" .venv/bin/python scripts/build_broad.py --export-dir artifacts/broad-integer-v1/export --build-dir build/broad-h96-seed3-reproduction --placer heap --seed 3 --heap-beta 0.9 --stage-timeout 1200
```

The published build can be used directly for SRAM-only loading and verification
on USB serial `2023112818`. The runner requires the frozen study, its vector
packages, the old validation-prefix package at `build/broad-usb-vectors-v1`,
the published movement-window package, and current source hashes matching
the repaired build. Output directories must not already exist:

```sh
caffeinate -disu .venv/bin/python artifacts/broad-h96-deployment-v2/run-physical.py --build-dir artifacts/broad-h96-deployment-v2/build --rtl-report artifacts/broad-rtl-v4/report.json --output-dir build/broad-h96-reproduce-physical
```

Add `--quick` and use another fresh output directory for the 16-step diagnostic.
The runner records separate core compute time and USB round-trip measurements
through the verifier. It writes a final physical report only after every
requested sequence passes; a started run is not a successful proof.
