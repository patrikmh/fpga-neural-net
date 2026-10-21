# Experimental absolute-motion forecast — accuracy rejected

This package is an engineering demonstration of the existing 9-input/H96/4-output
FPGA predicting absolute orientation 50 ms ahead. It **does not meet the accuracy
goal** and does not replace the original orientation deployment.

Validation uses 25 whole simulated trajectories and 50,025 scored samples after
32 warmup samples per trajectory:

| Method | Overall RMSE |
| --- | ---: |
| Floating-point neural network | 10.1039° |
| Exported integer neural network | 10.1329° |
| Same-input constant-velocity baseline | 0.9892° |

The integer network also fails the 5° family goal for spin, smooth acceleration
and changing-axis motion. Input, state and raw-output clipping counts are zero.
The full tails, cold starts and baseline comparisons are in
`validation-integer.json`. That report's prediction-file paths refer to retained
local study outputs; the large per-frame arrays are not included in this package.
No confirmation data has been generated or opened.

Engineering checks passed:

- A normal source-bound synthesis/place/route/pack at the actual 50 MHz clock:
  50.5561 MHz reported timing limit, 48/56 BSRAMs and 11,361 LUT4s.
- A temporary SRAM load followed by 1,000 reset-aligned changing-axis samples:
  every one of 96 hidden states and four raw outputs matched the integer model;
  sequence, CRC and 124,046 compute cycles were checked on every reply.
- Actual USB → HTTP → browser with keyboard-controlled motion, delayed matched
  scoring, run, pause, reset and disconnect. The final visual run used 47 verified
  samples, including cold start and abrupt changes. Its 28.7° model RMSE versus
  4.7° baseline RMSE is an interactive diagnostic, not a held-out score.
- Mobile rendering at 390 px and reduced motion. The native fixture separately
  exercises the no-WebGL text fallback.

The original SRAM model and orientation viewer were restored after the test.
Flash was not written. USB throughput and model sampling time are unchanged;
50 ms is model time, not a real-time wall-clock prediction claim.

To repeat the experiment, stop the existing viewer and ensure the USB interface
is free. Program this package's `broad.fs` into SRAM, then start the viewer with
this package's matching `export` directory:

```sh
.tools/tang-toolchain/oss-cad-suite/bin/openFPGALoader \
  --board tangprimer25k --usb-serial-num 2023112818 \
  --write-sram artifacts/motion-prediction-v1/experimental-s47/broad.fs
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.tools/tang-toolchain/oss-cad-suite/lib" \
  .venv/bin/python -m host.motion_demo --usb-serial 2023112818 --port 8766 \
  --export-dir artifacts/motion-prediction-v1/experimental-s47/export
```

Open <http://127.0.0.1:8766>. After disconnecting and stopping that viewer, restore
`artifacts/broad-h96-deployment-v2/build/broad.fs` with the same SRAM-only loader
command and restart `host.orientation_demo`. See
[the motion guide](../../../docs/motion-prediction.md) for timing and scoring semantics.
