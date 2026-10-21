# Motion prediction on the existing H96 core

This experiment changes the training target from the current orientation to the
absolute body-to-ENU orientation **50 ms ahead**. It retains the existing nine
sensor inputs, 96 recurrent states, four quaternion outputs, fixed-point core,
and full-state USB protocol. It requires a newly trained motion checkpoint;
renaming the original orientation checkpoint does not make it a forecaster.

The first two full training runs failed the accuracy goal: their integer
validation RMSE was 11.71° and 10.13°. The sensor-only constant-velocity baseline
was approximately 0.99° on the same effective inputs. These results do not show
a benefit from the neural network. See the [frozen validation record](../artifacts/motion-prediction-v1/initial-validation-v1.json).
A separately declared continuous-quaternion experiment tests signed raw-output
supervision and longer history windows. No candidate is qualified by this guide.

## What the app compares

The main cyan object is the latest processed simulated pose. The main magenta
object is the chip's latest forecast for a later time; these two objects have
different timestamps. Their separation is not an accuracy measurement.

After a forecast's target time has elapsed, the server compares its original
issued quaternion against the simulated pose at that exact target time. The
50 ms horizon lies 14 + 2/7 samples ahead, so the truth is interpolated between
the two enclosing verified samples. The comparison panel uses this older
forecast and matching truth, never the latest forecast against the current pose.

The panel also shows hold and constant-body-angular-velocity baselines. These
are explicit CPU comparisons using the same quantized sensor information as
the model. They do not replace any FPGA output. Rolling RMSE uses the same
latest 96 matured forecasts for all three methods. The first score arrives on
step 16. Reset and disconnect clear pending forecasts and rolling scores.

Every displayed chip prediction still comes from a verified USB reply. The host
checks all 96 states, four raw outputs, sequence, CRC, and 124,046 compute cycles
against the integer reference. A failed transaction cannot advance forecast
scoring. Normalizing the received quaternion on the CPU is the existing output
decoding operation.

## Domain and time

This experiment uses an ideal virtual accelerometer, gyroscope and magnetometer,
with a known identity pose at reset. It is not a real-IMU qualification. Data
contains holds, spins up to 6 rad/s, smooth acceleration, changing axes and
reversals, split by whole trajectories. No future sensor values enter the model.

The model advances **3.5 ms per accepted sample**. Fifty milliseconds refers to
model time, not wall time: the existing USB debug transport runs more slowly.
The forecast cannot know an arbitrary future mouse command. Smooth display
interpolation and animated network pulses create no additional sensor samples.

## Reproduce the software path

Use the project's locked `.venv`. The training recipe and dataset sources are
hashed into the checkpoint; keep them unchanged when evaluating an existing
checkpoint. Use new, empty output directories for each command.

```sh
.venv/bin/python -m model.motion_generate --help
.venv/bin/python -m model.motion_train --help
.venv/bin/python -m model.motion_train_lifted --help
.venv/bin/python -m model.motion_evaluate --help
```

For a selected checkpoint, export and evaluate its actual integer recurrence:

```sh
.venv/bin/python -m model.broad_export_integer \
  --checkpoint PATH/checkpoint.pt --output-dir build/my-motion-export
.venv/bin/python -m model.motion_evaluate \
  --dataset-dir PATH/identity-data --export-dir build/my-motion-export \
  --split validation --output build/my-motion-validation/report.json
```

Checkpoint selection uses validation only. The reserved confirmation split must
remain unopened until the selected checkpoint and export are frozen. The goal
is under 5° RMSE overall and in each motion family, with zero input and raw-output
clipping; report tails, cold starts and paired baseline scores as well.

## Start the viewer

The FPGA must already contain the bitstream built from the **same export**.
The viewer does not program the board. An orientation export is rejected before
USB is opened. Keep the original orientation bitstream available for rollback;
program experimental builds into SRAM only.

```sh
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.tools/tang-toolchain/oss-cad-suite/lib" \
  .venv/bin/python -m host.motion_demo --usb-serial 2023112818 \
  --export-dir build/my-motion-export --port 8766
```

Open <http://127.0.0.1:8766>. Connect, rotate with the mouse or keyboard, then
step or run. Pause waits for the in-flight reply. Reset requires the board's
acknowledgment before clearing history. The original
[current-orientation viewer](orientation-demo.md) remains a separate entry point.

The browser integration tests train and export a small real model, then exercise
the native USB fixture, HTTP server and browser. This tests connectivity and
timestamp semantics; it does not establish physical hardware accuracy.
