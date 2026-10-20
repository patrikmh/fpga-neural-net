# Approved BROAD CPU experiment

Patrik selected BROAD orientation estimation on 2026-10-20. This experiment is
separate from the verified scalar FPGA implementation.

Use nine ordered sensor inputs (accelerometer, gyroscope, magnetometer), an
8-state default CfC cell without a backbone, and four normalized quaternion
outputs in BROAD's wxyz body-to-ENU convention. No reference-derived alignment
or initialization is allowed. Preserve native 2000/7 Hz sampling and every
sample, including gaps in optical reference. Fit channel normalization on
training trials only. Mask invalid optical references before loss computation.

Split whole trials before training:

| Split | Trial IDs |
|---|---|
| Training | 01, 06, 10, 28 |
| Validation | 02, 30 |
| Held-out test | 03, 07, 11, 29 |

This is a small engineering experiment, not a benchmark over all 39 trials.
The previous preview inspected training trials 01, 06 and 28 only.

1. B1: prepare train/validation HDF5 as provenance-bound NPZ files. Conversion
   uses the isolated h5py environment; training keeps the existing environment.
2. B2: train a causal quaternion predictor with a fixed finite budget. Select
   the checkpoint using equal-trial mean full-orientation validation RMSE.
   Save normalization, seed, window/warmup policy, predictions and identities.
3. B3: freeze checkpoint and evaluation specification before reading held-out
   sensor/reference data. Report full orientation, inclination and heading
   errors, scored counts and exclusions per trial and equal-trial aggregates.
   Compare the existing causal accelerometer/gyro baseline for inclination.

TDD validation uses real command-line subprocesses and small real data files,
then the actual pinned BROAD recordings. Each phase is reviewed before the
next. The full existing suite and an independent CLI smoke test conclude work.

An unsuccessful accuracy result remains a valid experiment result. Further
model architecture searches or FPGA changes require a separate decision.
The earlier 600-second USB stability gate remains unresolved independently.

## Predeclared training budget

One seed (7), 50 epochs, 16 sampled batches per epoch, batch size 16,
256-step causal windows and 32 initial warmup steps per window. Adam at
0.003, gradient norm clipped to 1.0; quaternion sign-invariant loss.
Evaluate continuous full validation trials at epoch 1, every 10 epochs and
at the final epoch, resetting hidden state only between trials. Use 32
initial warmup samples and movement/valid-reference masks for scores.
Normalize dt to one native sample. No hyperparameter search on held-out data.
