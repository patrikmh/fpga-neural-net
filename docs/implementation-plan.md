# Approved implementation plan

Approved in the session on 2026-10-19. Interpret the requested "ctc" as CfC,
the closed-form continuous-time model discussed immediately before approval.

## Model and scope

- Train on the computer; use fixed weights for inference on the FPGA.
- One normalized scalar signal, eight recurrent units, one linear output.
- Match the pinned PyTorch `ncps` CfC implementation, `mode="default"`,
  `backbone_layers=0`: 329 parameters including the output projection.
- Predict the next observed sample using the current observation and prior
  recurrent state. Initially use a fixed normalized timestep of one.
- Reuse one arithmetic datapath. Keep old and new hidden-state buffers separate.
- Start numerical evaluation with signed 16-bit weights and states. Freeze all
  scales, rounding, saturation, lookup tables and intermediate widths only after
  calibration and long-sequence validation.
- Use on-chip memories. No external SDRAM, sensors, online training or runtime
  weight loading in the initial acceptance scope.

## Gates

1. Environment: executable tools respond, versions are recorded when available.
2. Probe: build and simulate UART, signed 16x16 multiplication, initialized ROM
   and working RAM. Prove actual block-memory mapping and report logic usage.
   Navigator approved a logic multiplier after the installed Yosys was shown not
   to infer GW5A DSP multiplication. DSP inference is not required for version one.
3. Board: verify revision/identity, load SRAM, match 1,000 request/response pairs.
4. Float model: independently generated training/validation/test trajectories;
   fit normalization on training only. Compare persistence and linear extrapolation.
5. Integer model: exact reference arithmetic, long recurrent sequences, export
   deterministic weights and test vectors.
6. RTL: match every hidden state and output against the integer reference for
   at least 10,000 steps, including reset and numeric boundaries.
7. Acceptance: timing closure, real-board vector comparison, graphical demo
   and separately reported compute latency and UART round-trip latency.

Do not advance FPGA model implementation past a failed hardware/toolchain gate.

## Proposed measurable targets

Targets are experimental acceptance goals, not measured results:

- At least 20% lower test MSE than persistence; also report linear extrapolation.
- Fixed-point MSE <= float MSE + max(0.1 * float MSE, epsilon).
  Select and record epsilon in normalized signal units before examining test data.
- Less than 100 microseconds compute latency per sample.
- At least 100 samples/second over UART for ten minutes without protocol errors.
- Explain the dataflow, numeric format and one complete timestep in the README.

## Board references, not yet validated on this unit

The official UART example selects GW5A-25A, GW5A-LV25MG121NC1/I0, uses a
50 MHz input clock and 115200 baud, with clock E2, reset H11, RX B3 and TX C3.
Match these against the actual board revision before loading a bitstream.

- [Official UART project](https://github.com/sipeed/TangPrimer-25K-example/tree/main/UART)
- [Official pin constraints](https://github.com/sipeed/TangPrimer-25K-example/blob/main/UART/simple_uart/src/top.cst)
- [CfC reference cell](https://github.com/mlech26l/ncps/blob/master/ncps/torch/cfc_cell.py)
- [CfC sequence wrapper](https://github.com/mlech26l/ncps/blob/master/ncps/torch/cfc.py)
- [Apicula board support](https://github.com/YosysHQ/apicula)
- [SRAM loader support](https://trabucayre.github.io/openFPGALoader/compatibility/board.html)

The CfC source, toolchain and board constraints must be pinned to specific
versions/revisions during their implementation gates. Links to moving branches
above document the research sources, not a reproducible build specification.

## Reversibility

Use a new project directory and load only FPGA SRAM. Keep existing board flash
untouched. Record reports and artifacts at each successful gate. Power cycling
removes our temporary SRAM configuration and returns to the board's existing
startup behavior.
