# Historical H96 mapped-logic diagnostics

These frozen diagnostics describe core SHA-256
`7ed85265a63e07bc327daa652b4067f5e7f1b72683f3296470a4605c643a07d8`,
before the subsequent bit-exact rounding rewrite. The
[original build report](../broad-capacity-v1/final/report.json) binds that
source to bitstream `a034a0563d656b1d64cea44e289beb5be7341a1394361f876c40e22eea924b4c`.
Its 92 software tests and RTL replay passed, but physical equivalence failed.
These reports must not be treated as verification of later RTL or builds.

Both the [post-synthesis report](report.json) and the
[post-route logical-cell report](post-route/report.json) passed zero-delay
simulation of UART RESET followed by the first old validation-02 STEP.
All 96 hidden values, four raw outputs, CRC, sequence and 124,046 cycles
matched Golden. Independent checks decoded the ROM INIT values; all 48 RAM
configurations and INIT parameters were unchanged after routing. The ALU
audit also checked 128 optimized truth-table cases.

The corresponding [physical diagnostic](physical-validation-diagnostic.json)
retains failures over 16 continuous steps. On its first step, hidden lane 75
was **949 physically versus 437 expected and in both simulations**. Passing
zero-delay simulation does not establish correctness of the packed physical
configuration or identify a timing cause. Routing delays, clock skew, PIPs,
analog behavior and bitstream reconstruction are outside these models.
The post-route adapter supports only the observed cells and configurations.
Its nextpnr source snapshot has not been independently matched to the actual
installed tool revision.

## Contents and integrity

Original reports, INIT/adapter audits, command records, compile/write/replay
logs, two UART requests and authored diagnostic models/harness/scripts are
copied byte for byte. `post-route/source-list.json` retains primary-source
URLs and Git blob identities; reports retain hashes of the original vendor
models and downloaded snapshots. [publication-audit.json](publication-audit.json)
records every copy and verification of all 35 original report file bindings.
Large netlists, adapted JSON, binaries, object files, vendor-derived
`cells_logic.v` and downloaded external source files are omitted.

## Reproduction dependencies

Run from the repository root with the original frozen `build/broad-fpga-v1/`
`synth.json` and `routed.json` matching the report hashes, the frozen integer
export, Python environment, installed Yosys/Verilator and a C++ compiler.
Rebuilding from current RTL creates a different experiment. Restore these
published files to their original `build/broad-post-synth-v1/` paths and the
physical diagnostic to its original path shown in `publication-audit.json`,
using fresh directories. Scripts and the C++ harness intentionally retain
those paths. Follow [commands.txt](commands.txt), then
[post-route/commands.txt](post-route/commands.txt); generated files and logs
require fresh output paths. The reports are the recorded results, not scripts
to rerun in place over existing evidence.

The command records explain how the installed Yosys `cells_sim.v` was reduced
to logic primitives, with the restricted INIT-based ROM model supplied here.
External references are the official [Yosys Gowin models](https://github.com/YosysHQ/yosys/tree/main/techlibs/gowin)
and nextpnr [pack.cc](https://github.com/YosysHQ/nextpnr/blob/main/himbaechel/uarch/gowin/pack.cc)
and [pack_luts.cc](https://github.com/YosysHQ/nextpnr/blob/main/himbaechel/uarch/gowin/pack_luts.cc).
Those branch URLs can change; compare retrieved bytes with the frozen report
hashes before using them. This package performs no USB operations and contains
no new heldout dataset access.
