# FPGA build outputs — YOLO26s kernel on ZCU102

These are the outputs of the **final** hardware build: the `y26_conv_top` convolution kernel, synthesized with
Vitis HLS and implemented with Vivado 2026.1 for the ZCU102 board (`xczu9eg-ffvb1156-2-e`) at a **4.000 ns (250 MHz)**
clock. The folders follow the build order. Each one holds what the next step, or the board run, needs.

| Folder | Stage | What is in it |
|---|---|---|
| `1_hls_synthesis/` | C++ to RTL (Vitis HLS) | `generated_rtl_verilog/`: the Verilog HLS generated for the kernel (top module `y26_conv_top.v`; `.dat` files initialize the on-chip ROM/RAM tables, and the `_ip.tcl` file creates the one Xilinx floating-point IP core the RTL uses). `kernel_ip/`: the same RTL packaged as the IP that Vivado's block design instantiates. `reports/`: csynth timing, latency and resource estimates. `hls_build_flags.txt`: the exact `-D` flag set used. |
| `2_cosim_latency/` | C/RTL co-simulation | Cycles for each of the 100 kernel convs of one frame. This is where the latency figure comes from. |
| `3_place_and_route/` | Vivado implementation | Routed timing summary, placed utilization, vectorless power, route status, DRC and methodology checks, clock utilization; `block_design/` (the Vivado block design connecting the kernel to the ARM processing system, `.bd`, plus its generated top-level Verilog wrapper). `routed_checkpoint/`: the fully placed and routed design (`.dcp`, 78 MB); open it in Vivado to re-run timing or power analysis without re-implementing. `scripts/`: the Tcl that built the block design and ran implementation, and the timing-closure re-implementation recipe that produced these results. |
| `4_bitstream/` | Bitstream + hardware handoff | `.bit` (FPGA configuration), `.hwh` (register and address map that PYNQ reads; it must keep the same base name as the `.bit`), and `.xsa` (hardware platform for Vitis / PetaLinux). |
| `final/` | Board deployment package | Only what the board needs: bitstream, host program source, weights, a first-run check and a frame converter. See `final/README.md`. |

## Key numbers

| | |
|---|---|
| Kernel latency (C/RTL co-sim, one 640×640 frame) | 17,308,492 cycles = **69.23 ms = 14.44 FPS** at 250 MHz, kernel only (host CPU work not included) |
| Kernel latency with the host's Img2Col (the host program's default, same bitstream) | The first conv runs as a 1×1 conv over 27 gathered planes: 1,324,208 → 503,048 cycles (co-sim at two input heights, extrapolated to the full frame), so 16,487,332 cycles = **65.95 ms = 15.16 FPS** |
| Timing after route | WNS **+0.023 ns**, 0 failing endpoints |
| Resources (placed) | 62,926 LUT (23.0%), 45,589 FF (8.3%), 212 BRAM tiles (23.3%), 717 DSP (28.5%) |
| Power report | 6.713 W total on-chip. This is Vivado's **vectorless** estimate for the whole chip, ARM processing system included. It is not a measured value and not the kernel alone. |

The co-sim CSV's `source` column shows 99 of the 100 convs measured by co-simulation. One,
`23.one2one_cv3.0.1.0`, is interpolated because its co-simulation runs out of memory.

## Provenance

`SHA256SUMS.txt` lists each file's checksum and the path it was copied from. The `.bit` is byte-identical
to the one Vivado wrote for the routed run (`impl_2/bd_wrapper.bit`).

Checks in the reports: all nets routed with 0 routing errors; DRC 0 errors (315 warnings, all DSP
input/output pipelining advisories); methodology checks 0.

The scripts are as run: their default paths point at the build machine (`C:/hls_y26/...`) and are passed as
arguments. `create_block_design_and_implement.tcl` takes `<validate|build> <MHz> <outdir> <kernel IP repo> [coh]`;
`reimplement_timing_recipe.tcl` then adds the `impl_2` run whose results are shipped here.

Not included:
- the full Vivado and HLS projects (hundreds of MB of regenerable intermediates);
- intermediate and superseded builds, including an untested variant with the kernel's activation ports on the
  cache-coherent HPC ports (still at `C:/hls_y26/bringup/ya128_250_coh_h`).
