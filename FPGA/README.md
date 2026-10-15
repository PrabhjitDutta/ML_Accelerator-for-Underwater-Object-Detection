# FPGA build outputs — YOLO26s kernel, attention block and ops block on ZCU102

These are the outputs of the **final** hardware build: the `y26_conv_top` convolution kernel and the
`y26_attn_top` attention block and the `y26_ops_top` ops block (residual adds, 2x upsamples and map copies on
8-bit codes), synthesized with Vitis HLS and implemented together with Vivado 2026.1 for the ZCU102 board (`xczu9eg-ffvb1156-2-e`) at a **4.000 ns (250 MHz)**
clock. The folders follow the build order. Each one holds what the next step, or the board run, needs.

| Folder | Stage | What is in it |
|---|---|---|
| `1_hls_synthesis/` | C++ to RTL (Vitis HLS) | `generated_rtl_verilog/`: the Verilog HLS generated for the kernel (top module `y26_conv_top.v`; `.dat` files initialize the on-chip ROM/RAM tables, and the `_ip.tcl` file creates the one Xilinx floating-point IP core the RTL uses). `kernel_ip/`: the same RTL packaged as the IP that Vivado's block design instantiates. `reports/`: csynth timing, latency and resource estimates. `hls_build_flags.txt`: the exact `-D` flag set used. The attention block has the same three items: `attention_generated_rtl_verilog/`, `attention_ip/` and `reports/attention_csynth_*.rpt` (it takes no `-D` flags); so has the ops block: `ops_generated_rtl_verilog/`, `ops_ip/` and `reports/ops_csynth_*.rpt`. |
| `2_cosim_latency/` | C/RTL co-simulation | Cycles for each of the 100 kernel convs of one frame, measured on the previous kernel build (without the code port). This is where the latency figure comes from. `direct_codes_kernel_4_convs.csv`: four convs co-simulated again on the shipped kernel (+0.20% to +0.44% cycles). `attention_block_cosim.rpt`: the attention block's co-simulation at the network's real shape. `ops_block_cosim.rpt`: the ops block's co-simulation, 12 cases (pass). |
| `3_place_and_route/` | Vivado implementation | Routed timing summary, placed utilization, vectorless power, route status, DRC and methodology checks, clock utilization; `block_design/` (the Vivado block design connecting the kernel, the attention block and the ops block to the ARM processing system, `.bd`, plus its generated top-level Verilog wrapper). `scripts/`: the Tcl that built the block design and ran implementation, and a timing-closure re-implementation recipe (not used for these results, see Provenance). |
| `4_bitstream/` | Bitstream + hardware handoff | `.bit` (FPGA configuration), `.hwh` (register and address map that PYNQ reads; it must keep the same base name as the `.bit`), and `.xsa` (hardware platform for Vitis / PetaLinux). For a Linux image without PYNQ (e.g. Ubuntu for ZCU102), both made from the `.xsa`/`.bit`: `.bit.bin` (the bitstream in the form the Linux FPGA manager loads) and `.dtbo` (device-tree overlay: sets the fabric clock to 250 MHz, configures the PS-PL port widths, declares the kernel at `0x80000000` and the attention block at `0x80010000` and the ops block at `0x80030000`, and enables the PS port of the kernel's code output), with their sources `.bif` and `.dtso`. |
| `5_board_files/` | Board deployment package | Only what the board needs: bitstream (PYNQ and plain-Linux forms), host program source, weights, a first-run check and a frame converter. See `5_board_files/README.md`. |

## Key numbers

| | |
|---|---|
| Kernel latency (C/RTL co-sim, one 640×640 frame) | 17,308,492 cycles = **69.23 ms = 14.44 FPS** at 250 MHz, kernel only (host CPU work not included). Measured on the previous kernel build; the shipped kernel also writes each output as 8-bit codes and takes 0.20% to 0.44% more cycles on the four convs co-simulated again. |
| Kernel latency with the host's Img2Col (the host program's default, same bitstream) | The first conv runs as a 1×1 conv over 27 gathered planes: 1,324,208 → 503,048 cycles (co-sim at two input heights, extrapolated to the full frame), so 16,487,332 cycles = **65.95 ms = 15.16 FPS** |
| Attention block latency (C/RTL co-sim, 4 heads, 400 positions) | 582,460 to 584,060 cycles per call = 2.33 ms; two calls per frame = **4.67 ms** |
| Timing after route | WNS **+0.035 ns**, 0 failing endpoints |
| Resources (placed) | 101,204 LUT (36.9%), 91,963 FF (16.8%), 256.5 BRAM tiles (28.1%), 1,113 DSP (44.2%) |
| On the board (Ubuntu, 20 test frames, one frame at a time, `Y26_OPS=1`) | **145.0 ms per frame = 6.90 FPS** end to end, host CPU work included. With four frames overlapped (`Y26_INFLIGHT=4`): **9.06 FPS** |
| Power on the board (sum of the 18 on-board rail sensors, FPGA and processor rails; 400 frames, logger running) | **4.45 W** one frame at a time (0.71 J per frame), 4.94 W with four overlapped (0.59 J per frame). Idle power not measured for this bitstream (3.55 W on the previous one) |
| Power report | 7.939 W total on-chip. This is Vivado's **vectorless** estimate for the whole chip, ARM processing system included. It is not a measured value and not the three blocks alone. |

The co-sim CSV's `source` column shows 99 of the 100 convs measured by co-simulation. One,
`23.one2one_cv3.0.1.0`, is interpolated because its co-simulation runs out of memory.

## Provenance

`SHA256SUMS.txt` lists each file's checksum and the path it was copied from. The `.bit` is byte-identical
to the one Vivado wrote for the routed run (`impl_1/bd_wrapper.bit`).

Checks in the reports: all nets routed with 0 routing errors; DRC 0 errors (1,986 warnings, all DSP
input/output pipelining, and 36 advisories); methodology checks 0.

The scripts are as run: their default paths point at the build machine (`C:/hls_y26/...`) and are passed as
arguments. `create_block_design_and_implement.tcl` takes `<validate|build> <MHz> <outdir> <kernel IP repo> attn <attention IP repo>`;
it finds the kernel's code port by itself and connects it to the HPC1 port. With the environment variable
`Y26_OPS_IP=<ops block IP repo>` it adds the ops block (control at `0x80030000`, its two masters on the attention
block's HPC0 port); `Y26_TWO=1` (a second kernel) is not part of this build. Its `impl_1` run is what is shipped here.
`reimplement_timing_recipe.tcl` adds an `impl_2` run; `impl_1` met timing, so it was not used.

The Linux load files were made with the Vitis 2026.1 tools, from `4_bitstream/`:
```
bootgen -arch zynqmp -process_bitstream bin -image yolo26s_zcu102_ops.bif -w     # -> yolo26s_zcu102_ops.bit.bin
sdtgen  # tcl: sdtgen set_dt_param -xsa yolo26s_zcu102_ops.xsa -dir sdt -board_dts zcu102-rev1.0; sdtgen generate_sdt
lopper -f --enhanced -O ov sdt/system-top.dts linux.dts -- xlnx_overlay_dt full sdt/pl.dtsi --firmware-name=yolo26s_zcu102_ops.bit.bin
dtc -@ -I dts -O dtb -o yolo26s_zcu102_ops.dtbo yolo26s_zcu102_ops.dtso        # yolo26s_zcu102_ops.dtso = ov/pl.dtso
```
The `.bit.bin` is the `.bit` without its 210-byte header, byte-swapped per 32-bit word. The overlay's
`firmware-name` must match the `.bit.bin` file name.

Not included:
- the routed checkpoint (`.dcp`, 103 MB, over GitHub's 100 MB file limit);
- the full Vivado and HLS projects (hundreds of MB of regenerable intermediates);
- the earlier builds: kernel only, kernel plus attention without the code port, and kernel plus attention without
  the ops block (`yolo26s_zcu102_yc`, 191.7 ms per frame) (in this repo's git history);
- intermediate and superseded builds, including an untested variant with the kernel's activation ports on the
  cache-coherent HPC ports (still at `C:/hls_y26/bringup/ya128_250_coh_h`).
