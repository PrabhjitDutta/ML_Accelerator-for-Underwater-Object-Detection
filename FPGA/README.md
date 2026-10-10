# FPGA build outputs — YOLO26s kernel and attention block on ZCU102

These are the outputs of the **final** hardware build: the `y26_conv_top` convolution kernel and the
`y26_attn_top` attention block, synthesized with Vitis HLS and implemented together with Vivado 2026.1 for the ZCU102 board (`xczu9eg-ffvb1156-2-e`) at a **4.000 ns (250 MHz)**
clock. The folders follow the build order. Each one holds what the next step, or the board run, needs.

| Folder | Stage | What is in it |
|---|---|---|
| `1_hls_synthesis/` | C++ to RTL (Vitis HLS) | `generated_rtl_verilog/`: the Verilog HLS generated for the kernel (top module `y26_conv_top.v`; `.dat` files initialize the on-chip ROM/RAM tables, and the `_ip.tcl` file creates the one Xilinx floating-point IP core the RTL uses). `kernel_ip/`: the same RTL packaged as the IP that Vivado's block design instantiates. `reports/`: csynth timing, latency and resource estimates. `hls_build_flags.txt`: the exact `-D` flag set used. The attention block has the same three items: `attention_generated_rtl_verilog/`, `attention_ip/` and `reports/attention_csynth_*.rpt` (it takes no `-D` flags). |
| `2_cosim_latency/` | C/RTL co-simulation | Cycles for each of the 100 kernel convs of one frame. This is where the latency figure comes from. `attention_block_cosim.rpt`: the attention block's co-simulation at the network's real shape. |
| `3_place_and_route/` | Vivado implementation | Routed timing summary, placed utilization, vectorless power, route status, DRC and methodology checks, clock utilization; `block_design/` (the Vivado block design connecting the kernel and the attention block to the ARM processing system, `.bd`, plus its generated top-level Verilog wrapper). `scripts/`: the Tcl that built the block design and ran implementation, and a timing-closure re-implementation recipe (not used for these results, see Provenance). |
| `4_bitstream/` | Bitstream + hardware handoff | `.bit` (FPGA configuration), `.hwh` (register and address map that PYNQ reads; it must keep the same base name as the `.bit`), and `.xsa` (hardware platform for Vitis / PetaLinux). For a Linux image without PYNQ (e.g. Ubuntu for ZCU102), both made from the `.xsa`/`.bit`: `.bit.bin` (the bitstream in the form the Linux FPGA manager loads) and `.dtbo` (device-tree overlay: sets the fabric clock to 250 MHz, configures the PS-PL port widths, declares the kernel at `0x80000000` and the attention block at `0x80010000`), with their sources `.bif` and `.dtso`. |
| `5_board_files/` | Board deployment package | Only what the board needs: bitstream (PYNQ and plain-Linux forms), host program source, weights, a first-run check and a frame converter. See `5_board_files/README.md`. |

## Key numbers

| | |
|---|---|
| Kernel latency (C/RTL co-sim, one 640×640 frame) | 17,308,492 cycles = **69.23 ms = 14.44 FPS** at 250 MHz, kernel only (host CPU work not included) |
| Kernel latency with the host's Img2Col (the host program's default, same bitstream) | The first conv runs as a 1×1 conv over 27 gathered planes: 1,324,208 → 503,048 cycles (co-sim at two input heights, extrapolated to the full frame), so 16,487,332 cycles = **65.95 ms = 15.16 FPS** |
| Attention block latency (C/RTL co-sim, 4 heads, 400 positions) | 582,460 to 584,060 cycles per call = 2.33 ms; two calls per frame = **4.67 ms** |
| Timing after route | WNS **+0.022 ns**, 0 failing endpoints |
| Resources (placed) | 84,868 LUT (31.0%), 66,907 FF (12.2%), 243.5 BRAM tiles (26.7%), 1,056 DSP (41.9%) |
| On the board (Ubuntu, 20 test frames, one frame at a time) | **205.9 ms per frame = 4.86 FPS** end to end, host CPU work included; the attention block takes 4.81 ms of it |
| Power on the board (sum of the 18 on-board rail sensors, FPGA and processor rails) | **4.32 W** while running, 3.50 W with the bitstream loaded and idle; 0.99 J per frame |
| Power report | 8.764 W total on-chip. This is Vivado's **vectorless** estimate for the whole chip, ARM processing system included. It is not a measured value and not the two blocks alone. |

The co-sim CSV's `source` column shows 99 of the 100 convs measured by co-simulation. One,
`23.one2one_cv3.0.1.0`, is interpolated because its co-simulation runs out of memory.

## Provenance

`SHA256SUMS.txt` lists each file's checksum and the path it was copied from. The `.bit` is byte-identical
to the one Vivado wrote for the routed run (`impl_1/bd_wrapper.bit`).

Checks in the reports: all nets routed with 0 routing errors; DRC 0 errors (1,890 warnings, all DSP
input/output pipelining, and 34 advisories); methodology checks 0.

The scripts are as run: their default paths point at the build machine (`C:/hls_y26/...`) and are passed as
arguments. `create_block_design_and_implement.tcl` takes `<validate|build> <MHz> <outdir> <kernel IP repo> attn <attention IP repo>`;
its `impl_1` run is what is shipped here. `reimplement_timing_recipe.tcl` adds an `impl_2` run; for this design it
reached WNS +0.020 ns, no better than `impl_1`, so it was not used.

The Linux load files were made with the Vitis 2026.1 tools, from `4_bitstream/`:
```
bootgen -arch zynqmp -process_bitstream bin -image yolo26s_zcu102_attn.bif -w     # -> yolo26s_zcu102_attn.bit.bin
sdtgen  # tcl: sdtgen set_dt_param -xsa yolo26s_zcu102_attn.xsa -dir sdt -board_dts zcu102-rev1.0; sdtgen generate_sdt
lopper -f --enhanced -O ov sdt/system-top.dts linux.dts -- xlnx_overlay_dt full sdt/pl.dtsi --firmware-name=yolo26s_zcu102_attn.bit.bin
dtc -@ -I dts -O dtb -o yolo26s_zcu102_attn.dtbo yolo26s_zcu102_attn.dtso        # yolo26s_zcu102_attn.dtso = ov/pl.dtso
```
The `.bit.bin` is the `.bit` without its 210-byte header, byte-swapped per 32-bit word. The overlay's
`firmware-name` must match the `.bit.bin` file name.

Not included:
- the routed checkpoint (`.dcp`, 103 MB, over GitHub's 100 MB file limit);
- the full Vivado and HLS projects (hundreds of MB of regenerable intermediates);
- the earlier kernel-only build (in this repo's git history);
- intermediate and superseded builds, including an untested variant with the kernel's activation ports on the
  cache-coherent HPC ports (still at `C:/hls_y26/bringup/ya128_250_coh_h`).
