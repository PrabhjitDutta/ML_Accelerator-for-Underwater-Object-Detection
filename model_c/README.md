# model_c — the C++ side of YOLO26s: reference model, FPGA kernel, board host

Everything here is plain C++ built with `g++` (MSYS2 ucrt64 on Windows, native `g++` on the board). Only the files
in `hls_kernel/` are synthesized to the FPGA; the rest checks them or drives them. The sources are the **final**
versions: the ones behind the shipping bitstream in `../FPGA/` (solution `sol_YA128`, 250 MHz).

| Folder | What it is |
|---|---|
| `hls_kernel/` | The convolution engine that becomes FPGA hardware. `conv_engine_top.cpp` holds the synthesis top function `y26_conv_top`; `conv_engine.cpp/.h` is the datapath; `silu_lut.h` is the fixed-point SiLU table. |
| `reference_model/` | The whole YOLO26s network in C++ (the "C-sim"): `yolo26_network.cpp` runs the layers, `layer_ops.h` holds conv/quantize/etc., `run_model.cpp` is the command-line driver. It is the bit-exact golden reference the kernel is checked against, and on the board it is also the host program's network code. |
| `testbench/` | `conv_engine_tb.cpp` runs every conv through the kernel and through the reference and compares them. `detection_decode_test.cpp` checks the box decoder. `frame_conv_geometry.txt` lists each of the 100 kernel convs with its real input size. |
| `board_host/` | The ZCU102 ARM-side program: `board_host.cpp` packs activations/weights into DDR, starts the kernel, unpacks results. Built together with `reference_model/` and `hls_kernel/`. |
| `scripts/` | Build and check scripts (below) and the Vitis HLS / Vivado Tcl flows. |
| `export/` | Python scripts that exported the PyTorch/OpenVINO model to `weights/` and validated each step. They were run from the original `UOD/hls/` layout, so their default paths assume it. |
| `weights/` | The shipping weights: INT8 SmoothQuant, channel-compacted, constant-folded (`weights_sq_compact_fold`), as flat `.bin` files plus manifests. |
| `reference_dumps/` | Known-good outputs for one 640×640 frame, `input.bin` (float32, 3×640×640). `int8_shipping_model/`: every dumped layer output of the reference model with `weights/`; the build script checks against these byte for byte. `fp32_dense_model_vs_pytorch/`: an earlier stage (FP32 model before compaction): C-sim (`*_csim.bin`) next to PyTorch (`*_python.bin`) outputs, as `export/compare_cosine.py` compared them. |

## How to run (from `repo/`, in MSYS2 bash; Vitis include dir at `C:/AMDDesignTools/2026.1/Vitis/include` or set `VITIS_INC`)

| Command | What it checks | Expected |
|---|---|---|
| `bash model_c/scripts/build_reference_model.sh` | Builds the reference model and runs it on `reference_dumps/input.bin`. | `32/32 layer outputs byte-identical` |
| `bash model_c/scripts/run_kernel_gates.sh` | Kernel vs reference on all convs at 16×16, three arithmetic modes. | `all three gates pass` |
| `bash model_c/scripts/run_kernel_gate_all_convs.sh` | Kernel vs reference, bit-exact, all 100 convs at their real sizes. | `100/100 PASS` |
| `bash model_c/board_host/test_packing.sh "$PWD/model_c/weights" "$PWD/model_c/reference_dumps/input.bin"` | The host's DDR packing: whole network through the host path vs plain CPU. | `PASS: 32 dumps byte-identical` |
| `bash model_c/board_host/build_board_host.sh board` | The board program (on the ZCU102, or `CXX=aarch64-linux-gnu-g++`). `sim` builds a PC stand-in. | refuses if its flags differ from `FPGA/1_hls_synthesis/hls_build_flags.txt` |

Binaries and outputs go to `model_c/build/` (git-ignored). All five were run on 2026-09-23 with the expected result.

**HLS synthesis** (Vitis HLS 2026.1): the path must be space-free, so map the repo to a drive first, then pass the
shipping flags:
```
subst Y: "C:\Users\Prabhjit Dutta\Desktop\MTP\Project\UOD\repo"
$env:Y26_FLAGS = (Get-Content Y:\FPGA\1_hls_synthesis\hls_build_flags.txt | Where-Object { $_ -notmatch '^#' }) -join ' '
$env:Y26_SOL = "sol_YA128"; vitis-run --mode hls --tcl Y:/model_c/scripts/hls_synth_cosim_skip_csim.tcl
```
`hls_csim_synth_cosim.tcl` is the same flow with Vitis C-simulation first. `vivado_power_report.tcl` produces the
post-route timing/utilization/power reports. Each script's header documents its environment variables.

## Rename map (for matching against notes and logs, which use the working-tree names)

| Here | Working-tree name |
|---|---|
| `hls_kernel/conv_engine.cpp`, `.h` | `yolo26_hls.cpp`, `.h` |
| `hls_kernel/conv_engine_top.cpp` | `yolo26_hls_top.cpp` |
| `hls_kernel/silu_lut.h` | `yolo26_silu_lut.h` |
| `reference_model/run_model.cpp` | `main.cpp` |
| `reference_model/yolo26_network.cpp`, `.h` | `yolo26_trunk.cpp`, `.h` |
| `reference_model/layer_ops.h` | `yolo26_utils.h` |
| `reference_model/tensor_types.h` | `yolo26_types.h` |
| `reference_model/synthesis_guards.h` | `hls_synth.h` |
| `reference_model/detection_decode.h` | `decode.h` |
| `testbench/conv_engine_tb.cpp` | `yolo26_hls_tb.cpp` |
| `testbench/detection_decode_test.cpp` | `decode_test.cpp` |
| `testbench/frame_conv_geometry.txt` | `pwr/frame_geom100.txt` |
| `board_host/board_host.cpp` | `host/src/y26_board.cpp` |
| `board_host/build_board_host.sh` | `host/src/build.sh` |
| `board_host/test_quantizer.cpp` | `host/src/test_quant.cpp` |
| `board_host/compare_dumps_cosine.py` | `host/src/cmp.py` |
| `board_host/frames_to_u8.py` | `host/src/to_u8.py` |
| `scripts/build_reference_model.sh` | `build_csim.sh` (rewritten for this layout) |
| `scripts/run_kernel_gates.sh` | `build_hls_tb.sh` |
| `scripts/run_kernel_gate_all_convs.sh` | `gate_realgeom.sh` (rewritten for this layout) |
| `scripts/hls_csim_synth_cosim.tcl` | `run_hls_zu9eg.tcl` |
| `scripts/hls_synth_cosim_skip_csim.tcl` | `run_hls_csynth_only.tcl` |
| `scripts/vivado_power_report.tcl` | `run_vivado_power.tcl` |
| `reference_dumps/int8_shipping_model/` | old `dumps/shipping/` |
| `reference_dumps/fp32_dense_model_vs_pytorch/` | old `dumps/dense/` |

Only file names, `#include` lines, paths and file-name mentions in comments changed; identifiers and logic are
untouched. The kernel's gate defaults were brought up to the shipping flag set.
