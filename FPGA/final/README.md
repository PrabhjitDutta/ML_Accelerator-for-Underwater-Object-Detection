# YOLO26s on the ZCU102: deployment package

Everything needed to run the YOLO26s detector on a ZCU102 board. Copy this whole folder onto the board
(it assumes a PYNQ or Linux image, root access, and a native `g++`).

The FPGA runs every INT8 convolution; the ARM cores run the rest of the network (attention, upsample,
concat, detection head) and drive the FPGA.

| Folder | Contents |
|---|---|
| `1_bitstream/` | `yolo26s_zcu102.bit` (the FPGA design) and `yolo26s_zcu102.hwh` (its hardware description, loaded with it) |
| `2_host_program/` | C++ source of the ARM program that runs the network and drives the FPGA, plus its build script |
| `3_model_weights/` | The trained, INT8-quantized model: 468 weight files + `manifest_sq.txt` |
| `4_first_run_check/` | A test image tensor and the output the board must reproduce, plus the comparison script |
| `5_frame_conversion_tool/` | Converts input frames to the compact 8-bit format the program reads fastest |

## 1. Load the bitstream

```
sudo python3 -c "from pynq import Overlay; Overlay('1_bitstream/yolo26s_zcu102.bit')"
```
The `.hwh` must stay next to the `.bit`. The fabric clock PL0 must read ~249975000 Hz (250 MHz).

## 2. Give the FPGA a DDR buffer

Preferred: the u-dma-buf kernel module (at least 32 MB):
```
sudo insmod u-dma-buf.ko udmabuf0=67108864
export Y26_UDMABUF=udmabuf0
```
Fallback: a device-tree `reserved-memory` region with `no-map` (e.g. 64 MB at 0x60000000), then
`export Y26_BUF_PHYS=0x60000000 Y26_BUF_SIZE=0x4000000`. This path is uncached and much slower.

## 3. Build the host program (on the board)

```
bash 2_host_program/board_host/build_board_host.sh board
```
Writes `2_host_program/build/y26_board`. The FPGA configuration flags are fixed inside the script and must not
be changed: they have to match the bitstream.

## 4. First run: check the board against the expected output

```
mkdir -p out
sudo -E YOLO26_HEADS_ONLY=1 Y26_BOARD_CHECK=. 2_host_program/build/y26_board \
     3_model_weights 4_first_run_check/reference_input.bin out
```
Expect `100 kernel convs` and `CHECK: 100 convs vs C-model, 0 MISMATCH`, then
```
python3 4_first_run_check/compare_output_maps.py out 4_first_run_check/expected_output_maps
```
(or `cmp` each `o2o_*_csim.bin`): the three output maps must be identical.

## 5. Normal runs

Drop `Y26_BOARD_CHECK` (it re-runs every conv on the CPU for comparison).

Single image: `y26_board <weights_dir> <input.bin> <output_dir>`.

Stream of frames (`-` reads file names from stdin, one timing line per frame):
```
python3 5_frame_conversion_tool/frames_to_u8.py frames/*.bin     # once: writes frames/<name>.u8
mkdir -p /dev/shm/out
ls frames/*.u8 | sudo -E YOLO26_HEADS_ONLY=1 2_host_program/build/y26_board 3_model_weights - /dev/shm/out
```
Input frames are 3x640x640 float32 tensors (RGB, letterboxed, divided by 255). The `.u8` form is the same
pixels as bytes; `.bin` frames also work. Outputs are the three one-to-one head maps `o2o_{0,1,2}_csim.bin`.

## Optional settings

| Variable | Effect |
|---|---|
| `YOLO26_HEADS_ONLY=1` | Write only the three output maps (not every layer) |
| `Y26_BOARD_CHECK=<name part>` | Also run matching convs (`.` = all) on the CPU and compare bit for bit |
| `Y26_FUSE_CHECK=1` | Compare every fast-path input packing with plain quantization; stop on the first difference |
| `Y26_IM2COL=0` | Turn off the host-side Img2Col rewrite of the first conv (on by default; same results) |
| `Y26_ASYNC=0` | Run independent branches serially instead of on separate threads |
| `Y26_PACK_MT=<elements>` | Pack a conv's input with OpenMP above this size (default 1048576) |
| `Y26_PACK_SCHED=dynamic` | OpenMP schedule for input packing (default static) |

`2_host_program/board_host/test_quantizer.cpp` checks the ARM (NEON) build of the input quantizer; build it with
the same `-D` flags as `build_board_host.sh` and it must print PASS.
