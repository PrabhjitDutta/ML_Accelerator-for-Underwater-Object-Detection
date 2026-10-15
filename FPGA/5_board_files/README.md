# YOLO26s on the ZCU102: deployment package

Everything needed to run the YOLO26s detector on a ZCU102 board. Copy this whole folder onto the board
(it assumes a PYNQ or Linux image, root access, and a native `g++`).

The FPGA runs every INT8 convolution and the core of the two attention layers (scores, softmax, weighted sum);
the ARM cores run the rest of the network (concat, pooling, detection head) and drive the FPGA. Where one
convolution feeds the next directly, the FPGA writes its output as the 8-bit input of the next one. A third block,
the ops block, works on those 8-bit codes: the residual adds, the 2x upsamples and the copy of 20x20 maps into the
next conv's layout (`Y26_OPS=1`, on in every command below).

**The host program and the bitstream must come from the same package.** A host program built from an older
package does not set the address of the kernel's code output, so on this bitstream the kernel would write to
physical address 0. `Y26_OPS=1` works only on this bitstream (on an older one the program dies with a bus error);
`Y26_ATTN2` must not be set (this bitstream has no second-version attention block: bus error).

| Folder | Contents |
|---|---|
| `1_bitstream/` | `yolo26s_zcu102_ops.bit` + `.hwh` (loaded together by PYNQ), and `yolo26s_zcu102_ops.bit.bin` + `.dtbo` (the same design for a plain Linux image such as Ubuntu for ZCU102) |
| `2_host_program/` | C++ source of the ARM program that runs the network and drives the FPGA, plus its build script |
| `3_model_weights/` | The trained, INT8-quantized model, refit so that tensors read by several convs share one 8-bit grid: 458 weight files + 4 metadata files (`manifest_sq.txt`, `attn_fq.txt`, `splits.txt`, `compaction_map.json`) |
| `4_first_run_check/` | A test image tensor and the output the board must reproduce, plus the comparison script; `expected_detections/` holds the detections of the 20 test frames |
| `5_frame_conversion_tool/` | Converts input frames to the compact 8-bit format the program reads fastest |
| `6_test_frames/` | 20 URPC2018 val2018 images, letterboxed to 640x640, already in `.u8` form (the first is the same image as `4_first_run_check`) |
| `7_udmabuf_source/` | Source of the u-dma-buf kernel module, v5.5.0 (github.com/ikwzm/udmabuf, BSD-2-Clause, unmodified) |
| `SHA256SUMS.txt` | Checksums of every file here: `sha256sum -c SHA256SUMS.txt` on the board confirms the copy |

## 1. Load the bitstream

PYNQ image:
```
sudo python3 -c "from pynq import Overlay; Overlay('1_bitstream/yolo26s_zcu102_ops.bit')"
```
The `.hwh` must stay next to the `.bit`.

Plain Linux image (e.g. Ubuntu for ZCU102), with `fpgautil`:
```
sudo fpgautil -b 1_bitstream/yolo26s_zcu102_ops.bit.bin -o 1_bitstream/yolo26s_zcu102_ops.dtbo
sudo cat /sys/kernel/debug/clk/clk_summary | grep -i pl0
```
The overlay loads the bitstream by its file name (`yolo26s_zcu102_ops.bit.bin`, so do not rename it), sets the
fabric clock and configures the PS-PL port widths. To load again, first remove the old overlay:
`sudo fpgautil -R`.

Either way, the fabric clock PL0 must read ~249975000 Hz (250 MHz).

## 2. Give the FPGA a DDR buffer

Preferred: the u-dma-buf kernel module (at least 64 MB; 128 MB for two overlapped frames, 192 MB = `201326592` for three, 256 MB = `268435456` for four), built once on the board against the running
kernel's headers:
```
sudo apt install build-essential linux-headers-$(uname -r)     # Ubuntu
(cd 7_udmabuf_source && make)     # run inside the folder: its Makefile builds the current directory
sudo insmod 7_udmabuf_source/u-dma-buf.ko udmabuf0=134217728
export Y26_UDMABUF=udmabuf0
```
Fallback: a device-tree `reserved-memory` region with `no-map` (e.g. 64 MB at 0x60000000), then
`export Y26_BUF_PHYS=0x60000000 Y26_BUF_SIZE=0x4000000`. This path is uncached and much slower.

## 3. Build the host program (on the board)

```
bash 2_host_program/board_host/build_board_host.sh board
```
Writes `2_host_program/build/y26_board`. The FPGA configuration flags are fixed inside the script and must not
be changed: they have to match the bitstream. `Y26_ATTN=0` in front of the command builds `y26_board_cpuattn`
instead, which computes attention on the ARM cores in floating point (slower; its output maps differ slightly
from the expected ones).

Build on the board, not on a PC: the program uses OpenMP, and a binary from the Vitis aarch64 cross-compiler
needs glibc 2.38 (Ubuntu 22.04 has 2.35) and has no OpenMP.

## 4. First run: check the board against the expected output

```
mkdir -p out
sudo -E YOLO26_HEADS_ONLY=1 Y26_OPS=1 Y26_BOARD_CHECK=. 2_host_program/build/y26_board \
     3_model_weights 4_first_run_check/reference_input.bin out
```
Expect `100 kernel convs`, `CHECK: 100 convs vs C-model, 0 MISMATCH` and two lines
`CHECK attention: 102400 J vs C-model, 0 differ -> BIT-EXACT`, then
```
python3 4_first_run_check/compare_output_maps.py out 4_first_run_check/expected_output_maps
```
(or `cmp` each `o2o_*_csim.bin`): the three output maps must be identical.

## 5. Normal runs

Drop `Y26_BOARD_CHECK` (it re-runs every conv on the CPU for comparison).

Single image: `y26_board <weights_dir> <input.bin> <output_dir>`.

Stream of frames (`-` reads file names from stdin, one timing line per frame):
```
mkdir -p /dev/shm/out
ls 6_test_frames/*.u8 | sudo -E YOLO26_HEADS_ONLY=1 Y26_OPS=1 2_host_program/build/y26_board 3_model_weights - /dev/shm/out
diff -r /dev/shm/out 4_first_run_check/expected_detections && echo SAME
```
Each frame prints `84 code-mode edges`, and the 20 detection files must equal the expected ones. The ops block
runs only in this frame loop (the single-image run of section 4 does not use it): with `Y26_BOARD_CHECK=.` added
and two frames (`head -2`), each frame also prints `CHECK ops block: ... 0 differ -> BIT-EXACT`.
Input frames are 3x640x640 float32 tensors (RGB, letterboxed, divided by 255). The `.u8` form is the same
pixels as bytes; `.bin` frames also work (convert them once with
`python3 5_frame_conversion_tool/frames_to_u8.py frames/*.bin`). A single-image run writes the three
one-to-one head maps `o2o_{0,1,2}_csim.bin`; the frame loop writes one `<frame>.txt` per frame, one detection
per line: `class score x1 y1 x2 y2` in 640x640 letterboxed pixels.

## Optional settings

| Variable | Effect |
|---|---|
| `Y26_OPS=1` | The ops block does the residual adds, the two upsamples and the 20x20 map copies on 8-bit codes (shipping setting; without it the ARM does them, about 9 ms slower, same weights, slightly different detections) |
| `YOLO26_HEADS_ONLY=1` | Write only the three output maps (not every layer) |
| `Y26_BOARD_CHECK=<name part>` | Also run matching convs (`.` = all), both attention blocks and every ops block call on the CPU and compare bit for bit |
| `Y26_FUSE_CHECK=1` | Compare every fast-path input packing with plain quantization; stop on the first difference |
| `Y26_IM2COL=0` | Turn off the host-side Img2Col rewrite of the first conv (on by default; same results) |
| `Y26_INFLIGHT=2` to `4` | Overlap frames in the frame loop (throughput, not latency; two need a 128 MB buffer, three a 192 MB one, four a 256 MB one, and the per-frame lines mix frames: read the final `[frames]` line) |
| `Y26_YC=0` | The ARM cores quantize and pack every conv input themselves (slower; detections then differ in a few boxes, because the FPGA rounds exact ties differently) |
| `Y26_SYNC_CMD=1` | One driver call per cache sync instead of four (about 2 ms per frame with one frame at a time) |
| `Y26_ASYNC=0` | Run independent branches serially instead of on separate threads |
| `Y26_PACK_MT=<elements>` | Pack a conv's input with OpenMP above this size (default 1048576) |
| `Y26_PACK_SCHED=dynamic` | OpenMP schedule for input packing (default static) |

`2_host_program/board_host/test_quantizer.cpp` checks the ARM (NEON) build of the input quantizer; build it with
the same `-D` flags as `build_board_host.sh` and it must print PASS.
