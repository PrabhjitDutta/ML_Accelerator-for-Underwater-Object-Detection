# Underwater Object Detection on an FPGA (YOLO26s)

This project takes **YOLO26s**, an object-detection neural network, and teaches it to find sea creatures in
underwater photos: sea cucumbers, sea urchins, scallops and starfish, from the URPC2018 dataset. It then runs the
network as custom hardware on a **Xilinx ZCU102 FPGA board**.

## What is in this repo

| Folder | What it contains |
|---|---|
| `model_python/` | The neural network in Python: training it, making it smaller (pruning), and converting it to 8-bit numbers (quantization). |
| `model_c/` | The same network rewritten in C++. It includes the part that becomes FPGA hardware, the tests that check it, and the program that runs on the board's ARM processor. |
| `FPGA/` | The finished hardware build: the generated Verilog, the build reports, and the bitstream file that programs the board. |

Each folder has its own `README.md` with the details.

## How it was built, step by step

1. **Train:** start from a YOLO26s model trained on everyday photos (COCO) and fine-tune it on underwater images.
2. **Shrink:** remove half of the network's channels (pruning), then retrain briefly to recover the lost accuracy.
3. **Quantize:** convert weights and activations from 32-bit floats to 8-bit integers (SmoothQuant, via OpenVINO).
   Smaller numbers need far less hardware.
4. **Export:** save every layer's weights as plain binary files in `model_c/weights/`, so the C++ code needs no
   Python.
5. **C++ model:** rebuild the network in C++. The part that runs on the FPGA, the convolution engine, is tested
   against a C++ reference model. It must match **exactly**, bit for bit, on every layer.
6. **Hardware:** Vitis HLS turns the C++ convolution engine into Verilog. Vivado then places and routes it on the
   chip and produces the bitstream (`FPGA/`).
7. **Run on the board:** the ZCU102's ARM processor loads an image and sends each layer to the FPGA hardware. It then
   turns the network's output into boxes around the detected objects (`model_c/board_host/`).

## Results so far

| | |
|---|---|
| FPGA clock | 250 MHz, timing met |
| Hardware time per 640×640 image | **66.0 ms, about 15 images per second** (69.2 ms without the host's Img2Col rewrite of the first layer). Measured in simulation; counts the FPGA's convolutions only (its attention block adds 4.7 ms). The figure is from the previous kernel build; the current one adds 0.2 to 0.4% on the four layers that were simulated again. |
| On the board, whole program | **191.7 ms per image, 5.22 images per second** (20 test images, one at a time); **8.4 images per second** with four images overlapped. This includes the ARM processor's share of the work. |
| Chip resources used | 32% of logic (LUTs), 27% of block RAM, 43% of DSP blocks |
| Correctness | On the board, the FPGA matches the C++ reference exactly on all 100 convolution layers and on both attention layers. |
| Accuracy | mAP50 0.755 on the 800 URPC2018 validation images, from the board program run on a PC; on the board it gives the same detections on the 20 test images. |

**Status:** the design runs on the ZCU102. The FPGA computes every convolution and the core of the two attention
layers; the ARM processor does the rest. Where one convolution feeds the next directly, the FPGA writes its
result in the 8-bit form the next convolution reads, so the processor does not touch that data.

## Quick start

To check the C++ side on a PC (needs `g++`; `model_c/README.md` lists every check):

```bash
bash model_c/scripts/build_reference_model.sh
```

This builds the C++ model, runs it on a sample image, and compares the result with saved known-good outputs.
