# YOLO26s Underwater Object Detection — FPGA Deployment

## Purpose

Deploy a YOLO26s object detector, trained for underwater imagery (URPC2018), onto a
Xilinx ZCU102 (`xczu9eg`) FPGA as a real-time HLS accelerator. The goal is a model small
and quantized enough to fit the board's LUT/BRAM/DSP budget and power envelope, with
bit-exact correctness carried all the way from the trained PyTorch model down to the
synthesizable C++ kernel before any hardware is built.

## Repo layout

```
model_python/   the model itself: how it was trained, pruned, and quantized
model_c/        the HLS kernel: C-sim, weight export, and the fixed-point conv engine
```

## Pipeline

1. **Fine-tune** — COCO-pretrained YOLO26s fine-tuned on URPC2018.
   (`model_python/scripts/train_ft_resumable.py` → `Weights/yolo26s_urpc2018_finetuned.pt`)
2. **Prune** — structured channel pruning via `torch_pruning`, 50% of channels removed.
   (`model_python/scripts/prune_yolo26s.py`)
3. **Recovery fine-tune** — the pruned model regains the accuracy pruning cost it.
   (`model_python/scripts/train_prune_finetune.py` → `Weights/yolo26s_urpc2018_finetuned_pruned50.pt`,
   the final PyTorch checkpoint)
4. **INT8 quantization** — SmoothQuant, applied externally via OpenVINO's quantizer, producing
   an OpenVINO IR (`Weights/OpenVINO quantized IR/`). `scripts/ingest_smoothquant.py` reads that
   IR's per-layer scales back out and applies them to the PyTorch model in memory; there is no
   separate "quantized .pt" — the IR *is* the quantized model. `scripts/sq_eval_map.py` runs COCO
   mAP directly against the IR to confirm quantization didn't break accuracy.
5. **Export** — `scripts/export_weights_yolo26.py` / `compact_yolo26.py` dump every layer's
   weights, biases, and quant scales as flat `.bin` files (`model_c/weights/`) for the C++ side
   to load — no PyTorch/OpenVINO runtime needed from here on.
6. **C-sim** — `model_c/csim/` is the synthesizable HLS kernel (`yolo26_hls.cpp`) plus a
   testbench (`yolo26_hls_tb.cpp`) that checks it **bit-exact**, not "close enough", against a
   double-precision reference (`yolo26_trunk.cpp`) fed the same exported weights.

## Status: through C-sim

The kernel passes its stage-1 gate: fixed-point (`ap_int<32>` accumulator) output matches the
double-precision reference **exactly**, conv by conv, across the full network. This is the same
source file used for both the plain g++ testbench and the Vitis HLS C-sim/cosim testbench, so
nothing is validated in one and skipped in the other.

Not yet covered here: synthesis, FPGA implementation (timing/area/power), and RTL cosimulation —
that work is tracked separately, outside this repo.
