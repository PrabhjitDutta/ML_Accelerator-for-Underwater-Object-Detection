#!/usr/bin/env python3
"""Dump per-layer PyTorch activations of the pruned50 YOLO26s trunk for cosine validation.

Loads pruned50 via YOLO(), wraps it in the existing Yolo26Trunk (training/yolo26s/yolo26_trunk.py),
feeds the shared hls/dumps/input.bin, and saves:
  * each top-level layer output 0..22  -> dumps/<i>_python.bin        ([C,H,W] float32)
  * the 6 raw head maps                -> dumps/o2m_{0..2}_python.bin, o2o_{0..2}_python.bin
  * a few layer-10 attention sub-outputs (cv1/attn/psablock/cv2) to localize attention bugs.
Every file is float32, batch dim squeezed, C-order [C,H,W] -- byte-identical layout to the C-sim dumps.

  conda run -n ueaod python hls/export/dump_yolo26_features.py
"""
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.path.join(HERE, "..", "dumps")
sys.path.insert(0, "/workspace/ckarfa/projects/UOD/training/yolo26s")
from yolo26_trunk import Yolo26Trunk  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CKPT = "/workspace/ckarfa/projects/UOD/final_models/pruned50/yolo26s_urpc2018_pruned50_fp32.pt"


def save(name, t):
    arr = t.detach().float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.ascontiguousarray(arr, dtype="<f4")
    arr.tofile(os.path.join(DUMPS, f"{name}_python.bin"))
    return arr.shape


def main():
    x = np.fromfile(os.path.join(DUMPS, "input.bin"), dtype="<f4")
    assert x.size == 3 * 640 * 640, x.size
    x = torch.from_numpy(x.copy()).view(1, 3, 640, 640)

    m = YOLO(CKPT)
    trunk = Yolo26Trunk(m.model).float().eval()

    feats = {}
    hooks = []

    def mk(name):
        def hook(mod, inp, out):
            feats[name] = out[0] if isinstance(out, (list, tuple)) else out
        return hook

    # top-level layers 0..22 (Detect is index 23, handled via forward return)
    for i in range(23):
        hooks.append(trunk.model[i].register_forward_hook(mk(str(i))))
    # layer-10 attention internals (the main risk area)
    l10 = trunk.model[10]
    hooks.append(l10.cv1.register_forward_hook(mk("l10_cv1")))
    hooks.append(l10.m[0].attn.register_forward_hook(mk("l10_attn")))
    hooks.append(l10.m[0].register_forward_hook(mk("l10_psablock")))

    with torch.no_grad():
        out = trunk(x)  # tuple(2*nl): o2m_0..2, o2o_0..2

    for name, t in feats.items():
        shp = save(name, t)
        print(f"  {name:12s} {tuple(shp)}")
    nl = trunk.nl
    for i in range(nl):
        save(f"o2m_{i}", out[i]);        save(f"o2o_{i}", out[nl + i])
    print(f"  head maps: o2m_0..{nl-1}, o2o_0..{nl-1}  no={trunk.no}")
    for h in hooks:
        h.remove()
    print(f"[dump] wrote python features -> {DUMPS}")


if __name__ == "__main__":
    main()
