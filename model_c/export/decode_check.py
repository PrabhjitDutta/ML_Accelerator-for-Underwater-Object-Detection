#!/usr/bin/env python3
"""Decode the C++ model's head maps with Yolo26Trunk.decode() and compare the detections to the PyTorch
pipeline on the same image.

  python model_c/export/decode_check.py [dumps_dir] [csim|python]
"""
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(HERE, "..", "dumps")           # input.bin always lives here
# argv[1]: head-map dumps dir; argv[2]: file suffix, csim (default) or python.
CSIM = sys.argv[1] if len(sys.argv) > 1 else INPUT_DIR
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else "csim"
sys.path.insert(0, "/workspace/ckarfa/projects/UOD/training/yolo26s")
from yolo26_trunk import Yolo26Trunk  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CKPT = "/workspace/ckarfa/projects/UOD/final_models/pruned50/yolo26s_urpc2018_pruned50_fp32.pt"
SHAPES = [(8, 80, 80), (8, 40, 40), (8, 20, 20)]

def main():
    m = YOLO(CKPT)
    trunk = Yolo26Trunk(m.model).float().eval()
    x = torch.from_numpy(np.fromfile(os.path.join(INPUT_DIR, "input.bin"), dtype="<f4").copy()).view(1, 3, 640, 640)
    with torch.no_grad():
        out = trunk(x)
        py = trunk.decode(out)[0].cpu().numpy()   # (max_det, 6): x1,y1,x2,y2,conf,cls

        # build the tuple from C-sim head maps
        csim = []
        for tag in ("o2m", "o2o"):
            for i, shp in enumerate(SHAPES):
                a = np.fromfile(os.path.join(CSIM, f"{tag}_{i}_{SUFFIX}.bin"), dtype="<f4").reshape(1, *shp)
                csim.append(torch.from_numpy(a.copy()))
        cs = trunk.decode(tuple(csim))[0].cpu().numpy()

    conf = 0.25
    py = py[py[:, 4] >= conf]
    cs = cs[cs[:, 4] >= conf]
    print(f"PyTorch detections (conf>={conf}): {len(py)}")
    print(f"C-sim   detections (conf>={conf}): {len(cs)}")
    n = min(len(py), len(cs))
    if n:
        d = np.abs(py[:n] - cs[:n])
        print(f"max abs diff over matched rows: box={d[:, :4].max():.4e} conf={d[:, 4].max():.4e} "
              f"cls={'same' if np.array_equal(py[:n, 5], cs[:n, 5]) else 'DIFF'}")
    for i in range(n):
        b = py[i]
        print(f"  cls{int(b[5])} conf={b[4]:.3f} box=[{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}]")

if __name__ == "__main__":
    main()
