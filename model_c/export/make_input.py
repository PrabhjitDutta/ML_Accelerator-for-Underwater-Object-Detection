#!/usr/bin/env python3
"""Produce the single fixed input tensor consumed by BOTH the PyTorch dumper and the C-sim.

Letterbox one deterministic val2018 image to 3x640x640 (RGB, /255, CHW) exactly as ultralytics
would for inference, and write the *actual float32 tensor* to hls/dumps/input.bin. Because both the
PyTorch reference (dump_yolo26_features.py) and the C++ trunk read this identical buffer, there is zero
preprocessing mismatch to confound the per-layer cosine check.

  conda run -n ueaod python hls/export/make_input.py
"""
import os
import sys
import numpy as np
import cv2
from ultralytics.data.augment import LetterBox

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.path.join(HERE, "..", "dumps")
IMG = "/workspace/ckarfa/projects/UOD/dataset/urpc 2018/val2018/images/CHN083846_0270.jpg"
SIZE = 640


def main():
    global IMG
    if len(sys.argv) > 1:
        IMG = sys.argv[1]
    os.makedirs(DUMPS, exist_ok=True)
    im0 = cv2.imread(IMG)  # BGR HxWx3
    assert im0 is not None, f"could not read {IMG}"
    lb = LetterBox((SIZE, SIZE), auto=False, stride=32)
    im = lb(image=im0)                      # letterboxed BGR, SIZE x SIZE x 3
    im = im[:, :, ::-1]                      # BGR -> RGB
    im = np.ascontiguousarray(im.transpose(2, 0, 1))  # HWC -> CHW
    x = im.astype(np.float32) / 255.0        # [3,640,640] in [0,1]
    assert x.shape == (3, SIZE, SIZE), x.shape
    out = os.path.join(DUMPS, "input.bin")
    x.tofile(out)
    print(f"[make_input] {IMG}")
    print(f"[make_input] wrote {out}  shape={x.shape}  "
          f"min={x.min():.4f} max={x.max():.4f} mean={x.mean():.4f}")
    print(f"[make_input] first5={x.flatten()[:5].tolist()}")


if __name__ == "__main__":
    main()
