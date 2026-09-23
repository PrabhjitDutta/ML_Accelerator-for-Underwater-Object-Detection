#!/usr/bin/env python3
"""Reference check for the C++ decoder (detection_decode.h): run Yolo26Trunk.decode on a dump dir's o2o
head maps and compare per detection with the C++ decoder's output on the same files.

  python model_c/export/decode_ref.py <dump_dir> [--cpp_out cpp_dets.txt] [--tol 1e-3]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/ckarfa/projects/UOD/training/yolo26s")
from yolo26_trunk import Yolo26Trunk  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CKPT = "/workspace/ckarfa/projects/UOD/final_models/pruned50/yolo26s_urpc2018_pruned50_fp32.pt"
SHAPES = [(8, 80, 80), (8, 40, 40), (8, 20, 20)]

def py_decode(dump_dir, trunk, conf):
    maps = []
    # decode() expects 2*nl maps; o2m is unused, so reuse the o2o maps.
    o2o = [np.fromfile(os.path.join(dump_dir, f"o2o_{i}_csim.bin"), dtype="<f4").reshape(1, *s)
           for i, s in enumerate(SHAPES)]
    maps = [torch.from_numpy(a.copy()) for a in o2o] + [torch.from_numpy(a.copy()) for a in o2o]
    with torch.no_grad():
        det = trunk.decode(tuple(maps))[0].cpu().numpy()   # [N,6] = x1,y1,x2,y2,score,cls
    det = det[det[:, 4] >= conf]
    return det

def load_cpp(path):
    rows = []
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) != 6:
                continue
            cls, sc, x1, y1, x2, y2 = int(p[0]), *map(float, p[1:])
            rows.append([x1, y1, x2, y2, sc, cls])
    return np.array(rows, dtype=np.float32).reshape(-1, 6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--cpp_out", default=None, help="C++ decode_test output to compare against")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--dump_ref", default=None, help="also write the Python reference dets here")
    args = ap.parse_args()

    # Canonical order: score desc, then cls and box, so tied scores align across backends.
    def canon(a):
        return a[np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0], a[:, 5], -a[:, 4]))]

    trunk = Yolo26Trunk(YOLO(CKPT).model).float().eval()
    ref = canon(py_decode(args.dump_dir, trunk, args.conf))
    print(f"[ref] python decode: {len(ref)} dets (conf>={args.conf})")
    if args.dump_ref:
        with open(args.dump_ref, "w") as f:
            for x1, y1, x2, y2, s, c in ref:
                f.write(f"{int(c)} {s:.6f} {x1:.5f} {y1:.5f} {x2:.5f} {y2:.5f}\n")

    if not args.cpp_out:
        print("(no --cpp_out given; ref-only)")
        return

    cpp = load_cpp(args.cpp_out)
    cpp = canon(cpp[cpp[:, 4] >= args.conf])
    print(f"[cpp] c++ decode:   {len(cpp)} dets (conf>={args.conf})")

    n = min(len(ref), len(cpp))
    if len(ref) != len(cpp):
        print(f"!! count mismatch: ref={len(ref)} cpp={len(cpp)} (comparing first {n})")
    dcls = (ref[:n, 5] != cpp[:n, 5]).sum()
    dscore = np.abs(ref[:n, 4] - cpp[:n, 4]).max() if n else 0.0
    dbox = np.abs(ref[:n, :4] - cpp[:n, :4]).max() if n else 0.0
    print(f"  class mismatches : {dcls}/{n}")
    print(f"  max |dscore|     : {dscore:.3e}")
    print(f"  max |dbox| (px)  : {dbox:.3e}")
    ok = (len(ref) == len(cpp)) and (dcls == 0) and (dscore <= args.tol) and (dbox <= args.tol)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()
