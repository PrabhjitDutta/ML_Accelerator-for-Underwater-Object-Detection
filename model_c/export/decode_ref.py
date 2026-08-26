#!/usr/bin/env python3
"""Golden-reference gate for the C++ decoder (hls/csim/decode.h).

Reads the three one2one head maps (o2o_{0,1,2}_csim.bin) from a C-sim dump dir, runs the ORIGINAL
ultralytics end2end decode via Yolo26Trunk.decode (the exact same call csim_eval_map.py scores mAP
with), and compares — detection-by-detection — against the C++ decoder's output on the identical bins.

Because the C++ decode consumes the very same float32 head maps the Python decode does, a correct port
must match to floating-point round-off. We sort both sides by (score desc) and check per-detection
agreement on class, score, and the four box coords.

  conda run -n ueaod python hls/export/decode_ref.py <dump_dir> [--cpp_out cpp_dets.txt] [--tol 1e-3]
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
    # trunk.decode wants (o2m_0..2, o2o_0..2); o2m is unused by the inference decode but split() expects
    # 2*nl entries, so feed the o2o maps in the o2m slots too (ignored) to keep the shape contract.
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

    # Canonical order: score desc, then geometry (cls, box coords) so that detections with tied scores
    # -- which torch.topk and std::partial_sort may order differently -- still align set-to-set. Without
    # the geometric tiebreak, a pure score sort leaves tied dets in each backend's own (arbitrary) order,
    # producing a spurious box mismatch at the tie even when both sides hold the identical detection set.
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
