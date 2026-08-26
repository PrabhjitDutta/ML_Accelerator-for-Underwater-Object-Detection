#!/usr/bin/env python3
"""Score the actual C-simulation over val2018 with pycocotools.

Phases 1-3 scored the PyTorch *oracle* (which the C-sim matches per-layer), because running the C++
over the whole val set is slow. The constant-folding variant is a weights transformation that only
exists on the C-sim side, so it has to be scored by really running the C++ binary on every image.

Preprocessing and decode are byte-identical to sq_eval_map.py (same LetterBox, same host-side
Yolo26Trunk.decode over the one2one head, same conf/scale_boxes), so the numbers are directly
comparable to the 0.7332 oracle / 0.7546 OpenVINO references.

  conda run -n ueaod python hls/export/csim_eval_map.py <weights_dir> [--limit N] [--cpp_decode]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/workspace/ckarfa/projects/UOD/training/yolo26s")
from yolo26_trunk import Yolo26Trunk  # noqa: E402
from ultralytics import YOLO  # noqa: E402

BASE = "/workspace/ckarfa/projects/UOD/dataset/urpc 2018"
GT = os.path.join(BASE, "annotations/instances_val2018.json")
IMG = os.path.join(BASE, "val2018/images")
CKPT = "/workspace/ckarfa/projects/UOD/final_models/pruned50/yolo26s_urpc2018_pruned50_fp32.pt"
# heads-only build: skips the ~40 MB/image of intermediate layer dumps that decode never
# reads (verified byte-identical head maps). ~4x faster over a whole val pass.
# YOLO26_CSIM_BIN overrides which binary is scored -- set it to csim/yolo26_csim_deploy to score the
# deployment shape (one2many head compiled out). That build emits no o2m_* dumps, so it REQUIRES
# --cpp_decode; the Python-decode path below reads both tags and would fail on the missing files.
CSIM = os.environ.get("YOLO26_CSIM_BIN") or os.path.join(HERE, "..", "csim", "yolo26_csim_fast")
DECODE = os.path.join(HERE, "..", "csim", "decode_test")   # C++ (ARM-PS) one2one decoder, decode.h
SHAPES = [(8, 80, 80), (8, 40, 40), (8, 20, 20)]


def cpp_decode(tmp, conf):
    """Run the standalone C++ decoder (decode.h via decode_test) on the head-map dumps the C-sim just
    wrote, and return dets as [N,6] = x1,y1,x2,y2,score,cls -- identical layout to Yolo26Trunk.decode's
    numpy output, so the downstream conf-filter/scale_boxes/COCO-format code is shared byte-for-byte."""
    out = os.path.join(tmp, "cpp_dets.txt")
    subprocess.run([DECODE, tmp, out], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = []
    with open(out) as f:
        for ln in f:
            p = ln.split()
            if len(p) == 6:
                cls, s, x1, y1, x2, y2 = int(p[0]), *map(float, p[1:])
                rows.append([x1, y1, x2, y2, s, cls])
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def preprocess(path, size=640):
    im0 = cv2.imread(path)
    lb = LetterBox((size, size), auto=False, stride=32)
    im = lb(image=im0)[:, :, ::-1]
    im = np.ascontiguousarray(im.transpose(2, 0, 1))
    return (im.astype(np.float32) / 255.0), im0.shape[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="C-sim weights dir (e.g. hls/weights_sq_compact_fold)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpp_decode", action="store_true",
                    help="use the C++ decoder (decode.h/decode_test) instead of Python trunk.decode -- "
                         "makes the whole image->detections path C++ (the deployment path)")
    args = ap.parse_args()
    wdir = os.path.abspath(args.weights)

    trunk = Yolo26Trunk(YOLO(CKPT).model).float().eval()   # decode() only; weights unused
    conf = 0.001

    coco = COCO(GT)
    stem2id = {os.path.splitext(im["file_name"])[0]: im["id"] for im in coco.dataset["images"]}
    paths = sorted(p for p in os.listdir(IMG) if p.endswith(".jpg"))
    if args.limit:
        paths = paths[:args.limit]

    results, eval_ids = [], []
    # ~5 MB of head-map dumps per run; this used to leak one dir per invocation into /tmp.
    tmp = tempfile.mkdtemp(prefix="csim_eval_")
    try:
        run_eval(tmp, paths, stem2id, wdir, args, trunk, conf, results, eval_ids)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    ev = COCOeval(coco, coco.loadRes(results), "bbox")
    ev.params.imgIds = eval_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    print(f"\n[csim-eval] {wdir}")
    print(f"[csim-eval] over {len(eval_ids)} val2018 images: mAP={ev.stats[0]:.4f} "
          f"mAP50={ev.stats[1]:.4f} mAP75={ev.stats[2]:.4f}")
    print("[csim-eval] references: SmoothQuant oracle mAP50=0.7332, deployed OpenVINO mAP50=0.7546")


def run_eval(tmp, paths, stem2id, wdir, args, trunk, conf, results, eval_ids):
    inbin = os.path.join(tmp, "input.bin")
    with torch.no_grad():
        for k, fn in enumerate(paths):
            iid = stem2id.get(os.path.splitext(fn)[0])
            if iid is None:
                continue
            eval_ids.append(iid)
            x, shape0 = preprocess(os.path.join(IMG, fn))
            x.astype("<f4").tofile(inbin)
            subprocess.run([CSIM, wdir, inbin, tmp], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           env={**os.environ, "YOLO26_HEADS_ONLY": "1",
                                # ultralytics sets OMP_NUM_THREADS=1 at import time to stop ITS
                                # workers oversubscribing. Inheriting that pins the OpenMP C-sim to
                                # a single core of 48 -- ~15 s/image instead of ~2.5 s.
                                "OMP_NUM_THREADS": str(min(32, os.cpu_count() or 8))})
            if args.cpp_decode:
                det = cpp_decode(tmp, conf)
            else:
                maps = []
                for tag in ("o2m", "o2o"):
                    for i, shp in enumerate(SHAPES):
                        a = np.fromfile(os.path.join(tmp, f"{tag}_{i}_csim.bin"),
                                        dtype="<f4").reshape(1, *shp)
                        maps.append(torch.from_numpy(a.copy()))
                det = trunk.decode(tuple(maps))[0].cpu().numpy()
            det = det[det[:, 4] >= conf] if len(det) else det
            if len(det) == 0:
                continue
            boxes = scale_boxes((640, 640), torch.from_numpy(det[:, :4].copy()), shape0).numpy()
            for (x1, y1, x2, y2), s, c in zip(boxes, det[:, 4], det[:, 5].astype(int)):
                results.append({"image_id": iid, "category_id": int(c),
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                "score": float(s)})
            if (k + 1) % 25 == 0:
                print(f"[csim-eval] {k+1}/{len(paths)} images, {len(results)} dets", flush=True)


if __name__ == "__main__":
    main()
