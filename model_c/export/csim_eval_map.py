#!/usr/bin/env python3
"""Score the C++ model over val2018 with pycocotools, running the binary on every image.
Preprocessing and decode match sq_eval_map.py (LetterBox, one2one Yolo26Trunk.decode).

  python model_c/export/csim_eval_map.py <weights_dir> [--limit N] [--cpp_decode]
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
# UOD_ROOT: the project root (defaults to the server path).
ROOT = os.environ.get("UOD_ROOT", "/workspace/ckarfa/projects/UOD")
sys.path.insert(0, os.path.join(ROOT, "training/yolo26s"))
from yolo26_trunk import Yolo26Trunk  # noqa: E402
from ultralytics import YOLO  # noqa: E402

BASE = os.path.join(ROOT, "dataset/urpc 2018")
GT = os.path.join(BASE, "annotations/instances_val2018.json")
IMG = os.path.join(BASE, "val2018/images")
CKPT = os.path.join(ROOT, "final_models/pruned50/yolo26s_urpc2018_pruned50_fp32.pt")
# Heads-only build: skips intermediate dumps decode never reads.
# YOLO26_CSIM_BIN overrides the binary; the deploy build (no o2m head) requires --cpp_decode.
CSIM = os.environ.get("YOLO26_CSIM_BIN") or os.path.join(HERE, "..", "build", "yolo26_csim")
DECODE = os.path.join(HERE, "..", "build", "decode_test")   # C++ one2one decoder (detection_decode.h)
SHAPES = [(8, 80, 80), (8, 40, 40), (8, 20, 20)]

def cpp_decode(tmp, conf):
    """Run the C++ decoder on the head-map dumps; returns [N,6] = x1,y1,x2,y2,score,cls, the same layout
    as Yolo26Trunk.decode."""
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
    ap.add_argument("--workers", type=int, default=1,
                    help="images run in parallel (one binary each, own tmp dir); OMP threads split between them")
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
    from concurrent.futures import ThreadPoolExecutor
    nw = max(1, args.workers)
    omp = str(max(1, min(32, os.cpu_count() or 8) // nw))

    def one(k_fn):
        k, fn = k_fn
        iid = stem2id.get(os.path.splitext(fn)[0])
        if iid is None:
            return None, []
        wt = os.path.join(tmp, "w%d" % (k % nw)) if nw == 1 else tempfile.mkdtemp(dir=tmp)
        os.makedirs(wt, exist_ok=True)
        inbin = os.path.join(wt, "input.bin")
        x, shape0 = preprocess(os.path.join(IMG, fn))
        x.astype("<f4").tofile(inbin)
        subprocess.run([CSIM, wdir, inbin, wt], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env={**os.environ, "YOLO26_HEADS_ONLY": "1",
                            # ultralytics sets OMP_NUM_THREADS=1 at import; override it so the OpenMP binary uses all cores.
                            "OMP_NUM_THREADS": omp})
        with torch.no_grad():
            if args.cpp_decode:
                det = cpp_decode(wt, conf)
            else:
                maps = []
                for tag in ("o2m", "o2o"):
                    for i, shp in enumerate(SHAPES):
                        a = np.fromfile(os.path.join(wt, f"{tag}_{i}_csim.bin"),
                                        dtype="<f4").reshape(1, *shp)
                        maps.append(torch.from_numpy(a.copy()))
                det = trunk.decode(tuple(maps))[0].cpu().numpy()
        if nw > 1:
            shutil.rmtree(wt, ignore_errors=True)
        det = det[det[:, 4] >= conf] if len(det) else det
        if len(det) == 0:
            return iid, []
        boxes = scale_boxes((640, 640), torch.from_numpy(det[:, :4].copy()), shape0).numpy()
        return iid, [{"image_id": iid, "category_id": int(c),
                      "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                      "score": float(s)}
                     for (x1, y1, x2, y2), s, c in zip(boxes, det[:, 4], det[:, 5].astype(int))]

    with ThreadPoolExecutor(nw) as ex:          # map() keeps image order, so results are deterministic
        for k, (iid, dets) in enumerate(ex.map(one, enumerate(paths))):
            if iid is not None:
                eval_ids.append(iid)
                results.extend(dets)
            if (k + 1) % 25 == 0:
                print(f"[csim-eval] {k+1}/{len(paths)} images, {len(results)} dets", flush=True)

if __name__ == "__main__":
    main()
