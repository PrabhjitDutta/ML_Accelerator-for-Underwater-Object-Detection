#!/usr/bin/env python3
"""Score the SmoothQuant integer oracle (per-layer equivalent of the C++ model) over val2018 with
pycocotools, decoding the one2one head on the host.

  python model_c/export/sq_eval_map.py [--limit N]
"""
import os
import sys
import argparse
import numpy as np
import torch
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ingest_smoothquant import ingest  # noqa: E402

BASE = "/workspace/ckarfa/projects/UOD/dataset/urpc 2018"
GT = os.path.join(BASE, "annotations/instances_val2018.json")
IMG = os.path.join(BASE, "val2018/images")

def preprocess(path, size=640):
    im0 = cv2.imread(path)
    lb = LetterBox((size, size), auto=False, stride=32)
    im = lb(image=im0)[:, :, ::-1]                        # BGR->RGB
    im = np.ascontiguousarray(im.transpose(2, 0, 1))      # HWC->CHW
    x = torch.from_numpy(im.astype(np.float32) / 255.0).unsqueeze(0)
    return x, im0.shape[:2]                               # (H0, W0)

IRXML = "/workspace/ckarfa/projects/UOD/final_models/pruned50/int8_smoothquant_openvino_model/best.xml"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (smoke)")
    ap.add_argument("--backend", choices=["oracle", "ov"], default="oracle",
                    help="oracle = SmoothQuant PyTorch integer oracle (== C-sim); "
                         "ov = the deployed OpenVINO model itself, through this SAME harness")
    args = ap.parse_args()

    if args.backend == "ov":
        import openvino as ov
        cm = ov.Core().compile_model(ov.Core().read_model(IRXML), "CPU")
        trunk = None
    else:
        trunk, _, _, _ = ingest()
        trunk.eval()
    conf = 0.001

    coco = COCO(GT)
    stem2id = {os.path.splitext(im["file_name"])[0]: im["id"] for im in coco.dataset["images"]}
    paths = sorted(p for p in os.listdir(IMG) if p.endswith(".jpg"))
    if args.limit:
        paths = paths[:args.limit]

    results, eval_ids = [], []
    with torch.no_grad():
        for k, fn in enumerate(paths):
            iid = stem2id.get(os.path.splitext(fn)[0])
            if iid is None:
                continue
            eval_ids.append(iid)
            x, shape0 = preprocess(os.path.join(IMG, fn))
            if args.backend == "ov":
                det = list(cm(x.numpy()).values())[0][0]  # (300, 6): x1y1x2y2, conf, cls (640 space)
            else:
                det = trunk.decode(trunk(x))[0].cpu().numpy()
            det = det[det[:, 4] >= conf]
            if len(det) == 0:
                continue
            boxes = scale_boxes((640, 640), torch.from_numpy(det[:, :4].copy()), shape0).numpy()
            for (x1, y1, x2, y2), s, c in zip(boxes, det[:, 4], det[:, 5].astype(int)):
                results.append({"image_id": iid, "category_id": int(c),
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                "score": float(s)})
            if (k + 1) % 50 == 0:
                print(f"[sq-eval] {k+1}/{len(paths)} images, {len(results)} dets", flush=True)

    ev = COCOeval(coco, coco.loadRes(results), "bbox")
    ev.params.imgIds = eval_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    print(f"\n[sq-eval] SmoothQuant oracle over {len(eval_ids)} val2018 images: "
          f"mAP={ev.stats[0]:.4f} mAP50={ev.stats[1]:.4f} mAP75={ev.stats[2]:.4f}")
    print("[sq-eval] deployed OpenVINO SmoothQuant reference: mAP50=0.7546")

if __name__ == "__main__":
    main()
