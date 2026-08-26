#!/usr/bin/env python3
"""Phase 3 end-to-end parity: compare the deployed OpenVINO SmoothQuant model against the C-sim and
the PyTorch integer oracle on the shared fixed image (dumps/input.bin).

The OV IR (best.xml) outputs [1, 300, 6] decoded end2end detections (x1y1x2y2, conf, cls) in 640-input
space -- exactly what trunk.decode() produces from the o2o maps. So we decode the C-sim's and the
oracle's o2o head maps with the same host-side decode and line all three up. This closes the loop:
IR scales -> C-sim integer W8A8 -> same detections as the artifact they came from.

  conda run -n ueaod python hls/export/ov_reference.py
"""
import os
import sys
import numpy as np
import torch
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.path.join(HERE, "..", "dumps")
DUMPS_SQ = os.path.join(HERE, "..", "dumps_sq")
IRXML = "/workspace/ckarfa/projects/UOD/final_models/pruned50/int8_smoothquant_openvino_model/best.xml"
CKPT = "/workspace/ckarfa/projects/UOD/final_models/pruned50/yolo26s_urpc2018_pruned50_fp32.pt"
sys.path.insert(0, "/workspace/ckarfa/projects/UOD/training/yolo26s")
from yolo26_trunk import Yolo26Trunk  # noqa: E402
from ultralytics import YOLO  # noqa: E402

SHAPES = [(8, 80, 80), (8, 40, 40), (8, 20, 20)]
CONF = 0.25


def decode_o2o(trunk, dumps, suffix):
    o2o = []
    for i, shp in enumerate(SHAPES):
        a = np.fromfile(os.path.join(dumps, f"o2o_{i}_{suffix}.bin"), dtype="<f4").reshape(1, *shp)
        o2o.append(torch.from_numpy(a.copy()))
    d = trunk.decode(tuple([torch.zeros(1, *s) for s in SHAPES]) + tuple(o2o))[0].cpu().numpy()
    return d[d[:, 4] >= CONF]


def summarize(tag, det):
    print(f"{tag:10s} detections(conf>={CONF}): {len(det)}")
    for b in det[:8]:
        print(f"   cls{int(b[5])} conf={b[4]:.3f} box=[{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}]")


def main():
    x = np.fromfile(os.path.join(DUMPS, "input.bin"), dtype="<f4").reshape(1, 3, 640, 640)

    # deployed OpenVINO SmoothQuant model
    cm = ov.Core().compile_model(ov.Core().read_model(IRXML), "CPU")
    ov_out = list(cm(x).values())[0][0]                      # (300, 6)
    ov_det = ov_out[ov_out[:, 4] >= CONF]

    trunk = Yolo26Trunk(YOLO(CKPT).model).float().eval()
    cs_det = decode_o2o(trunk, DUMPS_SQ, "csim")             # SmoothQuant C-sim
    or_det = decode_o2o(trunk, DUMPS_SQ, "python")           # SmoothQuant PyTorch oracle

    summarize("OpenVINO", ov_det)
    summarize("SQ-oracle", or_det)
    summarize("SQ-csim", cs_det)

    # C-sim vs oracle box agreement: match each C-sim det to its nearest-center oracle det. Pairs
    # within 2px are true matches (report their box/conf agreement); the rest are conf~0.25 borderline
    # detections the tiny arith residual (gate 0.9994) nudges across the threshold on one side only.
    if len(cs_det) and len(or_det):
        cc = (cs_det[:, :2] + cs_det[:, 2:4]) / 2
        oc = (or_det[:, :2] + or_det[:, 2:4]) / 2
        j = np.argmin(((cc[:, None] - oc[None]) ** 2).sum(-1), axis=1)
        dcenter = np.sqrt(((cc - oc[j]) ** 2).sum(-1))
        mm = dcenter < 2.0
        dbox = np.abs(cs_det[mm, :4] - or_det[j[mm], :4]).max() if mm.any() else 0.0
        dconf = np.abs(cs_det[mm, 4] - or_det[j[mm], 4]).max() if mm.any() else 0.0
        print(f"\nC-sim vs oracle: {int(mm.sum())}/{len(cs_det)} dets matched <2px "
              f"(box max|Δ|={dbox:.2e}px conf max|Δ|={dconf:.2e}); "
              f"{int((~mm).sum())} borderline near conf={CONF}")
    print(f"counts: OpenVINO={len(ov_det)}  SQ-oracle={len(or_det)}  SQ-csim={len(cs_det)}")


if __name__ == "__main__":
    main()
