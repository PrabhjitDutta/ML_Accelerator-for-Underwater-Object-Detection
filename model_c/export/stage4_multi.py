#!/usr/bin/env python3
"""End-to-end check over several images: per image, letterbox -> PyTorch head maps -> C++ model ->
decode both -> diff. Each image gets its own dumps dir; results go to stage4_results/ (JSON + markdown).

  python model_c/export/stage4_multi.py [--csim <binary>] [--weights <dir>] [img ...]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training", "yolo26s"))

from yolo26_trunk import Yolo26Trunk  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402

import cv2  # noqa: E402

CKPT = os.path.join(ROOT, "final_models", "pruned50", "yolo26s_urpc2018_pruned50_fp32.pt")
VAL = os.path.join(ROOT, "dataset", "urpc 2018", "val2018", "images")
# [0] is make_input.py's default image.
IMAGES = [
    os.path.join(VAL, "CHN083846_0270.jpg"),
    os.path.join(VAL, "CHN083846_0291.jpg"),
    os.path.join(VAL, "CHN083846_0316.jpg"),
]
SHAPES = [(8, 80, 80), (8, 40, 40), (8, 20, 20)]
SIZE = 640
CONF = 0.25

def csim_env():
    """The MSYS2-built .exe needs the ucrt64 DLLs first on PATH; set only for the child (process-wide it
    would shadow the Windows python).
    """
    env = dict(os.environ)
    ucrt = r"C:\msys64\ucrt64\bin"
    if os.path.isdir(ucrt):
        env["PATH"] = ucrt + os.pathsep + env.get("PATH", "")
    return env

def letterbox(img_path, out_bin):
    """Same tensor make_input.py writes."""
    im0 = cv2.imread(img_path)
    if im0 is None:
        raise FileNotFoundError(img_path)
    im = LetterBox((SIZE, SIZE), auto=False, stride=32)(image=im0)
    im = im[:, :, ::-1]                                     # BGR -> RGB
    im = np.ascontiguousarray(im.transpose(2, 0, 1))        # HWC -> CHW
    x = (im.astype(np.float32) / 255.0)
    assert x.shape == (3, SIZE, SIZE), x.shape
    x.tofile(out_bin)
    return x

def tensor_stats(a, b):
    """Same three numbers compare_cosine.py reports, in float64."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float(a @ b / denom) if denom > 0 else (1.0 if not a.any() and not b.any() else 0.0)
    return {
        "cosine": cos,
        "max_abs_err": float(np.max(np.abs(a - b))),
        "rel_L2": float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)),
    }

def detections(trunk, maps):
    """maps: 6 tensors in trunk order (o2m_0..2, o2o_0..2) -> (n,6) array filtered at CONF."""
    with torch.no_grad():
        d = trunk.decode(tuple(maps))[0].cpu().numpy()
    return d[d[:, 4] >= CONF]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", default=None)
    ap.add_argument("--csim", default=os.path.join(ROOT, "hls", "csim", "build", "yolo26_csim_win.exe"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "hls", "weights"))
    ap.add_argument("--dumps-root", default=os.path.join(ROOT, "hls", "dumps_stage4"))
    ap.add_argument("--out", default=os.path.join(HERE, "stage4_results"))
    args = ap.parse_args()
    images = args.images if args.images else IMAGES

    os.makedirs(args.out, exist_ok=True)
    m = YOLO(CKPT)
    trunk = Yolo26Trunk(m.model).float().eval()
    nl = trunk.nl
    names = [f"o2m_{i}" for i in range(nl)] + [f"o2o_{i}" for i in range(nl)]

    results = []
    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        ddir = os.path.join(args.dumps_root, stem)
        os.makedirs(ddir, exist_ok=True)
        print(f"\n=== {stem} ===")

        # 1. shared input tensor
        x = letterbox(path, os.path.join(ddir, "input.bin"))
        xt = torch.from_numpy(x.copy()).view(1, 3, SIZE, SIZE)

        # 2. PyTorch head maps (the oracle)
        with torch.no_grad():
            out = trunk(xt)
        py_maps = [out[i] for i in range(2 * nl)]
        for n, t in zip(names, py_maps):
            a = t.detach().float().cpu().numpy()[0]
            np.ascontiguousarray(a, dtype="<f4").tofile(os.path.join(ddir, f"{n}_python.bin"))

        # 3. the C++ trunk on the identical buffer
        t0 = time.time()
        subprocess.run([args.csim, args.weights, os.path.join(ddir, "input.bin"), ddir],
                       check=True, stdout=subprocess.DEVNULL, env=csim_env())
        csim_s = time.time() - t0

        # 4. head-map agreement (what decode actually consumes)
        cs_maps, hm = [], {}
        for n, shp in zip(names, SHAPES * 2):
            a = np.fromfile(os.path.join(ddir, f"{n}_csim.bin"), dtype="<f4").reshape(1, *shp)
            cs_maps.append(torch.from_numpy(a.copy()))
            hm[n] = tensor_stats(a, py_maps[names.index(n)].detach().cpu().numpy())

        # 5. same decode on both sides; only the trunk varies
        py = detections(trunk, py_maps)
        cs = detections(trunk, cs_maps)
        n = min(len(py), len(cs))
        d = np.abs(py[:n] - cs[:n]) if n else np.zeros((0, 6))
        rec = {
            "image": path,
            "stem": stem,
            "dumps_dir": ddir,
            "n_det_pytorch": int(len(py)),
            "n_det_csim": int(len(cs)),
            "n_compared": int(n),
            "box_max_abs_err_px": float(d[:, :4].max()) if n else None,
            "conf_max_abs_err": float(d[:, 4].max()) if n else None,
            "classes_identical": bool(np.array_equal(py[:n, 5], cs[:n, 5])) if n else None,
            "csim_seconds": round(csim_s, 1),
            "head_maps": hm,
            "detections_pytorch": [
                {"cls": int(r[5]), "conf": round(float(r[4]), 6),
                 "box": [round(float(v), 4) for v in r[:4]]} for r in py
            ],
        }
        results.append(rec)
        print(f"  detections: pytorch={rec['n_det_pytorch']} csim={rec['n_det_csim']}")
        print(f"  box max|d| = {rec['box_max_abs_err_px']:.4e} px   "
              f"conf max|d| = {rec['conf_max_abs_err']:.4e}   "
              f"classes {'identical' if rec['classes_identical'] else 'DIFFER'}")
        worst = min(hm[f'o2o_{i}']['cosine'] for i in range(nl))
        print(f"  worst o2o head-map cosine = {worst:.6f}   (c-sim {csim_s:.0f}s)")

    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": "windows / msys2 g++ 15.2.0, -O3 -march=native -fopenmp",
        "csim_binary": args.csim,
        "weights": args.weights,
        "checkpoint": CKPT,
        "conf_threshold": CONF,
        "note": "Rebuilt locally; -march=native FMA contraction means these are NOT bit-comparable "
                "to the server-built C-sim numbers recorded in notes/yolo26s-c-sim.md.",
        "results": results,
    }
    jpath = os.path.join(args.out, "stage4_fp32.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    lines = [
        "# Stage-4 (task-level) FP32 verification — per image",
        "",
        f"Generated {meta['generated']}. Build: {meta['host']}.",
        "",
        "> These numbers come from a **local Windows rebuild** of the C-sim. `-march=native` FMA",
        "> contraction perturbs the arithmetic, so they are not bit-comparable to the server-built",
        "> figures in `notes/yolo26s-c-sim.md` — same order of magnitude, different last digits.",
        "",
        f"Confidence threshold {CONF}. Both sides decoded with the *same* `Yolo26Trunk.decode()`,",
        "so every difference is attributable to the trunk.",
        "",
        "| image | dets (PyTorch) | dets (C-sim) | box max\\|Δ\\| (px) | conf max\\|Δ\\| | classes | worst o2o cosine |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        w = min(r["head_maps"][f"o2o_{i}"]["cosine"] for i in range(nl))
        lines.append(
            f"| `{r['stem']}` | {r['n_det_pytorch']} | {r['n_det_csim']} | "
            f"{r['box_max_abs_err_px']:.3e} | {r['conf_max_abs_err']:.3e} | "
            f"{'identical' if r['classes_identical'] else 'DIFFER'} | {w:.6f} |"
        )
    lines += ["", "## Per-image head-map agreement (the 3 maps decode actually reads)", ""]
    for r in results:
        lines += [f"### `{r['stem']}`", "",
                  "| map | cosine | max_abs_err | rel_L2 |", "|---|---|---|---|"]
        for i in range(nl):
            s = r["head_maps"][f"o2o_{i}"]
            lines.append(f"| o2o_{i} | {s['cosine']:.6f} | {s['max_abs_err']:.3e} | {s['rel_L2']:.3e} |")
        lines.append("")
    mpath = os.path.join(args.out, "stage4_fp32.md")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[stage4] wrote {jpath}")
    print(f"[stage4] wrote {mpath}")
    print(f"[stage4] per-image dumps kept under {args.dumps_root}")

if __name__ == "__main__":
    main()
