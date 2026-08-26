#!/usr/bin/env python3
"""Realized params/MACs/BRAM/DSP of the channel-compacted YOLO26s trunk (Phase 4).

Mask pruning zeroes whole output filters but keeps tensor shapes dense, so it buys nothing in
hardware. `compact_yolo26.py` physically deletes the dead channels; this script measures what that
actually bought, per conv and in total, and translates it to ZCU102 (XCZU9EG) resources.

Output spatial sizes come from a real 640x640 forward (hooks on every Conv2d) -- the same method
`training/yolo26s/measure_prune_macs.py` uses -- and the kept channel counts come from
`hls/weights_compact/compaction_map.json`.

  conda run -n ueaod python hls/export/measure_compaction.py [map_json]
"""
import json
import os
import sys

import torch
import torch.nn as nn
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "/workspace/ckarfa/projects/UOD/final_models/pruned50/yolo26s_urpc2018_pruned50_fp32.pt"
MAP = os.path.join(HERE, "..", "weights_compact", "compaction_map.json")

# ZCU102 / XCZU9EG
BRAM36_KBIT = 36          # one BRAM36 tile = 36 Kbit
N_BRAM36 = 912
N_DSP48 = 2520


def spatial():
    """name -> (Hout, Wout, kh, kw, groups) for every Conv2d, from a real forward."""
    model = YOLO(CKPT).model.float().eval()
    out, hooks = {}, []

    def mk(name):
        def hook(module, inp, o):
            out[name] = (o.shape[-2], o.shape[-1], module.kernel_size[0], module.kernel_size[1],
                         module.groups)
        return hook

    for n, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(mk(n[len("model."):] if n.startswith("model.") else n)))
    with torch.no_grad():
        model(torch.zeros(1, 3, 640, 640))
    for h in hooks:
        h.remove()
    return out


def main():
    mpath = sys.argv[1] if len(sys.argv) > 1 else MAP
    cmap = json.load(open(mpath))
    convs = cmap["convs"]
    # dense oc/ic come from the ORIGINAL manifest the compaction was run against
    dense_mf = {}
    with open(os.path.join(cmap["source"], cmap["manifest"])) as f:
        for line in f:
            if not line.strip():
                continue
            t = line.split()
            dense_mf[t[0]] = {"oc": int(t[1]), "ic": int(t[2])}
    sp = spatial()

    rows = []
    for name, c in convs.items():
        if name not in sp:
            print(f"[warn] no spatial for {name}, skipped")
            continue
        H, W, kh, kw, g = sp[name]
        d = dense_mf[name]
        doc, dic = int(d["oc"]), int(d["ic"])
        coc, cic = int(c["oc"]), int(c["ic"])
        # depthwise stays depthwise: groups scale with channels
        cg = g if g == 1 else coc
        dp = doc * (dic // g) * kh * kw
        cp = coc * (cic // cg) * kh * kw
        rows.append((name, H, W, doc, dic, coc, cic, dp, cp, dp * H * W, cp * H * W))

    dP = sum(r[7] for r in rows)
    cP = sum(r[8] for r in rows)
    dM = sum(r[9] for r in rows)
    cM = sum(r[10] for r in rows)

    print(f"{'conv':28s} {'HxW':>9s} {'dense oc/ic':>12s} {'comp oc/ic':>12s} "
          f"{'params':>10s} {'->':>10s} {'MMAC':>9s} {'->':>9s}  cut")
    for (n, H, W, doc, dic, coc, cic, dp, cp, dm, cm) in rows:
        cut = 100.0 * (1 - cm / dm) if dm else 0.0
        print(f"{n:28s} {H:4d}x{W:<4d} {doc:5d}/{dic:<6d} {coc:5d}/{cic:<6d} "
              f"{dp:10d} {cp:10d} {dm/1e6:9.2f} {cm/1e6:9.2f} {cut:5.1f}%")

    print()
    print(f"params : {dP/1e6:8.3f} M -> {cP/1e6:8.3f} M   ({100*(1-cP/dP):.1f}% smaller)")
    print(f"MACs   : {dM/1e9:8.3f} G -> {cM/1e9:8.3f} G   ({100*(1-cM/dM):.1f}% fewer)")
    print()
    # --- ZCU102 resource translation -------------------------------------------------------
    for bits, tag in ((8, "INT8 (SmoothQuant deploy)"), (32, "FP32")):
        dk = dP * bits / 1024.0
        ck = cP * bits / 1024.0
        print(f"weight storage @{bits:2d}b {tag:26s}: {dk/8:9.1f} KB -> {ck/8:9.1f} KB   "
              f"BRAM36 {dk/BRAM36_KBIT:7.1f} -> {ck/BRAM36_KBIT:7.1f} "
              f"of {N_BRAM36} ({100*ck/BRAM36_KBIT/N_BRAM36:.1f}% of device)")
    print()
    # DSP: a compacted layer needs proportionally fewer MACs for the same latency target, i.e. the
    # same DSP budget buys 1/(1-cut) more throughput. Report the II-equivalent both ways.
    print(f"DSP (XCZU9EG has {N_DSP48} DSP48E2):")
    for dsp in (256, 512, 1024):
        print(f"  {dsp:5d} DSPs @250MHz -> dense {dM/(dsp*250e6)*1e3:7.1f} ms/frame, "
              f"compact {cM/(dsp*250e6)*1e3:7.1f} ms/frame "
              f"({dM/cM:.2f}x speedup, {1e3/(cM/(dsp*250e6)*1e3):.1f} FPS)")


if __name__ == "__main__":
    main()
