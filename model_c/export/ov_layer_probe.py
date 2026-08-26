#!/usr/bin/env python3
"""Localize where the C-sim/oracle diverges from the deployed OpenVINO model, per layer.

Node names in the IR do NOT line up with our dump names by construction: our dumps are *module*
outputs (a C3k2's output includes its cv2; SPPF's includes the shortcut add), while IR nodes are
individual ops. Guessing the mapping is how you end up "measuring" layer 9 at cosine 0.29 and
concluding the model is broken when you actually just probed the wrong tensor.

So this matches empirically: expose every plausible activation node, then for each of our dumps pick
the OV tensor of the same shape with the highest cosine. That both establishes the mapping and gives
the divergence. A layer whose BEST match is still poor is a genuine divergence; a layer with no
same-shape candidate is reported as such rather than silently skipped.

  conda run -n ueaod python hls/export/ov_layer_probe.py [dumps_dir] [suffix]
"""
import os
import sys

import numpy as np
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
IRXML = ("/workspace/ckarfa/projects/UOD/final_models/pruned50/"
         "int8_smoothquant_openvino_model/best.xml")
DUMPS = os.path.join(HERE, "..", "dumps")

# Op types that carry a real activation tensor worth comparing against.
CAND = ("Add", "Multiply", "Swish", "Concat", "MaxPool", "Interpolate", "Convolution",
        "GroupConvolution", "MatMul", "SoftMax")


def cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else float("nan")


def main():
    ddir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "dumps_sq")
    suffix = sys.argv[2] if len(sys.argv) > 2 else "python"

    core = ov.Core()
    model = core.read_model(IRXML)
    ports, pnames = [], []
    for op in model.get_ops():
        if op.get_type_name() in CAND:
            try:
                o = op.output(0)
                if len(o.get_partial_shape()) == 4 and o.get_partial_shape().is_static:
                    ports.append(o)
                    pnames.append(op.get_friendly_name())
            except Exception:
                pass
    model.add_outputs(ports)
    cm = core.compile_model(model, "CPU")
    x = np.fromfile(os.path.join(DUMPS, "input.bin"), dtype="<f4").reshape(1, 3, 640, 640)
    res = cm(x)
    # add_outputs appends in order, so the last len(ports) compiled outputs correspond 1:1.
    outs = list(cm.outputs)[-len(ports):]
    ovt = {pnames[i]: np.array(res[outs[i]]) for i in range(len(ports))}
    byshape = {}
    for n, a in ovt.items():
        byshape.setdefault(a.shape[1:], []).append(n)
    print(f"[probe] exposed {len(ovt)} static 4-D OV activations")

    names = [str(i) for i in range(23)] + ["l10_cv1", "l10_attn", "l10_psablock"]
    print(f"\n{'dump':14s} {'shape':>18s} {'best cosine':>12s}  best-matching IR node")
    print("-" * 100)
    for nm in names:
        p = os.path.join(ddir, f"{nm}_{suffix}.bin")
        if not os.path.exists(p):
            continue
        a = np.fromfile(p, dtype="<f4")
        cands = [(k, v) for shp, ns in byshape.items() if int(np.prod(shp)) == a.size
                 for k in ns for v in [ovt[k]]]
        if not cands:
            print(f"{nm:14s} {'size=%d' % a.size:>18s} {'-':>12s}  <no same-size OV tensor>")
            continue
        best, bc = max(((k, cos(a, v)) for k, v in cands), key=lambda t: t[1])
        shp = ovt[best].shape
        print(f"{nm:14s} {str(tuple(shp[1:])):>18s} {bc:12.6f}  {best}")


if __name__ == "__main__":
    main()
