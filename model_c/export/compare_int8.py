#!/usr/bin/env python3
"""INT8 quantization error report: PyTorch fake-quant oracle (dumps_int8) vs the FP32 reference (dumps),
per layer. Not the C++ model gate (that is compare_cosine.py).

  python model_c/export/compare_int8.py

A missing dump is an error; --allow-missing counts it as a skip. Zero comparisons always fails.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FP32 = os.path.join(HERE, "..", "dumps")
INT8 = os.path.join(HERE, "..", "dumps_int8")

LAYERS = [str(i) for i in range(23)]
SUBS = ["l10_cv1", "l10_attn", "l10_psablock"]
HEAD = [f"o2m_{i}" for i in range(3)] + [f"o2o_{i}" for i in range(3)]

def stats(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    # cosine is undefined when either side is all-zero; only two all-zero tensors are "identical".
    cos = float(a @ b / denom) if denom > 0 else (1.0 if not a.any() and not b.any() else 0.0)
    mae = float(np.max(np.abs(a - b))) if a.size else 0.0
    rl2 = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    return cos, mae, rl2

def main():
    allow_missing = "--allow-missing" in sys.argv[1:]
    print(f"{'layer':<14}{'cosine':>12}{'max_abs_err':>14}{'rel_L2':>12}   (INT8 oracle vs FP32)")
    print("-" * 72)
    worst = 1.0
    compared = 0
    missing = []
    for n in LAYERS + SUBS + HEAD:
        pf = os.path.join(FP32, f"{n}_python.bin")
        pi = os.path.join(INT8, f"{n}_python.bin")
        if not (os.path.exists(pf) and os.path.exists(pi)):
            absent = "+".join(s for s, p in (("FP32", pf), ("INT8", pi)) if not os.path.exists(p))
            gated = n in LAYERS or n in HEAD
            if gated:
                missing.append((n, absent))
            print(f"{n:<14}{'(missing)':>12}   no {absent} dump"
                  f"{' **MISSING**' if gated else ' (not in worst)'}")
            continue
        b = np.fromfile(pf, dtype="<f4")
        a = np.fromfile(pi, dtype="<f4")
        cos, mae, rl2 = stats(a, b)
        if n in LAYERS or n in HEAD:
            worst = min(worst, cos)
            compared += 1
        print(f"{n:<14}{cos:>12.6f}{mae:>14.3e}{rl2:>12.3e}")
    print("-" * 72)
    ngated = len(LAYERS) + len(HEAD)
    worst_s = f"{worst:.6f}" if compared else "n/a"
    print(f"worst layer/head cosine (INT8 vs FP32) = {worst_s}   "
          f"layers compared = {compared}/{ngated}")

    # Zero comparisons never passes.
    if compared == 0:
        print(f"ERROR: NO LAYERS WERE COMPARED -- this report is empty, not clean.\n"
              f"  FP32 oracle: {os.path.abspath(FP32)}\n"
              f"  INT8 oracle: {os.path.abspath(INT8)}\n"
              f"  Both need <name>_python.bin per layer; run the export/dump step to produce them.")
        return 2
    if missing and not allow_missing:
        print(f"ERROR: {len(missing)} layer(s) had no dump pair to compare:")
        for n, absent in missing:
            print(f"  {n:<14} no {absent} dump")
        print("  Pass --allow-missing to report them as skips instead.")
        return 1
    if missing:
        print(f"SKIPPED (--allow-missing): {[n for n, _ in missing]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
