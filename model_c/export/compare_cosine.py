#!/usr/bin/env python3
"""Per-layer check: C++ <name>_csim.bin vs PyTorch <name>_python.bin (cosine, max abs error, relative L2).

Both cosine and rel_L2 gate: cosine is scale-invariant (a = k*b scores 1.0), so a wrong dequant scale
passes it; rel_L2 = |k-1| catches that.

  python model_c/export/compare_cosine.py [dumps_dir] [cos_gate] [rel_l2_gate]

A missing reference is an error; --allow-missing counts it as a skip. Zero comparisons always fails.
"""
import math
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
argv = [a for a in sys.argv[1:] if a != "--allow-missing"]
ALLOW_MISSING = len(argv) != len(sys.argv) - 1
# default FP32 dumps dir
DUMPS = argv[0] if argv else os.path.join(HERE, "..", "dumps")

# ordered names: backbone/neck layers, attention sub-outputs, head maps
LAYERS = [str(i) for i in range(23)]
SUBS = ["l10_cv1", "l10_attn", "l10_psablock"]
HEAD = [f"o2m_{i}" for i in range(3)] + [f"o2o_{i}" for i in range(3)]
# cosine gate: 0.9999 for FP32; pass a looser one for INT8 runs.
GATE = float(argv[1]) if len(argv) > 1 else 0.9999

# rel_L2 gate: 1e-5 on the FP32 path; with a looser cosine gate it is derived from it (rel_L2 ~
# sqrt(2*(1-cos)) for a rotation, x2), a backstop against a wrong scale rather than a precision bound.
RL2_GATE = (float(argv[2]) if len(argv) > 2
            else (1e-5 if GATE >= 0.9999 else 2.0 * math.sqrt(2.0 * (1.0 - GATE))))

def stats(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    # cosine is undefined for an all-zero side: two all-zero tensors match, otherwise score 0.
    cos = float(a @ b / denom) if denom > 0 else (1.0 if not a.any() and not b.any() else 0.0)
    mae = float(np.max(np.abs(a - b))) if a.size else 0.0
    rl2 = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    return cos, mae, rl2

def main():
    names = LAYERS + SUBS + HEAD
    print(f"{'layer':<14}{'cosine':>12}{'max_abs_err':>14}{'rel_L2':>12}   status")
    print("-" * 68)
    worst = 1.0
    worst_rl2 = 0.0
    fails = []
    missing = []
    gated_compared = 0
    for n in names:
        pp = os.path.join(DUMPS, f"{n}_python.bin")
        cp = os.path.join(DUMPS, f"{n}_csim.bin")
        if not (os.path.exists(pp) and os.path.exists(cp)):
            absent = [s for s, p in (("python", pp), ("csim", cp)) if not os.path.exists(p)]
            gated = n in LAYERS or n in HEAD
            if gated:
                missing.append((n, "+".join(absent)))
            print(f"{n:<14}{'(missing)':>12}{'':>14}{'':>12}   "
                  f"{'no ' + '+'.join(absent) + ' dump' + (' **MISSING**' if gated else ' (ungated)')}")
            continue
        a = np.fromfile(cp, dtype="<f4")
        b = np.fromfile(pp, dtype="<f4")
        if a.size != b.size:
            print(f"{n:<14}  SIZE MISMATCH csim={a.size} py={b.size}")
            fails.append((n, "size"))
            continue
        cos, mae, rl2 = stats(a, b)
        gated = n in LAYERS or n in HEAD
        # Both must hold; report which one failed.
        bad = [w for w, good in (("cos", cos >= GATE), ("rel_L2", rl2 <= RL2_GATE)) if not good]
        if gated:
            worst = min(worst, cos)
            worst_rl2 = max(worst_rl2, rl2)
            gated_compared += 1
            if bad:
                fails.append((n, "+".join(bad)))
        flag = "OK" if not bad else "**FAIL " + "+".join(bad) + "**"
        print(f"{n:<14}{cos:>12.6f}{mae:>14.3e}{rl2:>12.3e}   {flag}")
    print("-" * 68)
    ngated = len(LAYERS) + len(HEAD)
    worst_s = f"{worst:.6f}" if gated_compared else "n/a"
    worst_rl2_s = f"{worst_rl2:.3e}" if gated_compared else "n/a"
    print(f"worst gated cosine = {worst_s}   gate = {GATE}")
    print(f"worst gated rel_L2 = {worst_rl2_s}   gate = {RL2_GATE:.3e}"
          + ("" if len(argv) > 2 else "  (derived from the cosine gate)"))
    print(f"gated layers compared = {gated_compared}/{ngated}")

    # Zero comparisons never passes, even with --allow-missing.
    if gated_compared == 0:
        print(f"ERROR: NO GATED LAYERS WERE COMPARED -- nothing was verified.\n"
              f"  dumps dir: {os.path.abspath(DUMPS)}\n"
              f"  This gate needs BOTH <name>_csim.bin and <name>_python.bin per layer. If that dir\n"
              f"  holds only *_csim.bin it is a compacted/dense C-sim dump, which has no PyTorch\n"
              f"  oracle by construction -- gate it with compare_compact.py against the dense C-sim\n"
              f"  instead. NOT A PASS.")
        sys.exit(2)
    if missing and not ALLOW_MISSING:
        print(f"ERROR: {len(missing)} gated tensor(s) have no reference to compare against:")
        for n, absent in missing:
            print(f"  {n:<14} no {absent} dump")
        print("  A gated layer that was never compared is a gate failure, not a skip. Re-run the\n"
              "  C-sim/oracle to produce the dumps, or pass --allow-missing to count them as\n"
              "  reported skips (e.g. the -DYOLO26_NO_O2M deploy build emits no o2m_* dumps).")
        sys.exit(1)
    if fails:
        print("FAILED layers (metric that failed):")
        for n, why in fails:
            print(f"  {n:<14} {why}")
        sys.exit(1)
    if missing:
        print(f"SKIPPED (--allow-missing): {[n for n, _ in missing]}")
    print(f"ALL {gated_compared} COMPARED GATED LAYERS PASS"
          + (f" ({len(missing)} skipped)" if missing else ""))

if __name__ == "__main__":
    main()
