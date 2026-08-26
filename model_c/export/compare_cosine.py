#!/usr/bin/env python3
"""Per-layer parity check: C-sim <name>_csim.bin vs PyTorch <name>_python.bin.

For every dumped tensor, report cosine similarity, max absolute error, and relative L2 error. FP32
parity means cosine ~ 1.0000 for every layer; the milestone gate is cosine >= 0.9999 AND
rel_L2 <= 1e-5 for all of layers 0..22 plus the 6 head maps.

TWO metrics gate, not one. Cosine is scale-INVARIANT: it measures the angle between the tensors and
nothing else, so a C-sim output that is a uniform multiple of the reference (a = k*b, k > 0) scores
cosine = 1.000000 exactly, for ANY k. A dropped or doubled dequant scale is precisely that bug, and a
cosine-only gate certifies it as perfect. rel_L2 closes the hole: for a = k*b it equals |k-1|, so a
2x scale error reads 1.0 against a 1e-5 gate -- five orders of magnitude of margin.

rel_L2 was chosen over max_abs_err for the second gate because it is already normalized by the
reference norm. Across the FP32 dumps max_abs_err spans 48x layer-to-layer (it tracks activation
magnitude, so one threshold cannot fit every layer) while rel_L2 spans only 3.8x. Observed worst
rel_L2 on the FP32 path is 1.269e-06, so the 1e-5 gate passes every real layer with ~8x headroom.

  conda run -n ueaod python hls/export/compare_cosine.py

A missing reference is a HARD ERROR, not a skip: this gate exists to certify layers, and a run that
compared nothing must never print PASS. The compact dumps dirs (dumps_*_compact*) hold only C-sim
tensors and are gated by compare_compact.py against the dense C-sim -- pointing this script at one is
a mistake, and it now says so. Pass --allow-missing to downgrade absent tensors to counted skips (the
-DYOLO26_NO_O2M deploy build legitimately emits no o2m_* dumps); even then, zero comparisons fails.
"""
import math
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
argv = [a for a in sys.argv[1:] if a != "--allow-missing"]
ALLOW_MISSING = len(argv) != len(sys.argv) - 1
# default FP32 dumps dir; pass an alternate (e.g. ../dumps_int8) as argv[1] to gate the INT8 run
DUMPS = argv[0] if argv else os.path.join(HERE, "..", "dumps")

# ordered names: backbone/neck layers, attention sub-outputs, head maps
LAYERS = [str(i) for i in range(23)]
SUBS = ["l10_cv1", "l10_attn", "l10_psablock"]
HEAD = [f"o2m_{i}" for i in range(3)] + [f"o2o_{i}" for i in range(3)]
# gate: FP32 parity is 0.9999; for the INT8 integer C-sim pass a looser gate as argv[2] (the
# convolutional dataflow is bit-exact, but the FP32 attention matmuls + SiLU exp differ from torch
# at the ULP level and get amplified at INT8 rounding boundaries -- see hls README / notes).
GATE = float(argv[1]) if len(argv) > 1 else 0.9999

# rel_L2 gate (argv[3]). The default is 1e-5 only on the FP32-parity path, where the observed floor is
# ~1e-6 and the gate is a real precision bound. Loosening the cosine gate means real arithmetic
# divergence is expected, and rel_L2 must loosen with it or every INT8 run fails on a bound written for
# a different regime: for a pure rotation rel_L2 ~ sqrt(2*(1-cos)), so the derived default is that with
# a 2x safety factor. Be honest about what that buys -- on the INT8 path rel_L2 is a BACKSTOP against a
# grossly wrong scale, not a precision gate, because genuine quantization noise dominates the budget.
# Pass an explicit value when you want a tighter bound than the derived one.
RL2_GATE = (float(argv[2]) if len(argv) > 2
            else (1e-5 if GATE >= 0.9999 else 2.0 * math.sqrt(2.0 * (1.0 - GATE))))


def stats(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    # cosine is undefined when either side is all-zero. Only two all-zero tensors are "identical";
    # an all-zero C-sim against a live reference is a total failure, so score it 0.0 -- returning
    # 1.0 there would let a dead kernel pass the gate.
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
        # Both must hold. cos alone cannot see a uniform scale error; rel_L2 alone is blind to nothing
        # cosine catches, but is noisier -- so report WHICH one failed rather than a bare FAIL.
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

    # A gate that compared nothing must never pass, in any mode -- not even with --allow-missing.
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
