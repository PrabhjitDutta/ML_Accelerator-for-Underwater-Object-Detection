#!/usr/bin/env python3
"""Lossless gate for Phase 4: the channel-compacted C-sim vs the dense C-sim.

Compaction only drops channels that are identically zero, so the compacted graph must reproduce the
dense one EXACTLY -- not approximately. Intermediate layer dumps have fewer channels now, so each is
scattered back into its dense channel slots (compaction_map.json["dumps"]) and the dropped slots are
asserted to be zero in the dense reference. The head maps (o2m_*/o2o_*) are full width and compare
directly; those are the ones that actually feed decode.

  conda run -n ueaod python hls/export/compare_compact.py [dense_dumps] [compact_dumps] [map.json]

A tensor missing from either side is a HARD ERROR, not a silent skip -- a run that compared nothing
must never print PASS. Pass --allow-missing to downgrade absent tensors to counted skips (the
-DYOLO26_NO_O2M deploy build legitimately emits no o2m_* dumps); even then, zero comparisons fails.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(d, name):
    p = os.path.join(d, f"{name}_csim.bin")
    return np.fromfile(p, dtype="<f4") if os.path.exists(p) else None


def main():
    argv = [a for a in sys.argv[1:] if a != "--allow-missing"]
    allow_missing = len(argv) != len(sys.argv) - 1
    dense = argv[0] if len(argv) > 0 else os.path.join(HERE, "..", "dumps")
    comp = argv[1] if len(argv) > 1 else os.path.join(HERE, "..", "dumps_compact")
    mpath = argv[2] if len(argv) > 2 else os.path.join(HERE, "..", "weights_compact",
                                                       "compaction_map.json")
    tol = float(argv[3]) if len(argv) > 3 else 0.0
    cmap = json.load(open(mpath))
    keep = cmap["dumps"]

    names = [str(i) for i in range(23)] + [f"o2m_{i}" for i in range(3)] + \
            [f"o2o_{i}" for i in range(3)]
    bad = 0
    compared = 0
    missing = []
    print(f"{'tensor':10s} {'dense C':>8s} {'comp C':>7s} {'max|d|':>10s} {'dropped max':>12s}  status")
    for n in names:
        a, b = load(dense, n), load(comp, n)
        if a is None or b is None:
            absent = "+".join(s for s, t in (("dense", a), ("compact", b)) if t is None)
            missing.append((n, absent))
            print(f"{n:10s} {'-':>8s} {'-':>7s} {'-':>10s} {'-':>12s}  MISSING (no {absent} dump)")
            continue
        compared += 1
        idx = keep.get(n)
        if idx is None:                      # full-width tensor (head maps): direct compare
            if a.size != b.size:
                print(f"{n:10s} size mismatch {a.size} vs {b.size}"); bad += 1
                continue
            d = float(np.abs(a - b).max())
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            ok = d <= tol
            print(f"{n:10s} {'-':>8s} {'-':>7s} {d:10.3e} {'cos=%.9f' % cos:>12s}  {'OK' if ok else 'DIFF'}")
            bad += (not ok)
            continue
        C = a.size // (b.size // len(idx)) if idx else 0
        HW = b.size // len(idx)
        A = a.reshape(-1, HW)
        B = b.reshape(-1, HW)
        assert B.shape[0] == len(idx), (n, B.shape, len(idx))
        d = float(np.abs(A[idx] - B).max())
        dropped = np.setdiff1d(np.arange(A.shape[0]), idx)
        dm = float(np.abs(A[dropped]).max()) if dropped.size else 0.0
        ok = (d <= tol and dm == 0.0)
        bad += (not ok)
        print(f"{n:10s} {A.shape[0]:8d} {B.shape[0]:7d} {d:10.3e} {dm:12.3e}  {'OK' if ok else 'DIFF'}")

    print()
    print(f"[compact-gate] tensors compared = {compared}/{len(names)}")

    # A gate that compared nothing must never pass, in any mode -- not even with --allow-missing.
    if compared == 0:
        print(f"[compact-gate] ERROR - NO TENSORS WERE COMPARED, nothing was verified. NOT A PASS.\n"
              f"  dense:   {os.path.abspath(dense)}\n"
              f"  compact: {os.path.abspath(comp)}\n"
              f"  Both dirs need <name>_csim.bin per tensor; re-run build_csim.sh to produce them.")
        return 2
    if missing and not allow_missing:
        print(f"[compact-gate] FAIL - {len(missing)} tensor(s) could not be compared at all:")
        for n, absent in missing:
            print(f"  {n:10s} no {absent} dump")
        print("  An uncompared tensor is a gate failure, not a skip. Re-run the C-sim, or pass\n"
              "  --allow-missing to count them as reported skips (the -DYOLO26_NO_O2M deploy build\n"
              "  emits no o2m_* dumps).")
        return 1
    if bad == 0:
        kind = "BIT-IDENTICAL" if tol == 0 else f"within {tol:g}"
        skipped = f" ({len(missing)} skipped: {[n for n, _ in missing]})" if missing else ""
        print(f"[compact-gate] PASS - {compared} compared tensor(s){skipped}: compacted C-sim is "
              f"{kind} vs dense, and every dropped channel was exactly zero in the dense reference.")
    else:
        print(f"[compact-gate] FAIL - {bad} tensor(s) differ (missed add-union or concat-offset bug).")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
