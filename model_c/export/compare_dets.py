#!/usr/bin/env python3
"""Detection-level check: decode the C++ model's and the PyTorch oracle's o2o head maps (one dumps dir,
o2o_{0,1,2}_csim.bin / _python.bin) with the same Yolo26Trunk.decode() and diff the detection lists.

  python model_c/export/compare_dets.py <dumps_dir> [--conf 0.001] [--tol 1e-2]

Detections are paired greedily by IoU, not by rank: when the head maps differ slightly, scores move and
the ranking changes, so rank pairing would report false box errors. Class disagreement is reported on
matched pairs; unmatched detections are listed. Without --tol the script only reports; a run that
compared nothing always fails.
"""
import argparse
import os
import sys

import numpy as np

SHAPES = [(8, 80, 80), (8, 40, 40), (8, 20, 20)]

# Resolve paths repo-relative first, then fall back to the server absolute path.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_SERVER = "/workspace/ckarfa/projects/UOD"

def _resolve(rel):
    local = os.path.join(_ROOT, *rel.split("/"))
    return local if os.path.exists(local) else f"{_SERVER}/{rel}"

CKPT = _resolve("final_models/pruned50/yolo26s_urpc2018_pruned50_fp32.pt")
TRUNK_PATH = _resolve("training/yolo26s")

def canon(a):
    """Canonical detection order: score desc, then class, then box geometry (so ties align).
    """
    if not len(a):
        return a
    return a[np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0], a[:, 5], -a[:, 4]))]

def decode_side(dump_dir, suffix, trunk, conf):
    """Run the unchanged Yolo26Trunk.decode on one side's o2o head maps."""
    import torch

    paths = [os.path.join(dump_dir, f"o2o_{i}_{suffix}.bin") for i in range(3)]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    o2o = [np.fromfile(p, dtype="<f4").reshape(1, *s) for p, s in zip(paths, SHAPES)]
    for p, a, s in zip(paths, o2o, SHAPES):
        want = int(np.prod(s))
        if a.size != want:
            raise ValueError(f"{p}: {a.size} floats, expected {want} for shape {s}")
    # decode() expects 2*nl maps (o2m then o2o); o2m is unused by the o2o decode, so reuse the o2o maps.
    maps = [torch.from_numpy(a.copy()) for a in o2o] * 2
    with torch.no_grad():
        det = trunk.decode(tuple(maps))[0].cpu().numpy()
    return det[det[:, 4] >= conf]

def fmt(d):
    return (f"cls{int(d[5])} conf={d[4]:.4f} "
            f"box=[{d[0]:.1f},{d[1]:.1f},{d[2]:.1f},{d[3]:.1f}]")

def iou_matrix(a, b):
    """Pairwise IoU between two [N,6] and [M,6] detection arrays (boxes in cols 0..3, xyxy)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax1, ay1, ax2, ay2 = (a[:, i][:, None].astype(np.float64) for i in range(4))
    bx1, by1, bx2, by2 = (b[:, i][None, :].astype(np.float64) for i in range(4))
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)

def match_by_iou(a, b, iou_thr):
    """Greedy highest-IoU-first pairing, ties broken by index. Returns (pairs, unmatched_a_idx,
    unmatched_b_idx). Class is ignored so a class flip shows up on a matched pair.
    """
    M = iou_matrix(a, b)
    pairs = []
    used_a, used_b = set(), set()
    # sort all candidate pairs by (-iou, i, j) so ties resolve deterministically
    cand = [(-M[i, j], i, j) for i in range(len(a)) for j in range(len(b)) if M[i, j] >= iou_thr]
    for negiou, i, j in sorted(cand):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, -negiou))
    ua = [i for i in range(len(a)) if i not in used_a]
    ub = [j for j in range(len(b)) if j not in used_b]
    return pairs, ua, ub

def report(a, b, name_a, name_b, tol, iou_thr=0.5):
    """Diff two detection sets by IoU correspondence. Returns (ok, n_matched)."""
    a, b = canon(a), canon(b)
    print(f"{name_a:<24} {len(a)} dets")
    print(f"{name_b:<24} {len(b)} dets")
    if not len(a) or not len(b):
        print("\nERROR: one side produced NO detections -- nothing was compared. NOT A PASS.")
        return False, 0

    pairs, ua, ub = match_by_iou(a, b, iou_thr)
    if not pairs:
        print(f"\nERROR: no detection pair reached IoU >= {iou_thr} -- the two sets share nothing. "
              f"NOT A PASS.")
        return False, 0

    ia = np.array([p[0] for p in pairs])
    ib = np.array([p[1] for p in pairs])
    dcls = int((a[ia, 5] != b[ib, 5]).sum())
    dscore = float(np.abs(a[ia, 4] - b[ib, 4]).max())
    dbox = float(np.abs(a[ia, :4] - b[ib, :4]).max())
    min_iou = min(p[2] for p in pairs)
    # Rank agreement is reported, never gated: near-equal scores may swap.
    rank_swaps = int((ia != ib).sum())

    print(f"\n  matched pairs    : {len(pairs)}  (IoU >= {iou_thr}, worst matched IoU {min_iou:.4f})")
    print(f"  class mismatches : {dcls}/{len(pairs)}")
    print(f"  max |dscore|     : {dscore:.3e}")
    print(f"  max |dbox| (px)  : {dbox:.3e}")
    if rank_swaps:
        print(f"  rank swaps       : {rank_swaps}  (score reordering, not a box error -- not gated)")

    # List detections the pairing could not match.
    for idx, arr, nm in ((ua, a, name_a), (ub, b, name_b)):
        if not idx:
            continue
        print(f"\n  !! {len(idx)} detection(s) in {nm} with NO counterpart above IoU {iou_thr}:")
        for i in idx[:10]:
            print(f"       {fmt(arr[i])}")
        if len(idx) > 10:
            print(f"       ... and {len(idx) - 10} more")
        # An unmatched detection near the conf cutoff is a borderline flip; far above it, a real bug.
        print(f"     lowest unmatched confidence: {float(arr[idx][:, 4].min()):.4f}"
              f"   (vs the --conf cutoff: near it => threshold artifact, far above => real bug)")

    if tol is None:
        print("\nREPORT ONLY -- no --tol given, so this is a measurement, not a gate.")
        return True, len(pairs)

    ok = (not ua) and (not ub) and dcls == 0 and dscore <= tol and dbox <= tol
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}   (tol = {tol:g})")
    return ok, len(pairs)

def selftest():
    """Self-test of the comparator on synthetic data (numpy only).
    """
    base = np.array([
        [10., 10., 50., 50., 0.90, 1],
        [20., 20., 60., 60., 0.80, 2],
        [30., 30., 70., 70., 0.80, 2],   # deliberate score tie with the row above
        [40., 40., 80., 80., 0.30, 0],
    ], dtype=np.float32)
    fails = []

    # 1. identical input must compare clean
    ok, n = report(base, base.copy(), "A", "B", tol=1e-6)
    if not (ok and n == 4):
        fails.append("identical inputs did not pass")

    # 2. a tie permuted between the two sides must pass
    print()
    perm = base[[0, 2, 1, 3]].copy()
    ok, _ = report(base, perm, "A", "B(tie permuted)", tol=1e-6)
    if not ok:
        fails.append("permuted score tie was reported as a difference (canonical sort is broken)")

    # 3. a rank swap from shifted scores must pass (same boxes)
    print()
    swapped = base.copy()
    swapped[1, 4] = 0.75
    swapped[2, 4] = 0.82
    ok, _ = report(base, swapped, "A", "B(rank swapped)", tol=0.1)
    if not ok:
        fails.append("rank swap from shifted scores was reported as a box error (IoU matching broken)")

    # 4. a class flip must fail
    print()
    flipped = base.copy()
    flipped[0, 5] = 3
    ok, _ = report(base, flipped, "A", "B(class flipped)", tol=1e-6)
    if ok:
        fails.append("class flip was not caught")

    # 5. a count mismatch must fail and name the unmatched detection
    print()
    ok, _ = report(base, base[:3].copy(), "A", "B(one short)", tol=1e-6)
    if ok:
        fails.append("count mismatch was not caught")

    # 6. report-only mode must not claim pass/fail
    print()
    ok, _ = report(base, flipped, "A", "B(class flipped)", tol=None)
    if not ok:
        fails.append("report-only mode should not fail on differences")

    print("\n" + "=" * 60)
    if fails:
        for f in fails:
            print(f"SELFTEST FAIL: {f}")
        return 1
    print("SELFTEST PASS -- comparator handles ties, class flips, and count mismatches.")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", nargs="?", default=None,
                    help="dumps dir holding BOTH o2o_*_csim.bin and o2o_*_python.bin")
    ap.add_argument("--a", default="csim", help="suffix for side A (default: csim)")
    ap.add_argument("--b", default="python", help="suffix for side B (default: python)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence cutoff (default 0.25, the deployment threshold)")
    ap.add_argument("--tol", type=float, default=None,
                    help="turn the report into a gate at this tolerance on score and box")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU threshold for pairing detections between the two sides (default 0.5)")
    ap.add_argument("--selftest", action="store_true",
                    help="exercise the comparator on synthetic data (no torch/dumps needed)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.dump_dir:
        ap.error("dump_dir is required (or pass --selftest)")

    sys.path.insert(0, TRUNK_PATH)
    from yolo26_trunk import Yolo26Trunk  # noqa: E402
    from ultralytics import YOLO  # noqa: E402

    trunk = Yolo26Trunk(YOLO(CKPT).model).float().eval()
    try:
        a = decode_side(args.dump_dir, args.a, trunk, args.conf)
        b = decode_side(args.dump_dir, args.b, trunk, args.conf)
    except FileNotFoundError as e:
        print(f"ERROR: missing head map: {e}\n"
              f"  This gate needs BOTH o2o_*_{args.a}.bin and o2o_*_{args.b}.bin in\n"
              f"  {os.path.abspath(args.dump_dir)}\n"
              f"  A compacted dumps dir holds only C-sim tensors and has no oracle by construction.\n"
              f"  NOT A PASS.")
        sys.exit(1)

    print(f"dumps: {os.path.abspath(args.dump_dir)}   conf >= {args.conf}\n")
    ok, n = report(a, b, f"[{args.a}]", f"[{args.b}]", args.tol, args.iou)
    sys.exit(0 if (ok and n) else 2)

if __name__ == "__main__":
    main()
