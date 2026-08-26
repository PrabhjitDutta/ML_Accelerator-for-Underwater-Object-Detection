#!/usr/bin/env python3
"""Detection-level C-sim vs PyTorch-oracle check -- the leg-A end-to-end gap.

The per-layer gate (compare_cosine.py) compares TENSORS; csim_eval_map.py compares AGGREGATE mAP over
800 images. Neither answers the question in between: on one image, do the C-sim and the PyTorch integer
oracle emit the same DETECTIONS? Nothing in the harness answered it, because decode_check.py always
anchors its reference on a live FP32 PyTorch forward -- pointing it at an INT8 dump measures
quantization COST (oracle-vs-FP32), not implementation FIDELITY (C-sim-vs-oracle).

This script closes that gap. It reads BOTH head-map sets out of one dumps dir --
  o2o_{0,1,2}_csim.bin    (the C++ C-sim)
  o2o_{0,1,2}_python.bin  (the PyTorch integer oracle)
-- pushes each through the SAME unchanged Yolo26Trunk.decode(), and diffs the two detection lists
against each other. Decode is identical on both branches, so every difference is attributable to the
trunk. That is the same isolation logic decode_check.py uses, with the reference moved from FP32
PyTorch to the oracle.

  conda run -n ueaod python hls/export/compare_dets.py hls/dumps_sq
  conda run -n ueaod python hls/export/compare_dets.py hls/dumps_sq --conf 0.001 --tol 1e-2

Why this matters at all: on the SmoothQuant path the head maps agree at cosine ~0.9995, NOT 1.0. They
genuinely differ. A couple of borderline detections can therefore land on opposite sides of the
confidence threshold, and mAP's 0.0004 aggregate agreement over 800 images cannot see it -- two models
can score identical mAP while disagreeing on every single image.

Detections are paired by IoU, NOT by rank. This is the whole difficulty of the comparison and it is
worth being precise about why, because the neighbouring scripts pair by rank and are right to.

decode_ref.py compares two decodes of the SAME head maps, so scores agree to ~5e-7 and rank order is
stable; there, sorting by score and walking rows in parallel is correct, and its only hazard is an
exact TIE ordered differently by torch.topk vs std::partial_sort (which once manufactured a 633-PIXEL
box difference on img1 out of two identical detection sets merely permuted -- fixed with a geometric
tiebreak in the comparator, not in decode.h).

Here the head maps genuinely DIFFER (cosine ~0.9995), so scores move by up to ~0.09 and the ranking
itself changes. That breaks rank pairing outright, not just at ties: measured on hls/dumps_sq, the two
sides hold the identical 12 objects with boxes agreeing to ~0.3 px, yet rank pairing reports
max|dbox| = 175 px purely because ranks 1<->2 and 5<->6 swapped. A score-sorted comparator cannot fix
this, because the sort key is the very quantity under test.

So detections are matched greedily by IoU (highest first, each detection claimed once). Class
disagreement is REPORTED on matched pairs rather than used for matching, so a genuine class flip
surfaces as a finding instead of silently splitting one object into two unmatched ones.

Unmatched detections are NAMED, never silently truncated. decode_check.py's `n = min(len(a), len(b))`
means that when one side emits fewer detections, the missing one is absent from the diff entirely --
the very case you most want to see (Phase 2: oracle 5, C-sim 4, and the max-abs line said nothing at
all about the detection that vanished).

DEFAULT IS REPORT-ONLY. No --tol means no pass/fail: the expected C-sim-vs-oracle detection divergence
has never been measured, so inventing a threshold before the first measurement would be a gate written
against a guess. Pass --tol to turn it into a gate once you know the floor. A run that compared
NOTHING always fails, in every mode.
"""
import argparse
import os
import sys

import numpy as np

SHAPES = [(8, 80, 80), (8, 40, 40), (8, 20, 20)]

# Resolve repo-relative FIRST, then fall back to the server absolute path. The other export scripts
# hardcode /workspace/... which makes them server-only; this one runs from a checkout too, so the
# comparator can be exercised wherever the dumps and the checkpoint happen to live.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_SERVER = "/workspace/ckarfa/projects/UOD"


def _resolve(rel):
    local = os.path.join(_ROOT, *rel.split("/"))
    return local if os.path.exists(local) else f"{_SERVER}/{rel}"


CKPT = _resolve("final_models/pruned50/yolo26s_urpc2018_pruned50_fp32.pt")
TRUNK_PATH = _resolve("training/yolo26s")


def canon(a):
    """Canonical detection order: score desc, then class, then box geometry.

    The geometric tiebreak is load-bearing -- see the module docstring's 633 px incident.
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
    # decode() splits 2*nl entries (o2m then o2o); the o2m half is unused by the one2one inference
    # decode, so feed the o2o maps into those slots to satisfy the shape contract. Same trick as
    # decode_ref.py -- and safe precisely BECAUSE o2m is dead on this path.
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
    """Greedy highest-IoU-first pairing. Returns (pairs, unmatched_a_idx, unmatched_b_idx).

    Greedy-by-IoU is the standard detection-set correspondence and is deterministic here: pairs are
    ordered by IoU descending with index tiebreaks, so the result does not depend on input order.
    Matching ignores class deliberately -- a class flip should be REPORTED on a matched pair, not
    turned into two unmatched detections that hide the fact that both sides found the same object.
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
    # Rank agreement is reported but NEVER gated: the two sides' scores differ by construction, so a
    # rank swap between two near-equal detections is expected and harmless. It is surfaced only
    # because a rank-pairing comparator would have blamed the box for it.
    rank_swaps = int((ia != ib).sum())

    print(f"\n  matched pairs    : {len(pairs)}  (IoU >= {iou_thr}, worst matched IoU {min_iou:.4f})")
    print(f"  class mismatches : {dcls}/{len(pairs)}")
    print(f"  max |dscore|     : {dscore:.3e}")
    print(f"  max |dbox| (px)  : {dbox:.3e}")
    if rank_swaps:
        print(f"  rank swaps       : {rank_swaps}  (score reordering, not a box error -- not gated)")

    # Name what the pairing could not account for. This is the case decode_check.py's
    # `n = min(len(a), len(b))` drops on the floor (Phase 2: oracle 5, C-sim 4, and the max-abs line
    # said nothing about the detection that vanished).
    for idx, arr, nm in ((ua, a, name_a), (ub, b, name_b)):
        if not idx:
            continue
        print(f"\n  !! {len(idx)} detection(s) in {nm} with NO counterpart above IoU {iou_thr}:")
        for i in idx[:10]:
            print(f"       {fmt(arr[i])}")
        if len(idx) > 10:
            print(f"       ... and {len(idx) - 10} more")
        # The confidence of an unmatched detection separates a threshold artifact from a real bug:
        # near the conf cutoff it is a borderline detection tipping over; far above it, something is
        # actually wrong. Phase 2 assumed the former and never checked; Phase 5 hit the same symptom
        # at conf 0.77 and it was a genuine per-channel FakeQuantize bug.
        print(f"     lowest unmatched confidence: {float(arr[idx][:, 4].min()):.4f}"
              f"   (vs the --conf cutoff: near it => threshold artifact, far above => real bug)")

    if tol is None:
        print("\nREPORT ONLY -- no --tol given, so this is a measurement, not a gate.")
        return True, len(pairs)

    ok = (not ua) and (not ub) and dcls == 0 and dscore <= tol and dbox <= tol
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}   (tol = {tol:g})")
    return ok, len(pairs)


def selftest():
    """Exercise the comparator on synthetic data -- no torch, no dumps, no checkpoint.

    The decode itself is borrowed unchanged from the shipping harness; what is NEW here is the
    comparator, so that is what gets tested. Runs anywhere numpy does.
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

    # 2. a TIE permuted between the two sides must still pass -- this is the 633 px trap
    print()
    perm = base[[0, 2, 1, 3]].copy()
    ok, _ = report(base, perm, "A", "B(tie permuted)", tol=1e-6)
    if not ok:
        fails.append("permuted score tie was reported as a difference (canonical sort is broken)")

    # 3. a RANK SWAP from genuinely shifted scores must still pass. Same objects, same boxes, but the
    # scores moved enough to reorder the list -- measured for real on hls/dumps_sq, where rank pairing
    # turned a 0.3 px agreement into a 175 px "failure". IoU matching must be immune to it.
    print()
    swapped = base.copy()
    swapped[1, 4] = 0.75
    swapped[2, 4] = 0.82
    ok, _ = report(base, swapped, "A", "B(rank swapped)", tol=0.1)
    if not ok:
        fails.append("rank swap from shifted scores was reported as a box error (IoU matching broken)")

    # 3. a real class flip must fail
    print()
    flipped = base.copy()
    flipped[0, 5] = 3
    ok, _ = report(base, flipped, "A", "B(class flipped)", tol=1e-6)
    if ok:
        fails.append("class flip was not caught")

    # 4. a count mismatch must fail AND name the unmatched detection
    print()
    ok, _ = report(base, base[:3].copy(), "A", "B(one short)", tol=1e-6)
    if ok:
        fails.append("count mismatch was not caught")

    # 5. report-only mode must not claim a pass/fail on a real difference
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
