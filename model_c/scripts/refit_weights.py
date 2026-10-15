#!/usr/bin/env python3
"""One grid per shared tensor, on a C-model weights folder (notes/onlyPL.md, stage 2).

A conv that reads one tensor which a concat reader also reads adopts the concat reader's quantizer for those
channels (ssc slice, lo, step). Its weights are rescaled per input channel and re-rounded to int8:
    real weight on x_i   Wf[o,i] = wsc[o] * w[o,i] * ssc[i]
    on the owner's grid  V[o,i]  = Wf[o,i] / ssc_owner[off+i],  wsc'[o] = max|V[o]|/127,  w' = round(V/wsc')
The manifest line keeps its shape; sa and lo become the owner's, and a per-channel (perch) reader becomes scalar.

  python refit_weights.py <weights in> <weights out>
"""
import os
import shutil
import sys
import numpy as np

SRC, DST = sys.argv[1], sys.argv[2]
shutil.copytree(SRC, DST)                      # fails if DST exists: never refit a folder twice
man = [l.split() for l in open(os.path.join(SRC, "manifest_sq.txt")) if l.strip()]
M = {l[0]: l for l in man}
ic_of = lambda n: int(M[n][2])
f32 = lambda n, ext: np.fromfile(os.path.join(SRC, f"{n}.{ext}.bin"), dtype="<f4")

# (reader, owner, owner's channel offset): "last" = the reader's tensor is the owner's last input
PAIRS = []
for i in (2, 4):
    PAIRS.append((f"{i}.m.0.cv1.conv", f"{i}.cv2.conv", "split"))
for i in (6, 8, 13, 16, 19):
    PAIRS += [(f"{i}.m.0.cv1.conv", f"{i}.cv2.conv", "split"), (f"{i}.m.0.cv2.conv", f"{i}.cv2.conv", "split")]
PAIRS += [("22.m.0.0.cv1.conv", "22.cv2.conv", "split"), ("5.conv", "16.cv1.conv", "last"), ("7.conv", "13.cv1.conv", "last")]
for i, own in ((0, "17.conv"), (1, "20.conv")):
    PAIRS += [(f"23.one2one_cv2.{i}.0.conv", own, 0), (f"23.one2one_cv3.{i}.0.0.conv", own, 0)]
PAIRS.append(("23.one2one_cv3.2.0.0.conv", "23.one2one_cv2.2.0.conv", 0))
splits = dict(map(int, l.split()) for l in open(os.path.join(SRC, "splits.txt")))

worst = 0.0
for r, o, off in PAIRS:
    L, LO = M[r], M[o]
    oc, ic, kh, kw, groups = int(L[1]), int(L[2]), int(L[3]), int(L[4]), int(L[9])
    assert L[10] == "1" and LO[10] == "1" and LO[14] == "0", (r, o)
    off = splits[int(r.split(".")[0])] if off == "split" else ic_of(o) - ic if off == "last" else off
    assert 0 <= off and off + ic <= ic_of(o), (r, o, off)
    icg = ic // groups
    w = f32(r, "w").reshape(oc, icg, kh, kw).astype(np.float64)
    ch = (np.arange(oc)[:, None] // (oc // groups)) * icg + np.arange(icg)[None, :]      # input channel of w[o, j]
    ssc_new = f32(o, "ssc")[off:off + ic]
    v = w * f32(r, "sw").astype(np.float64)[:, None, None, None] \
        * (f32(r, "ssc").astype(np.float64) / ssc_new.astype(np.float64))[ch][:, :, None, None]
    wsc = np.abs(v).reshape(oc, -1).max(1) / 127.0
    wsc[wsc == 0] = 1.0
    w2 = np.round(v / wsc[:, None, None, None])
    assert np.abs(w2).max() <= 127
    worst = max(worst, float(np.abs(w2 * wsc[:, None, None, None] - v).max() / np.abs(v).max()))
    w2.astype("<f4").tofile(os.path.join(DST, f"{r}.w.bin"))
    wsc.astype("<f4").tofile(os.path.join(DST, f"{r}.sw.bin"))
    ssc_new.astype("<f4").tofile(os.path.join(DST, f"{r}.ssc.bin"))
    for ext in ("sa", "lo"):                       # a per-channel reader becomes scalar
        p = os.path.join(DST, f"{r}.{ext}.bin")
        if os.path.exists(p):
            os.remove(p)
    L[12], L[13] = LO[12], LO[13]
    if len(L) > 14:
        L[14] = "0"
with open(os.path.join(DST, "manifest_sq.txt"), "w", newline="\n") as f:
    f.writelines(" ".join(l) + "\n" for l in man)
print(f"refitted {len(PAIRS)} convs -> {DST}; largest weight rounding error {worst:.4f} of the conv's largest weight")
