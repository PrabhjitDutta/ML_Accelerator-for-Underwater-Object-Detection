"""frames_to_u8.py - input.bin (3*640*640 float32, k/255) -> .u8 (the bytes k), for the frame loop (run_model.cpp).

    python3 frames_to_u8.py frames/*.bin      # writes frames/<stem>.u8 beside each; fails if any value is not k/255.f

The loader turns byte k back into (float)k / 255.f, so this refuses any file it could not reproduce bit for bit.
Standard library only.
"""
import struct
import sys
from fractions import Fraction


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def div255(k):
    """(float)k / 255.f as float32 bits: the float32 nearest k/255 (ties to even), as IEEE division gives."""
    b = struct.unpack("<I", struct.pack("<f", k / 255.0))[0]
    exact = Fraction(k, 255)
    return min((b - 1, b, b + 1) if b else (b, b + 1),
               key=lambda c: (abs(Fraction(f32(c)) - exact), c & 1))


TAB = {div255(k): k for k in range(256)}   # float32 bits -> k

for p in sys.argv[1:]:
    raw = open(p, "rb").read()
    if len(raw) != 3 * 640 * 640 * 4:
        sys.exit("%s: %d bytes, expected 3*640*640 float32" % (p, len(raw)))
    try:
        out = bytes(TAB[b] for b in struct.unpack("<%dI" % (len(raw) // 4), raw))
    except KeyError:
        sys.exit("%s: not exactly k/255.f - keep the .bin" % p)
    q = p[:-4] + ".u8" if p.endswith(".bin") else p + ".u8"
    open(q, "wb").write(out)
    print(p, "->", q)
