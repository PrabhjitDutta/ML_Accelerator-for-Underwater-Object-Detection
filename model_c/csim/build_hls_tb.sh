#!/usr/bin/env bash
# Build and run the two HLS-kernel gates. Companion to build_csim.sh, which covers the golden C-sim.
#
# These were originally built ad hoc, which is exactly the failure build_csim.sh warns about for
# yolo26_csim_fast: on 2026-08-09 tb_s1.exe was 4 minutes OLDER than all three of its sources, so the
# recorded "100/100 bit-exact" PASS came from a binary that did not contain the final edits. It
# happened to still hold (the edits were confined to the Y26_FX_DEQUANT branch and to reporting), but
# a stale gate silently certifies whatever the code used to do. Always rebuild via this script.
#
#   bash hls/csim/build_hls_tb.sh [weights_dir] [spatial]
#
# All three gates read the SAME source (yolo26_hls_tb.cpp); the only difference is the defines, which
# also flip the gate's bar:
#   tb_s1    - float dequant    -> EXACT equality vs conv2d(). Any mismatch is a real bug.
#   tb_s2    - ap_fixed dequant -> tolerance + cosine, because the dequant is genuinely re-rounded.
#   tb_s2lut - + fixed-point SiLU LUT, replacing std::exp(double). Same tolerance bar.
#
# These own the project's EXHAUSTIVE coverage: all 100 integer convs in seconds, and tb_s1 is the
# only place the bit-exact bar is enforced anywhere. Vitis csim runs the same source but is ~200x
# slower (-fhls-csim instrumentation), so run_hls_zu9eg.tcl caps it at a handful of convs. Do not
# let that cap migrate here.
# Binaries land in build/ so the source directory stays sources-only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p build

# The real ap_int/ap_fixed headers ship with Vitis -- NOT htdet's hls_stubs/, which silently changes
# the arithmetic these gates exist to check. Override for a non-default install or a Linux host.
VITIS_INC="${VITIS_INC:-C:/AMDDesignTools/2026.1/Vitis/include}"
if [ ! -f "$VITIS_INC/ap_int.h" ]; then
    echo "[build] ERROR: no ap_int.h under $VITIS_INC -- set VITIS_INC to your Vitis include dir." >&2
    exit 2
fi

# Under MSYS2 the produced .exe needs the ucrt64 runtime DLLs on PATH or it exits 127 with no message.
case "$(uname -s)" in MINGW*|MSYS*) export PATH="/c/msys64/ucrt64/bin:$PATH";; esac

# Extra defines, for sweeping a compile-time parameter through the gates:
#   Y26_EXTRA_FLAGS="-DY26_LANES=32" bash hls/csim/build_hls_tb.sh
# This exists because a LANE-COUNT CHANGE IS A CORRECTNESS CHANGE, not just a resource knob: it moves
# the bank assignment (ic & (Y26_LANES-1)) and the partial-tile zero-fill in the MAC pass. Never
# synthesize or place-and-route a new Y26_LANES without running tb_s1 at that value first - a
# bit-exact failure would otherwise show up as a plausible-looking utilization number for a design
# that computes the wrong answer. Y26_LANES must stay a POWER OF TWO (the banking relies on the
# mask); the old "must be prime" invariant applied to the abandoned cyclic-partition scheme.
Y26_EXTRA_FLAGS="${Y26_EXTRA_FLAGS:-}"
[ -n "$Y26_EXTRA_FLAGS" ] && echo "[build] extra flags: $Y26_EXTRA_FLAGS"
CXXFLAGS=(-O2 -std=c++14 -fopenmp -I"$VITIS_INC" $Y26_EXTRA_FLAGS)
# yolo26_hls_top.cpp is in the list because the tb now calls y26_conv_top() rather than
# y26_conv2d_hls() directly -- see the comment at that call site. The top is a pure forwarder, so
# this does not change what the gates measure; it makes the SAME tb usable for HLS co-simulation,
# which can only trace arguments at the synthesis boundary. g++ ignores its #pragma HLS lines.
SRC=(yolo26_hls_tb.cpp yolo26_hls.cpp yolo26_hls_top.cpp)

echo "[build] compiling tb_s1 (stage-1, float dequant, bit-exact bar) ..."
g++ "${CXXFLAGS[@]}" "${SRC[@]}" -o build/tb_s1.exe
echo "[build] compiling tb_s2 (stage-2, ap_fixed dequant, tolerance bar) ..."
g++ "${CXXFLAGS[@]}" -DY26_FX_DEQUANT "${SRC[@]}" -o build/tb_s2.exe
echo "[build] compiling tb_s2lut (stage-2 + fixed-point SiLU LUT, tolerance bar) ..."
g++ "${CXXFLAGS[@]}" -DY26_FX_DEQUANT -DY26_SILU_LUT "${SRC[@]}" -o build/tb_s2lut.exe

WEIGHTS="${1:-../weights_sq_compact_fold}"
SPATIAL="${2:-16}"
echo "[build] running gates on $WEIGHTS at ${SPATIAL}x${SPATIAL} ..."
./build/tb_s1.exe    "$WEIGHTS" "$SPATIAL"
./build/tb_s2.exe    "$WEIGHTS" "$SPATIAL"
./build/tb_s2lut.exe "$WEIGHTS" "$SPATIAL"
echo "[build] OK -- all three gates pass"
