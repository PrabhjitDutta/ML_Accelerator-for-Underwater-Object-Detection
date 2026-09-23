#!/usr/bin/env bash
# Build and run the two HLS-kernel gates. Companion to build_reference_model.sh, which covers the golden C-sim.
#
# These were originally built ad hoc, which is exactly the failure build_reference_model.sh warns about for
# yolo26_csim_fast: on 2026-08-09 tb_s1.exe was 4 minutes OLDER than all three of its sources, so the
# recorded "100/100 bit-exact" PASS came from a binary that did not contain the final edits. It
# happened to still hold (the edits were confined to the Y26_FX_DEQUANT branch and to reporting), but
# a stale gate silently certifies whatever the code used to do. Always rebuild via this script.
#
#   bash scripts/run_kernel_gates.sh [weights_dir] [spatial]
#
# All three gates read the SAME source (conv_engine_tb.cpp); the only difference is the defines, which
# also flip the gate's bar:
#   tb_s1    - float dequant    -> EXACT equality vs conv2d(). Any mismatch is a real bug.
#   tb_s2    - ap_fixed dequant -> tolerance + cosine, because the dequant is genuinely re-rounded.
#   tb_s2lut - + fixed-point SiLU LUT, replacing std::exp(double). Same tolerance bar.
#
# These own the project's EXHAUSTIVE coverage: all 100 integer convs in seconds, and tb_s1 is the
# only place the bit-exact bar is enforced anywhere. Vitis csim runs the same source but is ~200x
# slower (-fhls-csim instrumentation), so hls_csim_synth_cosim.tcl caps it at a handful of convs. Do not
# let that cap migrate here.
# Binaries land in build/ so the source directory stays sources-only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."   # model_c/
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
#   Y26_EXTRA_FLAGS="-DY26_LANES=32" bash scripts/run_kernel_gates.sh
# This exists because a LANE-COUNT CHANGE IS A CORRECTNESS CHANGE, not just a resource knob: it moves
# the bank assignment (ic & (Y26_LANES-1)) and the partial-tile zero-fill in the MAC pass. Never
# synthesize or place-and-route a new Y26_LANES without running tb_s1 at that value first - a
# bit-exact failure would otherwise show up as a plausible-looking utilization number for a design
# that computes the wrong answer. Y26_LANES must stay a POWER OF TWO (the banking relies on the
# mask); the old "must be prime" invariant applied to the abandoned cyclic-partition scheme.
# ---- GATE BASE: the SHIPPING configuration (added 2026-08-25) -------------------------------
# This script used to compile with NO -D flags at all, so every gate ran against the HEADER
# DEFAULTS -- Y26_LANES=16, no Y26_OCPACK, Y26_ACT_WORD=8 -- while the design that is synthesized,
# routed and reported is Y26_LANES=512 with the packed datapath. The gates were green the whole
# time and could not have been anything else: the tb's own depth guard (tb:163) is
#     ic*H*W <= Y26_ACT_MAX_ELEMS
# and xbuf feasibility is
#     ceil(ic/LANES)*H*W <= Y26_ACT_MAX_ELEMS/LANES ,
# which are the SAME inequality whenever LANES divides ic -- true for essentially every conv at
# LANES=16. So at the default the tb skipped a conv exactly when it would have overflowed, and no
# spatial size could ever have exposed the xbuf capacity bug. The warning 40 lines above ("never
# synthesize a new Y26_LANES without running tb_s1 at that value first") was correct and was not
# followed, because nothing made the mismatch visible.
# Default = the SHIPPING flags (FPGA/1_hls_synthesis/hls_build_flags.txt) minus the two below. Y26_FX_DEQUANT and Y26_SILU_LUT are
# deliberately EXCLUDED: this script layers them itself per gate, and putting Y26_FX_DEQUANT here
# would silently downgrade tb_s1 from its bit-exact bar to the tolerance bar.
# Override with Y26_GATE_BASE="..." ; set it EMPTY to get the old header-default behaviour.
Y26_GATE_BASE="${Y26_GATE_BASE--DY26_A1 -DY26_ACT_LATENCY=100 -DY26_ACT_MAX_ELEMS=4194304UL -DY26_ACT_NRO=128 -DY26_ACT_WORD=64 -DY26_DWP=16 -DY26_EPI_WIDE=8 -DY26_LANES=512 -DY26_OCPACK -DY26_OCPACK_P=16 -DY26_OUT_LATENCY=8 -DY26_OUT_WORD=256 -DY26_PREFETCH -DY26_ROWS=24 -DY26_SPLIT_OUT -DY26_WT_LATENCY=8 -DY26_WT_TAPMAJOR -DY26_WT_WORD=64 -DY26_XB64 -DY26_XBUF_WIDE=8 -DY26_YSTRIDE_PAD -DY26_YDIRECT -UY26_ACT_WORD -DY26_ACT_WORD=128 -UY26_XBUF_WIDE -DY26_XBUF_WIDE=16}"
Y26_EXTRA_FLAGS="${Y26_EXTRA_FLAGS:-}"
case " $Y26_GATE_BASE $Y26_EXTRA_FLAGS " in
  *" -DY26_FX_DEQUANT "*) echo "[build] REFUSING: -DY26_FX_DEQUANT in the flags downgrades tb_s1's"
                          echo "        bit-exact bar to a tolerance bar and it will still say PASS."
                          exit 2 ;;
esac
# Always print what was actually compiled. A gate result is only meaningful next to its flags.
echo "[build] gate base : ${Y26_GATE_BASE:-<header defaults>}"
[ -n "$Y26_EXTRA_FLAGS" ] && echo "[build] extra flags: $Y26_EXTRA_FLAGS"
CXXFLAGS=(-O2 -std=c++14 -fopenmp -I"$VITIS_INC" $Y26_GATE_BASE $Y26_EXTRA_FLAGS)
# conv_engine_top.cpp is in the list because the tb now calls y26_conv_top() rather than
# y26_conv2d_hls() directly -- see the comment at that call site. The top is a pure forwarder, so
# this does not change what the gates measure; it makes the SAME tb usable for HLS co-simulation,
# which can only trace arguments at the synthesis boundary. g++ ignores its #pragma HLS lines.
SRC=(testbench/conv_engine_tb.cpp hls_kernel/conv_engine.cpp hls_kernel/conv_engine_top.cpp)

echo "[build] compiling tb_s1 (stage-1, float dequant, bit-exact bar) ..."
g++ "${CXXFLAGS[@]}" "${SRC[@]}" -o build/tb_s1.exe
echo "[build] compiling tb_s2 (stage-2, ap_fixed dequant, tolerance bar) ..."
g++ "${CXXFLAGS[@]}" -DY26_FX_DEQUANT "${SRC[@]}" -o build/tb_s2.exe
echo "[build] compiling tb_s2lut (stage-2 + fixed-point SiLU LUT, tolerance bar) ..."
g++ "${CXXFLAGS[@]}" -DY26_FX_DEQUANT -DY26_SILU_LUT "${SRC[@]}" -o build/tb_s2lut.exe

WEIGHTS="${1:-weights}"
SPATIAL="${2:-16}"
echo "[build] running gates on $WEIGHTS at ${SPATIAL}x${SPATIAL} ..."
./build/tb_s1.exe    "$WEIGHTS" "$SPATIAL"
./build/tb_s2.exe    "$WEIGHTS" "$SPATIAL"
./build/tb_s2lut.exe "$WEIGHTS" "$SPATIAL"
echo "[build] OK -- all three gates pass"
