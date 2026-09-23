#!/usr/bin/env bash
# Build and run the kernel gates (all from testbench/conv_engine_tb.cpp, differing only in defines):
#   tb_s1    - float dequant    -> exact equality vs conv2d()
#   tb_s2    - ap_fixed dequant -> tolerance + cosine
#   tb_s2lut - + fixed-point SiLU LUT; same tolerance
#   bash scripts/run_kernel_gates.sh [weights_dir] [spatial]
# Always rebuild through this script so a gate never runs a stale binary. Binaries go to build/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."   # model_c/
mkdir -p build

# Real Vitis ap_int/ap_fixed headers; override for a non-default install.
VITIS_INC="${VITIS_INC:-C:/AMDDesignTools/2026.1/Vitis/include}"
if [ ! -f "$VITIS_INC/ap_int.h" ]; then
    echo "[build] ERROR: no ap_int.h under $VITIS_INC -- set VITIS_INC to your Vitis include dir." >&2
    exit 2
fi

# MSYS2: the .exe needs the ucrt64 runtime DLLs on PATH.
case "$(uname -s)" in MINGW*|MSYS*) export PATH="/c/msys64/ucrt64/bin:$PATH";; esac

# Y26_EXTRA_FLAGS adds defines, e.g. Y26_EXTRA_FLAGS="-DY26_LANES=32". Changing Y26_LANES changes bank
# assignment, so run tb_s1 at any new value before synthesizing it (Y26_LANES must be a power of two).
# Y26_GATE_BASE defaults to the shipping flags (FPGA/1_hls_synthesis/hls_build_flags.txt) minus Y26_FX_DEQUANT
# and Y26_SILU_LUT, which this script adds per gate. Set it empty for the header defaults.
Y26_GATE_BASE="${Y26_GATE_BASE--DY26_A1 -DY26_ACT_LATENCY=100 -DY26_ACT_MAX_ELEMS=4194304UL -DY26_ACT_NRO=128 -DY26_ACT_WORD=64 -DY26_DWP=16 -DY26_EPI_WIDE=8 -DY26_LANES=512 -DY26_OCPACK -DY26_OCPACK_P=16 -DY26_OUT_LATENCY=8 -DY26_OUT_WORD=256 -DY26_PREFETCH -DY26_ROWS=24 -DY26_SPLIT_OUT -DY26_WT_LATENCY=8 -DY26_WT_TAPMAJOR -DY26_WT_WORD=64 -DY26_XB64 -DY26_XBUF_WIDE=8 -DY26_YSTRIDE_PAD -DY26_YDIRECT -UY26_ACT_WORD -DY26_ACT_WORD=128 -UY26_XBUF_WIDE -DY26_XBUF_WIDE=16}"
Y26_EXTRA_FLAGS="${Y26_EXTRA_FLAGS:-}"
case " $Y26_GATE_BASE $Y26_EXTRA_FLAGS " in
  *" -DY26_FX_DEQUANT "*) echo "[build] REFUSING: -DY26_FX_DEQUANT in the flags downgrades tb_s1's"
                          echo "        bit-exact bar to a tolerance bar and it will still say PASS."
                          exit 2 ;;
esac
# Print what was compiled: a gate result only means something next to its flags.
echo "[build] gate base : ${Y26_GATE_BASE:-<header defaults>}"
[ -n "$Y26_EXTRA_FLAGS" ] && echo "[build] extra flags: $Y26_EXTRA_FLAGS"
CXXFLAGS=(-O2 -std=c++14 -fopenmp -I"$VITIS_INC" $Y26_GATE_BASE $Y26_EXTRA_FLAGS)
# conv_engine_top.cpp: the tb calls the synthesis top y26_conv_top().
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
