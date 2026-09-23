#!/usr/bin/env bash
# Bit-exact gate over all 100 kernel convs, each at its own input size in the real frame
# (testbench/frame_conv_geometry.txt). Catches size-dependent bugs a uniform test size can hide.
#   bash model_c/scripts/run_kernel_gate_all_convs.sh ["<extra -D flags>"]
# Flags = the shipping set minus FX_DEQUANT/SILU_LUT (those replace the exact bar with a tolerance).
set -u
M="$(cd "$(dirname "$0")/.." && pwd)"; cd "$M"
case "$(uname -s)" in MINGW*|MSYS*) export PATH="/c/msys64/ucrt64/bin:$PATH";; esac
VITIS_INC="${VITIS_INC:-C:/AMDDesignTools/2026.1/Vitis/include}"
export WD="$M/weights"
B="-DY26_A1 -DY26_ACT_LATENCY=100 -DY26_ACT_MAX_ELEMS=4194304UL -DY26_ACT_NRO=128 -DY26_ACT_WORD=64 -DY26_DWP=16 -DY26_EPI_WIDE=8 -DY26_LANES=512 -DY26_OCPACK -DY26_OCPACK_P=16 -DY26_OUT_LATENCY=8 -DY26_OUT_WORD=256 -DY26_PREFETCH -DY26_ROWS=24 -DY26_SPLIT_OUT -DY26_WT_LATENCY=8 -DY26_WT_TAPMAJOR -DY26_WT_WORD=64 -DY26_XB64 -DY26_XBUF_WIDE=8 -DY26_YSTRIDE_PAD -DY26_YDIRECT -UY26_ACT_WORD -DY26_ACT_WORD=128 -UY26_XBUF_WIDE -DY26_XBUF_WIDE=16 ${1:-}"
echo "[gate] flags: $B"
mkdir -p build/rg
g++ -O2 -std=c++14 -fopenmp -I"$VITIS_INC" $B testbench/conv_engine_tb.cpp hls_kernel/conv_engine.cpp \
    hls_kernel/conv_engine_top.cpp -o build/tb_rg.exe || exit 2
run(){ ./build/tb_rg.exe "$WD" "$2" 1 "$1" > "build/rg/$1.log" 2>&1; grep -q "^PASS" "build/rg/$1.log" && echo "PASS $1" || echo "FAIL $1 @$2"; }
export -f run
awk '{print $1, $2}' testbench/frame_conv_geometry.txt | xargs -P 8 -n 2 bash -c 'run "$0" "$1"' > build/rg/summary.txt
np=$(grep -c ^PASS build/rg/summary.txt); nt=$(wc -l < build/rg/summary.txt)
grep ^FAIL build/rg/summary.txt
echo "[gate] $np/$nt PASS"; [ "$np" -eq "$nt" ] && [ "$nt" -eq 100 ]
