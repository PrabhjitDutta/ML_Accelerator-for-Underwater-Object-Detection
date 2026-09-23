#!/usr/bin/env bash
# Build the ZCU102 host (y26_board) or its PC stand-in (y26_board_sim).
#   bash board_host/build_board_host.sh board   # on the ZCU102, or CXX=aarch64-linux-gnu-g++ on a PC
#   bash board_host/build_board_host.sh sim     # PC: C-model backend, same packing, same flags
# Needs the Vitis ap_int headers: VITIS_INC=<dir containing ap_int.h>.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="$(cd "$HERE/.." && pwd)"   # model_c/
OUTD="$M/build"; mkdir -p "$OUTD"   # binaries land in model_c/build/
VITIS_INC_DEF="C:/AMDDesignTools/2026.1/Vitis/include"
[ -d "$M/vitis_include" ] && VITIS_INC_DEF="$M/vitis_include"   # a copy next to the sources (on the board)
VITIS_INC="${VITIS_INC:-$VITIS_INC_DEF}"
CXX="${CXX:-g++}"
[ -f "$VITIS_INC/ap_int.h" ] || { echo "[host] no ap_int.h under $VITIS_INC - set VITIS_INC" >&2; exit 2; }
case "$(uname -s)" in MINGW*|MSYS*) export PATH="/c/msys64/ucrt64/bin:$PATH";; esac

# The shipping kernel flags (FPGA/1_hls_synthesis/hls_build_flags.txt): they set the arithmetic and the port
# layout, so a host built with anything else does not match the bitstream.
SHIP="-DY26_A1 -DY26_ACT_LATENCY=100 -DY26_ACT_MAX_ELEMS=4194304UL -DY26_ACT_NRO=128 -DY26_ACT_WORD=64 -DY26_DWP=16 -DY26_EPI_WIDE=8 -DY26_FX_DEQUANT -DY26_LANES=512 -DY26_OCPACK -DY26_OCPACK_P=16 -DY26_OUT_LATENCY=8 -DY26_OUT_WORD=256 -DY26_PREFETCH -DY26_ROWS=24 -DY26_SILU_LUT -DY26_SPLIT_OUT -DY26_WT_LATENCY=8 -DY26_WT_TAPMAJOR -DY26_WT_WORD=64 -DY26_XB64 -DY26_XBUF_WIDE=8 -DY26_YSTRIDE_PAD -DY26_YDIRECT -UY26_ACT_WORD -DY26_ACT_WORD=128 -UY26_XBUF_WIDE -DY26_XBUF_WIDE=16"
FLAGS="$M/../FPGA/1_hls_synthesis/hls_build_flags.txt"
if [ -f "$FLAGS" ] && [ "$(grep -v '^#' "$FLAGS" | tr -d '\r' | tr '\n' ' ' | sed 's/ $//')" != "-std=c++14 $SHIP" ]; then
    echo "[host] REFUSING: SHIP flags differ from $FLAGS - a host must match its bitstream" >&2; exit 2
fi

MODE="${1:-board}"
case "$MODE" in
    # the one2many head is training-only; only the PC modes keep it for the o2m dumps
    board) OUT=y26_board;     EXTRA="-DY26_BOARD -DYOLO26_NO_O2M" ;;
    sim)   OUT=y26_board_sim; EXTRA="-DY26_BOARD -DY26_BOARD_SIM" ;;
    # packing test: float dequant, so SIM and CPU trunks must match byte for byte
    exact|exactcpu) SHIP="${SHIP/-DY26_FX_DEQUANT /}"; SHIP="${SHIP/-DY26_SILU_LUT /}"; OUT=y26_$MODE
           EXTRA="-DY26_BOARD -DY26_BOARD_SIM -DY26_HOST_EXACT"
           [ "$MODE" = exactcpu ] && EXTRA="" ;;   # the plain CPU trunk: no host, no hook
    *) echo "usage: build_board_host.sh board|sim|exact|exactcpu" >&2; exit 2 ;;
esac
# -ffp-contract=off: no FMA fusion, so float ops round as on the PC that produced the reference dumps.
# x86: -msse4.1 inlines nearbyint and -fno-trapping-math lets GCC if-convert the clamp, so the quantizer
# vectorizes as on the A53 (same IEEE results).
case "$("$CXX" -dumpmachine)" in aarch64*) ARCH="-mcpu=cortex-a53";; x86_64*) ARCH="-msse4.1 -fno-trapping-math";; *) ARCH="";; esac
# Y26_YQ8=1: code-mode host (-DY26_YQ8); pairs only with a bitstream built with -DY26_YQ8.
[ "${Y26_YQ8:-0}" = 1 ] && { SHIP="$SHIP -DY26_YQ8"; OUT="${OUT}_yq8"; }
HOSTSRC=(); [ "$MODE" = exactcpu ] || HOSTSRC=("$HERE/board_host.cpp")   # quoted: the path may hold spaces
echo "[host] $CXX $MODE : $SHIP $EXTRA $ARCH"
"$CXX" -O2 $ARCH -ffp-contract=off -std=c++17 -fopenmp -pthread -Wno-unknown-pragmas -I"$VITIS_INC" $SHIP $EXTRA \
    "$M/reference_model/run_model.cpp" "$M/reference_model/yolo26_network.cpp" "$M/hls_kernel/conv_engine.cpp" "$M/hls_kernel/conv_engine_top.cpp" \
    "${HOSTSRC[@]}" -o "$OUTD/$OUT"
echo "[host] -> $OUTD/$OUT"
