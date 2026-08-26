#!/usr/bin/env bash
# Build the C-simulation with g++ (no Vitis needed) and run both modes.
# The same binary runs FP32 or INT8: the mode is auto-detected from the weights dir (manifest_int8.txt
# => INT8 integer-accumulation W8A8 dataflow; manifest.txt => FP32). A weights dir that also has
# splits.txt is channel-compacted (physically shrunk shapes) and needs no separate binary.
#   bash hls/csim/build_csim.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
echo "[build] compiling yolo26_csim ..."
g++ -O3 -fopenmp -march=native -std=c++17 \
    main.cpp yolo26_trunk.cpp -o yolo26_csim
# yolo26_csim_fast is the SAME binary -- "fast" is the YOLO26_HEADS_ONLY env gate, not a build flag.
# csim_eval_map.py runs this one over the val set, so it MUST be refreshed here: it was originally
# built ad hoc, and a stale copy silently scores whatever the code used to do.
cp yolo26_csim yolo26_csim_fast
# yolo26_csim_deploy is the DEPLOYMENT shape: the one2many head compiled out. That branch is 12.9% of
# the model's MACs and is dead at inference (see the comment in yolo26_trunk.cpp), so this binary is
# what the HLS kernel should mirror -- not yolo26_csim, which keeps o2m alive for the comparison gates.
# Flags are otherwise IDENTICAL to yolo26_csim on purpose: the o2o dumps must come out md5-identical,
# and -march=native FMA contraction would break that gate if the two builds diverged.
echo "[build] compiling yolo26_csim_deploy (-DYOLO26_NO_O2M) ..."
g++ -O3 -fopenmp -march=native -std=c++17 -DYOLO26_NO_O2M \
    main.cpp yolo26_trunk.cpp -o yolo26_csim_deploy
# decode_test: standalone host-side (ARM PS) one2one decoder validation harness (decode.h). No OpenMP,
# no deps -- the same decode.h cross-compiles for the ZCU102 PS unchanged. Gated by decode_ref.py.
echo "[build] compiling decode_test ..."
g++ -O3 -std=c++17 decode_test.cpp -o decode_test
echo "[build] running FP32 ..."
./yolo26_csim ../weights ../dumps/input.bin ../dumps
if [ -f ../weights_int8/manifest_int8.txt ]; then
    echo "[build] running INT8 ..."
    mkdir -p ../dumps_int8
    ./yolo26_csim ../weights_int8 ../dumps/input.bin ../dumps_int8
fi
if [ -f ../weights_sq/manifest_sq.txt ]; then
    echo "[build] running SmoothQuant ..."
    mkdir -p ../dumps_sq
    ./yolo26_csim ../weights_sq ../dumps/input.bin ../dumps_sq
fi
if [ -f ../weights_sq_compact/splits.txt ]; then
    echo "[build] running SmoothQuant channel-compacted ..."
    mkdir -p ../dumps_sq_compact
    ./yolo26_csim ../weights_sq_compact ../dumps/input.bin ../dumps_sq_compact
fi
if [ -f ../weights_compact/splits.txt ]; then
    echo "[build] running channel-compacted ..."
    mkdir -p ../dumps_compact
    ./yolo26_csim ../weights_compact ../dumps/input.bin ../dumps_compact
fi
for d in weights_compact_fold weights_sq_compact_fold; do
    if [ -f "../$d/splits.txt" ]; then
        echo "[build] running $d ..."
        mkdir -p "../dumps_${d#weights_}"
        ./yolo26_csim "../$d" ../dumps/input.bin "../dumps_${d#weights_}"
    fi
done
echo "[build] OK"
