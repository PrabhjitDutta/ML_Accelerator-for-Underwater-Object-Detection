#!/usr/bin/env bash
# Build the C++ reference model (no Vitis needed), run it on the reference frame, and check every layer output
# against reference_dumps/int8_shipping_model/ byte for byte.
#   bash model_c/scripts/build_reference_model.sh
# Binaries in model_c/build/:
#   yolo26_csim         the full model (both heads). YOLO26_HEADS_ONLY=1 skips the per-layer dumps.
#   yolo26_csim_deploy  -DYOLO26_NO_O2M (the deployed shape); otherwise identical, so its o2o dumps match.
#   decode_test         the detection decoder (reference_model/detection_decode.h) on its own.
# The weights mode is detected from the weights dir.
set -euo pipefail
M="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$(uname -s)" in MINGW*|MSYS*) export PATH="/c/msys64/ucrt64/bin:$PATH";; esac
R="$M/reference_model"; B="$M/build"; mkdir -p "$B/dumps"
echo "[build] yolo26_csim, yolo26_csim_deploy, decode_test ..."
g++ -O3 -fopenmp -march=native -std=c++17 "$R/run_model.cpp" "$R/yolo26_network.cpp" -o "$B/yolo26_csim"
g++ -O3 -fopenmp -march=native -std=c++17 -DYOLO26_NO_O2M "$R/run_model.cpp" "$R/yolo26_network.cpp" -o "$B/yolo26_csim_deploy"
g++ -O3 -std=c++17 "$M/testbench/detection_decode_test.cpp" -o "$B/decode_test"
echo "[build] running the shipping INT8 model on reference_dumps/input.bin ..."
"$B/yolo26_csim" "$M/weights" "$M/reference_dumps/input.bin" "$B/dumps"
bad=0; n=0
for f in "$M"/reference_dumps/int8_shipping_model/*.bin; do
    n=$((n+1)); cmp -s "$f" "$B/dumps/$(basename "$f")" || { echo "DIFF $(basename "$f")"; bad=$((bad+1)); }
done
[ "$bad" -eq 0 ] && echo "[build] OK - $n/$n layer outputs byte-identical to the reference" \
                 || { echo "[build] $bad/$n outputs differ (a different CPU's -march=native can change float rounding)"; exit 1; }
