#!/usr/bin/env bash
# Host self-check (PC). With float dequant the kernel C-model is bit-exact to conv2d(), so the trunk run through
# the host's DDR image must give byte-identical dumps to the plain CPU trunk.
#   bash board_host/test_packing.sh <weights_dir> <input.bin>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$(uname -s)" in MINGW*|MSYS*) export PATH="/c/msys64/ucrt64/bin:$PATH";; esac
bash "$HERE/build_board_host.sh" exact
bash "$HERE/build_board_host.sh" exactcpu
B="$HERE/../build"; T="$B/test_packing_out"; mkdir -p "$T/sim" "$T/cpu"
"$B/y26_exact"    "$1" "$2" "$T/sim" > "$T/sim.log"
"$B/y26_exactcpu" "$1" "$2" "$T/cpu" > "$T/cpu.log"
n=$(awk '/ kernel convs:/ {print $2}' "$T/sim.log"); n=${n:-0}
[ "$n" -gt 0 ] || { echo "FAIL: no conv went through the host path"; exit 1; }
diff -r "$T/sim" "$T/cpu" > /dev/null || { echo "FAIL: dumps differ"; diff -rq "$T/sim" "$T/cpu"; exit 1; }
echo "PASS: $(ls "$T/sim" | wc -l) dumps byte-identical, $n convs through the host DDR image"
