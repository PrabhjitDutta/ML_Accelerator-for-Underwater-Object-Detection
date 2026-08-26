// main.cpp - C-simulation driver. Loads BN-folded weights, reads the shared input.bin, runs the
// trunk, and dumps every layer output for cosine comparison against PyTorch.
//   ./yolo26_csim <weights_dir> <input.bin> <dump_dir>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "yolo26_trunk.h"
#include "yolo26_utils.h"

// The loader and the dumper throw on a bad weights dir / unwritable dump dir. Catching here turns an
// abort+core-dump into a clean "[csim] error: ..." and exit 1, which is what the eval harness (which
// runs this per image under subprocess check=True) can actually report.
static int run(int argc, char** argv);

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[csim] error: %s\n", e.what());
        return 1;
    }
}

static int run(int argc, char** argv) {
    const std::string wdir  = argc > 1 ? argv[1] : "../weights";
    const std::string inbin = argc > 2 ? argv[2] : "../dumps/input.bin";
    const std::string ddir  = argc > 3 ? argv[3] : "../dumps";

    Weights W;
    W.load(wdir);
    const char* mode = W.sq ? "SmoothQuant W8A8 (asym uint8 + per-IC smooth)"
                            : (W.int8 ? "INT8 fake-quant (symmetric)" : "FP32");
    std::printf("[csim] loaded %zu convs from %s  (mode=%s%s)\n", W.conv.size(), wdir.c_str(), mode,
                W.compact ? ", channel-compacted" : "");

    std::vector<float> raw = read_bin(inbin);
    if (raw.size() != (size_t)3 * 640 * 640) {
        std::fprintf(stderr, "input.bin has %zu floats, expected %d\n", raw.size(), 3 * 640 * 640);
        return 1;
    }
    Tensor input(3, 640, 640);
    input.d = std::move(raw);
    std::printf("[csim] input %dx%dx%d\n", input.C, input.H, input.W);

    run_trunk(W, input, ddir);
    std::printf("[csim] done -> %s\n", ddir.c_str());
    return 0;
}
