// Reference-model driver.
//   ./yolo26_csim <weights_dir> <input.bin> <dump_dir>   one frame, dump every layer
//   ./yolo26_csim <weights_dir> - <out_dir>              frame loop: input paths on stdin, detections to
//                                                        <out_dir>/<stem>.txt ("cls score x1 y1 x2 y2")
// An input is either input.bin (float32 3x640x640) or .u8 (the same frame as bytes k, value k/255.f).
#include <chrono>
#include <cstdio>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include "yolo26_network.h"
#include "layer_ops.h"
#include "detection_decode.h"

// Turn loader/dumper exceptions into an error message and exit 1.
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

    auto load = [](const std::string& p) {
        const size_t n = (size_t)3 * 640 * 640;
        if (p.size() > 3 && p.compare(p.size() - 3, 3, ".u8") == 0) {
            auto u = std::make_shared<std::vector<uint8_t>>(n);
            FILE* f = std::fopen(p.c_str(), "rb");
            if (!f) throw std::runtime_error("cannot open " + p);
            const size_t got = std::fread(u->data(), 1, n + 1, f);   // n + 1: detects a longer file
            std::fclose(f);
            if (got != n) throw std::runtime_error(p + ": " + std::to_string(got) + " bytes, expected 3*640*640");
            float tab[256];
            for (int k = 0; k < 256; ++k) tab[k] = (float)k / 255.f;
            Tensor t(3, 640, 640, NoInit());
            for (size_t i = 0; i < n; ++i) t.d[i] = tab[(*u)[i]];
            y26_input_u8(t.d.data(), std::move(u));
            return t;
        }
        std::vector<float> raw = read_bin(p);
        if (raw.size() != (size_t)3 * 640 * 640)
            throw std::runtime_error(p + ": " + std::to_string(raw.size()) + " floats, expected 3*640*640");
        Tensor t(3, 640, 640, NoInit());
        t.d.assign(raw.begin(), raw.end());
        y26_input_u8(t.d.data(), nullptr);         // storage reused from an earlier .u8 frame
        return t;
    };
    if (inbin == "-") {
        using clk = std::chrono::steady_clock;
        auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        // The next frame is read on another thread while this one runs.
        auto next_path = [](std::string& p) {
            while (std::getline(std::cin, p)) {
                if (!p.empty() && p.back() == '\r') p.pop_back();
                if (!p.empty()) return true;
            }
            return false;
        };
        std::string p, pn;
        bool more = next_path(pn);
        std::future<Tensor> pending;
        if (more) pending = std::async(std::launch::async, load, pn);
        for (int n = 0; more;) {
            p = pn;
            const auto t0 = clk::now();
            const Tensor input = pending.get();
            more = next_path(pn);
            if (more) pending = std::async(std::launch::async, load, pn);
            const auto t1 = clk::now();
            y26_board_weights(W);
            const std::vector<Tensor> o = run_trunk(W, input, "");
            const auto t2 = clk::now();
            const auto dets = yolo26::decode_o2o(o[0].d.data(), o[1].d.data(), o[2].d.data(), o[0].C - 4, 300);
            std::string stem = p.substr(p.find_last_of("/\\") + 1);
            if (stem.size() > 4 && stem.compare(stem.size() - 4, 4, ".bin") == 0) stem.resize(stem.size() - 4);
            else if (stem.size() > 3 && stem.compare(stem.size() - 3, 3, ".u8") == 0) stem.resize(stem.size() - 3);
            const std::string op = ddir + "/" + stem + ".txt";
            FILE* f = std::fopen(op.c_str(), "w");
            if (!f) throw std::runtime_error("cannot write " + op);
            for (const auto& d : dets) std::fprintf(f, "%d %.6f %.5f %.5f %.5f %.5f\n", d.cls, d.score, d.x1, d.y1, d.x2, d.y2);
            std::fclose(f);
            const auto t3 = clk::now();
            std::printf("[frame] %d %s: %.1f ms (read %.1f | trunk %.1f | decode+write %.1f), %zu dets\n", n, stem.c_str(),
                        ms(t0, t3), ms(t0, t1), ms(t1, t2), ms(t2, t3), dets.size());
            y26_frame_end(n++);
        }
        return 0;
    }
    Tensor input = load(inbin);
    std::printf("[csim] input %dx%dx%d\n", input.C, input.H, input.W);

    run_trunk(W, input, ddir);
    std::printf("[csim] done -> %s\n", ddir.c_str());
    return 0;
}
