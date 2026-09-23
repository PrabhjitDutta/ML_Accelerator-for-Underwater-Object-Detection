// Runs detection_decode.h on the three o2o_{0,1,2}_csim.bin maps in <dump_dir>.
//   ./decode_test <dump_dir> <out.txt>     out: "cls score x1 y1 x2 y2" per line
#include "../reference_model/detection_decode.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::vector<float> load_bin(const std::string& path, size_t n) {
    std::vector<float> v(n);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    size_t got = std::fread(v.data(), sizeof(float), n, f);
    std::fclose(f);
    if (got != n) { std::fprintf(stderr, "%s: read %zu != %zu\n", path.c_str(), got, n); std::exit(1); }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <dump_dir> <out.txt>\n", argv[0]); return 1; }
    const std::string d = argv[1];
    // shapes: [8,80,80], [8,40,40], [8,20,20]
    auto m0 = load_bin(d + "/o2o_0_csim.bin", (size_t)8 * 80 * 80);
    auto m1 = load_bin(d + "/o2o_1_csim.bin", (size_t)8 * 40 * 40);
    auto m2 = load_bin(d + "/o2o_2_csim.bin", (size_t)8 * 20 * 20);

    auto dets = yolo26::decode_o2o(m0.data(), m1.data(), m2.data(), /*nc=*/4, /*max_det=*/300);

    FILE* out = std::fopen(argv[2], "w");
    if (!out) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    for (const auto& x : dets)
        std::fprintf(out, "%d %.6f %.5f %.5f %.5f %.5f\n", x.cls, x.score, x.x1, x.y1, x.x2, x.y2);
    std::fclose(out);
    std::fprintf(stderr, "decode_test: %zu detections -> %s\n", dets.size(), argv[2]);
    return 0;
}
