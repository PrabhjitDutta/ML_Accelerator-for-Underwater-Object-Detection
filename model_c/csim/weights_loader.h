// weights_loader.h - manifest-driven loader for the per-conv float32 .bin files.
//
// Reads weights/manifest.txt (one line per conv: name oc ic kh kw sh sw ph pw groups has_bn act)
// and the matching <name>.w.bin (+ .s.bin if BN-folded) + <name>.b.bin. Every conv is addressable
// by its PyTorch module name (e.g. "10.m.0.attn.qkv.conv"), so the trunk graph reads like the model.
#pragma once
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "yolo26_types.h"

struct ConvW {
    std::string name;        // for diagnostics only
    int oc, ic, kh, kw, sh, sw, ph, pw, groups;
    bool has_bn = false;
    int act;                 // 0 identity, 1 silu
    std::vector<float> w;    // OC * (IC/groups) * KH * KW (FP32; INT8 mode: already fake-quantized)
    std::vector<float> s;    // OC fused scale (empty if !has_bn -> treated as 1)
    std::vector<float> b;    // OC fused/conv bias
    bool quant = false;      // INT8 mode: integer-accumulation W8A8 dataflow
    float sa = 1.f;          // per-tensor input-activation scale/step (INT8/SmoothQuant only)
    std::vector<float> wsc;  // OC per-channel weight scale (INT8/SmoothQuant only); w holds int8 codes
    bool asym = false;       // SmoothQuant: asymmetric uint8 activations + per-input-channel smooth
    float lo = 0.f;          // SmoothQuant: per-tensor activation FakeQuantize in_low (zero-point)
    std::vector<float> ssc;  // SmoothQuant: IC per-input-channel smooth pre-scale
    // The 6 one2one_cv3 head depthwise convs carry a PER-INPUT-CHANNEL activation range instead of a
    // per-tensor one. That is only expressible because they are depthwise: output o reads input o, so
    // one step/lo governs the whole accumulation for o. A non-depthwise conv would need a different
    // step per accumulated input channel, which a single int32 accumulator cannot represent -- hence
    // the loader asserts depthwise here rather than trusting the flag.
    bool perch = false;
    std::vector<float> sa_v; // IC per-input-channel step (perch only)
    std::vector<float> lo_v; // IC per-input-channel in_low (perch only)
    inline float step_of(int ch) const { return perch ? sa_v[ch] : sa; }
    inline float lo_of(int ch)   const { return perch ? lo_v[ch] : lo; }
};

inline std::vector<float> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<float> v(n / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

// Same, but fails loudly when the file length disagrees with what the manifest implies. Without this
// a truncated or stale .bin is read at whatever length it happens to be, and conv2d then indexes past
// the end of the vector -- which produces silently wrong output and still exits 0. Since the export
// pipeline regenerates several weight dirs (dense / compact / fold, FP32 / INT8 / SmoothQuant), a
// manifest-vs-binary desync is a realistic failure and must not be silent.
inline std::vector<float> read_bin_n(const std::string& path, size_t n, const std::string& what) {
    std::vector<float> v = read_bin(path);
    if (v.size() != n)
        throw std::runtime_error(path + ": " + what + " has " + std::to_string(v.size()) +
                                 " floats but the manifest implies " + std::to_string(n));
    return v;
}

// One fake-quant range (OV FakeQuantize with in==out ranges, 256 levels).
struct FQ {
    float lo = 0.f, hi = 0.f, step = 0.f;
    bool on = false;
    void set(float l, float h) { lo = l; hi = h; step = (h - l) / 255.f; on = true; }
    // clamp(round((x-lo)/step),0,255)*step+lo -- nearbyint to match conv2d's rounding and torch.round
    inline float operator()(float x) const {
        if (!on) return x;
        float q = std::nearbyint((x - lo) / step);
        q = q < 0.f ? 0.f : (q > 255.f ? 255.f : q);
        return q * step + lo;
    }
};

// The five quantizers OV places INSIDE each Attention block (see ingest_smoothquant.py ATTN_FQ).
// v is quantized twice, with separately calibrated ranges, for its two consumers.
struct AttnFQ {
    FQ q, k, sm, v_mm, v_pe;
    bool on = false;
};

struct Weights {
    std::string dir;
    std::unordered_map<std::string, ConvW> conv;
    std::unordered_map<std::string, AttnFQ> attn;   // block prefix, e.g. "10.m.0.attn"

    bool int8 = false;
    bool sq = false;
    bool compact = false;
    // Channel-compacted weights: a C3k2/C2PSA cv1 output no longer splits at C/2, because the two
    // halves lose different numbers of dead channels. splits.txt (written by compact_yolo26.py)
    // gives the compacted split index per block; absent -> dense, split at C/2.
    std::unordered_map<int, int> splits;

    // Split index for block `i` given the (compacted) cv1 output width C.
    int split_of(int i, int C) const {
        auto it = splits.find(i);
        return it == splits.end() ? C / 2 : it->second;
    }

    void load(const std::string& weights_dir) {
        dir = weights_dir;
        {
            std::ifstream sf(dir + "/splits.txt");
            compact = (bool)sf;
            int i, c;
            while (sf >> i >> c) splits[i] = c;
        }
        // Mode is auto-detected from the weights dir by which manifest is present:
        //   manifest_sq.txt   -> SmoothQuant (per-conv qmode; asymmetric uint8 + per-IC smooth scale)
        //   manifest_int8.txt -> Phase-2 symmetric INT8 (per-tensor `sa`, weights BN-folded+quantized)
        //   manifest.txt      -> FP32
        std::ifstream sqf(dir + "/manifest_sq.txt");
        sq = (bool)sqf;
        std::ifstream i8(dir + "/manifest_int8.txt");
        int8 = !sq && (bool)i8;
        const std::string mpath = sq ? (dir + "/manifest_sq.txt")
                                     : (int8 ? (dir + "/manifest_int8.txt") : (dir + "/manifest.txt"));
        std::ifstream mf(mpath);
        if (!mf) throw std::runtime_error("cannot open manifest in " + dir);
        std::string line;
        while (std::getline(mf, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string name;
            ConvW c;
            int flag;   // has_bn (FP32/INT8) OR qmode (SmoothQuant: 0=FP32-fallback, 1=SmoothQuant)
            ss >> name >> c.oc >> c.ic >> c.kh >> c.kw >> c.sh >> c.sw >> c.ph >> c.pw
               >> c.groups >> flag >> c.act;
            if (sq) {
                ss >> c.sa >> c.lo;               // both columns always present (0 when qmode==0)
                int perch = 0;
                ss >> perch;                      // trailing column; absent in pre-Phase-5 manifests
                c.perch = (perch == 1);
                if (flag == 1) { c.quant = true; c.asym = true; }
                else           { c.has_bn = false; }   // FP32-folded (pe + one2many head): scale 1
            } else {
                c.has_bn = (flag != 0);
                if (int8) { ss >> c.sa; c.quant = true; }
            }
            const size_t nw = (size_t)c.oc * (c.ic / c.groups) * c.kh * c.kw;
            c.w = read_bin_n(dir + "/" + name + ".w.bin", nw, "weights");
            c.b = read_bin_n(dir + "/" + name + ".b.bin", (size_t)c.oc, "bias");
            if (c.has_bn) c.s = read_bin_n(dir + "/" + name + ".s.bin", (size_t)c.oc, "BN scale");
            if (c.quant) c.wsc = read_bin_n(dir + "/" + name + ".sw.bin", (size_t)c.oc,
                                            "per-OC weight scale");
            if (c.asym)  c.ssc = read_bin_n(dir + "/" + name + ".ssc.bin", (size_t)c.ic,
                                            "per-IC smooth scale");
            if (c.perch) {
                if (c.groups != c.oc || c.oc != c.ic)
                    throw std::runtime_error("conv " + name + ": per-channel activation range on a "
                                             "non-depthwise conv (groups=" + std::to_string(c.groups) +
                                             " oc=" + std::to_string(c.oc) +
                                             " ic=" + std::to_string(c.ic) + ")");
                c.sa_v = read_bin_n(dir + "/" + name + ".sa.bin", (size_t)c.ic, "per-IC step");
                c.lo_v = read_bin_n(dir + "/" + name + ".lo.bin", (size_t)c.ic, "per-IC in_low");
            }
            c.name = name;
            conv.emplace(name, std::move(c));
        }
        // attn_fq.txt: "<block> qlo qhi klo khi smlo smhi vmmlo vmmhi vpelo vpehi". Absent -> the
        // attention interiors stay FP32 (Phase 3 behaviour), so older weight dirs still load.
        std::ifstream af(dir + "/attn_fq.txt");
        std::string blk;
        float v[10];
        while (af >> blk) {
            for (float& t : v) af >> t;
            AttnFQ a;
            a.q.set(v[0], v[1]); a.k.set(v[2], v[3]); a.sm.set(v[4], v[5]);
            a.v_mm.set(v[6], v[7]); a.v_pe.set(v[8], v[9]);
            a.on = true;
            attn.emplace(blk, a);
        }
    }

    // Attention quantizers for a block prefix; all-off (FP32 interior) when the dir has no attn_fq.txt.
    const AttnFQ& attn_fq(const std::string& block) const {
        static const AttnFQ none;
        auto it = attn.find(block);
        return it == attn.end() ? none : it->second;
    }

    const ConvW& get(const std::string& name) const {
        auto it = conv.find(name);
        if (it == conv.end()) throw std::runtime_error("missing conv " + name);
        return it->second;
    }
};
