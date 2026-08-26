// yolo26_trunk.cpp - the trunk graph, wired to mirror ultralytics' YOLO26s forward exactly.
//
// Topology (verified against the live pruned50 model):
//   0,1 Conv | 2,4 C3k2(shallow) | 3,5,7 Conv | 6,8 C3k2(deep-C3k) | 9 SPPF | 10 C2PSA
//   11,14 Upsample2x | 12,15,18,21 Concat | 13,16,19 C3k2(deep) | 17,20 Conv | 22 C3k2(attn)
//   23 Detect pred-convs (reg_max=1, nc=4 -> 8 ch/scale), feats from layers [16,19,22].
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include "yolo26_trunk.h"
#include "yolo26_utils.h"

// Dumping every intermediate layer is ~40 MB/image, which dominates runtime when the C-sim is driven
// over a whole val set. YOLO26_HEADS_ONLY=1 keeps only the 6 head maps (all decode actually needs).
static bool heads_only() {
    // Read the VALUE, not just presence: YOLO26_HEADS_ONLY=0 must mean off, or a caller trying to
    // disable it (and expecting full dumps) silently gets head maps only.
    static const bool v = [] {
        const char* e = std::getenv("YOLO26_HEADS_ONLY");
        return e && *e && std::string(e) != "0";
    }();
    return v;
}

static void dump(const std::string& dir, const std::string& name, const Tensor& t) {
    if (heads_only() && name.rfind("o2m_", 0) != 0 && name.rfind("o2o_", 0) != 0) return;
    std::string path = dir + "/" + name + "_csim.bin";
    FILE* f = std::fopen(path.c_str(), "wb");
    // A missing/unwritable dump dir used to segfault here (fwrite to a null FILE*) -- and only after
    // the whole forward pass had run. Fail with the path instead.
    if (!f) throw std::runtime_error("cannot write dump " + path + " (does the dump dir exist?)");
    std::fwrite(t.d.data(), sizeof(float), t.d.size(), f);
    std::fclose(f);
}

// --- small helpers -----------------------------------------------------------------------------
static inline Tensor cv(const Weights& W, const std::string& n, const Tensor& x) {
    return conv2d(x, W.get(n));
}

// Standard Bottleneck: cv2(cv1(x)), + residual when add. prefix e.g. "6.m.0.m.0".
static Tensor bottleneck(const Weights& W, const std::string& p, const Tensor& x, bool addr) {
    Tensor y = cv(W, p + ".cv1.conv", x);
    y = cv(W, p + ".cv2.conv", y);
    return addr ? add(x, y) : y;
}

// C3k (deep) block = C3.forward: cv3(cat(m(cv1(x)), cv2(x))), m = 2 bottlenecks. prefix "6.m.0".
static Tensor c3k(const Weights& W, const std::string& p, const Tensor& x) {
    Tensor a = cv(W, p + ".cv1.conv", x);
    a = bottleneck(W, p + ".m.0", a, true);
    a = bottleneck(W, p + ".m.1", a, true);
    Tensor b = cv(W, p + ".cv2.conv", x);
    Tensor cc = concat({&a, &b});
    return cv(W, p + ".cv3.conv", cc);
}

// C2f-style forward for C3k2, with the inner block chosen by `mode`:
//   0 = shallow (single Bottleneck m.0), 1 = deep (single C3k m.0)
static Tensor c3k2(const Weights& W, int i, const Tensor& x, int mode) {
    const std::string s = std::to_string(i);
    Tensor t = cv(W, s + ".cv1.conv", x);          // -> 2c channels (dense); c + (C-c) compacted
    const int c = W.split_of(i, t.C);
    Tensor a = slice_ch(t, 0, c), b = slice_ch(t, c, t.C - c);
    Tensor mb = (mode == 0) ? bottleneck(W, s + ".m.0", b, true) : c3k(W, s + ".m.0", b);
    Tensor cc = concat({&a, &b, &mb});
    return cv(W, s + ".cv2.conv", cc);
}

// PSABlock: x = x + attn(x); x = x + ffn(x). prefix "10.m.0" or "22.m.0.1".
static Tensor psablock(const Weights& W, const std::string& p, const Tensor& x) {
    Tensor a = attention(x, W.get(p + ".attn.qkv.conv"), W.get(p + ".attn.proj.conv"),
                         W.get(p + ".attn.pe.conv"), W.attn_fq(p + ".attn"));
    Tensor x2 = add(x, a);
    Tensor f = cv(W, p + ".ffn.0.conv", x2);
    f = cv(W, p + ".ffn.1.conv", f);
    return add(x2, f);
}

// C2PSA (layer 10): a,b = cv1(x).split; b = psablock(b); cv2(cat(a,b)).
static Tensor c2psa(const Weights& W, int i, const Tensor& x, const std::string& dir) {
    const std::string s = std::to_string(i);
    Tensor t = cv(W, s + ".cv1.conv", x);
    dump(dir, "l10_cv1", t);
    const int c = W.split_of(i, t.C);
    Tensor a = slice_ch(t, 0, c), b = slice_ch(t, c, t.C - c);
    Tensor att = attention(b, W.get(s + ".m.0.attn.qkv.conv"), W.get(s + ".m.0.attn.proj.conv"),
                           W.get(s + ".m.0.attn.pe.conv"), W.attn_fq(s + ".m.0.attn"));
    dump(dir, "l10_attn", att);
    Tensor pb = psablock(W, s + ".m.0", b);
    dump(dir, "l10_psablock", pb);
    Tensor cc = concat({&a, &pb});
    return cv(W, s + ".cv2.conv", cc);
}

// C3k2 with attention (layer 22): m.0 = Sequential(Bottleneck m.0.0, PSABlock m.0.1).
static Tensor c3k2_attn(const Weights& W, int i, const Tensor& x) {
    const std::string s = std::to_string(i);
    Tensor t = cv(W, s + ".cv1.conv", x);
    const int c = W.split_of(i, t.C);
    Tensor a = slice_ch(t, 0, c), b = slice_ch(t, c, t.C - c);
    Tensor mb = bottleneck(W, s + ".m.0.0", b, true);
    mb = psablock(W, s + ".m.0.1", mb);
    Tensor cc = concat({&a, &b, &mb});
    return cv(W, s + ".cv2.conv", cc);
}

// SPPF (layer 9): cv1 -> 3x maxpool(k5,s1,p2) -> cat(4) -> cv2, + residual (YOLO26 sets shortcut,
// and c1==c2==512 here so self.add is True: forward returns cv2(cat) + x).
static Tensor sppf(const Weights& W, int i, const Tensor& x) {
    const std::string s = std::to_string(i);
    Tensor y0 = cv(W, s + ".cv1.conv", x);
    Tensor y1 = maxpool(y0, 5, 1, 2);
    Tensor y2 = maxpool(y1, 5, 1, 2);
    Tensor y3 = maxpool(y2, 5, 1, 2);
    Tensor cc = concat({&y0, &y1, &y2, &y3});
    Tensor y = cv(W, s + ".cv2.conv", cc);
    // SPPF shortcut (self.add). YOLO26 sets shortcut and c1==c2 here, so the residual is ALWAYS taken
    // -- including after channel compaction, where the add-union liveness rule keeps the two widths
    // equal by construction. Asserting beats the old `(y.C==x.C) ? add : y` ternary, which would have
    // silently dropped the residual (a real accuracy bug) if compaction ever broke that invariant.
    CSIM_HOST_ASSERT(y.C == x.C, "SPPF layer " + s + ": residual width mismatch, cv2 out C=" +
                                 std::to_string(y.C) + " vs input C=" + std::to_string(x.C));
    return add(y, x);
}

// Detect head branch for one scale. box branch cv2.i (2 conv3x3 + 1x1->4); cls branch cv3.i
// (2 x [dw3x3 + pw1x1] + 1x1->4). Returns concat(box, cls) = 8 channels. `pre` is "cv2"/"cv3"
// prefix root, e.g. "23.cv2" / "23.cv3" / "23.one2one_cv2" / "23.one2one_cv3".
static Tensor head_box(const Weights& W, const std::string& root, int i, const Tensor& f) {
    const std::string p = root + "." + std::to_string(i);
    Tensor y = cv(W, p + ".0.conv", f);
    y = cv(W, p + ".1.conv", y);
    return cv(W, p + ".2", y);              // bare 1x1 (bias, identity)
}
static Tensor head_cls(const Weights& W, const std::string& root, int i, const Tensor& f) {
    const std::string p = root + "." + std::to_string(i);
    Tensor y = cv(W, p + ".0.0.conv", f);   // dw3x3
    y = cv(W, p + ".0.1.conv", y);          // pw1x1
    y = cv(W, p + ".1.0.conv", y);          // dw3x3
    y = cv(W, p + ".1.1.conv", y);          // pw1x1
    return cv(W, p + ".2", y);              // bare 1x1
}

void run_trunk(const Weights& W, const Tensor& input, const std::string& dir) {
    Tensor x0  = cv(W, "0.conv", input);          dump(dir, "0", x0);
    Tensor x1  = cv(W, "1.conv", x0);             dump(dir, "1", x1);
    Tensor x2  = c3k2(W, 2, x1, 0);               dump(dir, "2", x2);
    Tensor x3  = cv(W, "3.conv", x2);             dump(dir, "3", x3);
    Tensor x4  = c3k2(W, 4, x3, 0);               dump(dir, "4", x4);
    Tensor x5  = cv(W, "5.conv", x4);             dump(dir, "5", x5);
    Tensor x6  = c3k2(W, 6, x5, 1);               dump(dir, "6", x6);
    Tensor x7  = cv(W, "7.conv", x6);             dump(dir, "7", x7);
    Tensor x8  = c3k2(W, 8, x7, 1);               dump(dir, "8", x8);
    Tensor x9  = sppf(W, 9, x8);                  dump(dir, "9", x9);
    Tensor x10 = c2psa(W, 10, x9, dir);           dump(dir, "10", x10);
    Tensor x11 = upsample2x(x10);                 dump(dir, "11", x11);
    Tensor x12 = concat({&x11, &x6});             dump(dir, "12", x12);   // Concat [-1, 6]
    Tensor x13 = c3k2(W, 13, x12, 1);             dump(dir, "13", x13);
    Tensor x14 = upsample2x(x13);                 dump(dir, "14", x14);
    Tensor x15 = concat({&x14, &x4});             dump(dir, "15", x15);   // Concat [-1, 4]
    Tensor x16 = c3k2(W, 16, x15, 1);             dump(dir, "16", x16);
    Tensor x17 = cv(W, "17.conv", x16);           dump(dir, "17", x17);
    Tensor x18 = concat({&x17, &x13});            dump(dir, "18", x18);   // Concat [-1, 13]
    Tensor x19 = c3k2(W, 19, x18, 1);             dump(dir, "19", x19);
    Tensor x20 = cv(W, "20.conv", x19);           dump(dir, "20", x20);
    Tensor x21 = concat({&x20, &x10});            dump(dir, "21", x21);   // Concat [-1, 10]
    Tensor x22 = c3k2_attn(W, 22, x21);           dump(dir, "22", x22);

    const Tensor* feats[3] = {&x16, &x19, &x22};   // Detect.f = [16, 19, 22]
    for (int i = 0; i < 3; ++i) {
        // The one2many head (23.cv2/23.cv3) is YOLOv10-style training-time auxiliary supervision.
        // NOTHING at inference consumes it, confirmed four independent ways: decode.h implements only
        // decode_o2o; the deployed OpenVINO IR that scores the reference 0.7546 has 102 conv nodes to
        // this graph's 126, exactly these 24 missing; they carry sa=0 in manifest_sq.txt (SmoothQuant
        // never assigned them scales, so they are not even quantized); and decode_ref.py feeds the o2o
        // maps into the o2m slots to satisfy split()'s 2*nl shape contract, which works precisely
        // because the values are ignored. On the FPGA it is 0.535 GMAC -- 12.9% of the model's 4.155 --
        // plus 0.273 M dead params, so the synthesizable kernel must NOT instantiate it.
        // Build with -DYOLO26_NO_O2M for that deployment shape; it stays on by default because
        // compare_cosine.py / compare_int8.py / compare_compact.py / decode_check.py and
        // csim_eval_map.py's Python-decode path all read the o2m_* dumps.
#ifndef YOLO26_NO_O2M
        Tensor box = head_box(W, "23.cv2", i, *feats[i]);
        Tensor cls = head_cls(W, "23.cv3", i, *feats[i]);
        Tensor o2m = concat({&box, &cls});
        dump(dir, "o2m_" + std::to_string(i), o2m);
#endif
        Tensor obox = head_box(W, "23.one2one_cv2", i, *feats[i]);
        Tensor ocls = head_cls(W, "23.one2one_cv3", i, *feats[i]);
        Tensor o2o = concat({&obox, &ocls});
        dump(dir, "o2o_" + std::to_string(i), o2o);
    }
}
