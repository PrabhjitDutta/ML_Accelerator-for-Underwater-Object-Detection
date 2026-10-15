// ZCU102 host for the shipping kernel (FPGA/4_bitstream).
// With -DY26_BOARD, conv2d() (layer_ops.h) sends every SmoothQuant integer conv (c.quant && c.asym) here; the
// rest of the trunk (FP32 attn.pe convs, attention, upsample, concat, head) runs on the A53s.
//   ./y26_board <weights_dir> <input.bin> <dump_dir>
//   ls frames/*.bin | ./y26_board <weights_dir> - <out_dir>     frame loop, one timing line per frame
//
// DDR layout (as the testbench): X = uint8 codes, ic*H*W; Wt = int8 codes at Y26_WIDX() inside Y26_WSTRIDE() per
// oc; wsc/bias/step_v/lo_v = float; Y = float at row stride YS.
// One DMA buffer, one first-fit heap (Arena): tensor storage, each conv's X codes (freed after the run) and every
// conv's weights (packed once on first use, never freed). The kernel writes Y straight into the output tensor
// when it is in the buffer with unpadded rows; otherwise into a scratch Y that is unpacked. A full buffer falls
// back to the heap (correct, slower). Async (default): independent branches run on threads; the kernel is one
// mutex, so one branch's PS work overlaps another's conv.
//
// Backends:
//   -DY26_BOARD_SIM  the buffer is host memory and the C-model y26_conv_top() reads it (everything but silicon).
//   default (board)  root, bitstream loaded, PL0 at 250 MHz, and a DDR buffer the kernel owns:
//     Y26_UDMABUF=udmabuf0   preferred: cached u-dma-buf buffer, cache-synced per run
//                            (insmod u-dma-buf.ko udmabuf0=67108864).
//     Y26_BUF_PHYS/_SIZE     fallback: uncached O_SYNC /dev/mem on a reserved-memory (no-map) window;
//                            X and Y are copied through Device memory every conv.
// Environment:
//   Y26_BOARD_CHECK=<substr>  also run matching convs ('.' = all) through the C-model and compare bit for bit.
//   Y26_PACK_MT=<elements>    pack X with OpenMP when a conv's X has >= this many elements (default 1048576).
//   Y26_PACK_SCHED=dynamic    pack X's OpenMP schedule (default static).
//   Y26_COHERENT=1            only with a bitstream that puts X/Y on coherent HPC ports: skip their cache syncs.
//   YOLO26_HEADS_ONLY=1       write only the 3 o2o head maps.
// -DY26_YC builds (direct codes, frame loop only; they pair with a bitstream built with -DY26_YC):
//   Y26_YC=0                  every conv in float mode (the kernel's codes go to a scrap block).
//   Y26_YC_CHECK=1            compare every direct code with the host quantizer of the kernel's float output, and
//                             every sum made on codes with the quantized float sum (floats are kept for it).
//   Y26_YC_HOLE=<phys addr>   board: point the output nobody reads at this address instead of DDR. It must be an
//                             address the interconnect does not map, so the write ends in its decode-error slave.
//   Y26_OPS=1                 the bitstream has the ops block (y26_ops_top): sums on codes, the copies out of padded
//                             20-wide code rows and the 2x upsamples are made by it, not by the ARM. Never set it
//                             on a bitstream without the block: its registers are not mapped there.
//   Y26_ATTN2=1               the bitstream's attention block is v2: it reads qkv's codes and writes proj's, with pe
//                             inside. Never set it on a bitstream with the first attention block.
// Load the bitstream with pynq Overlay('y26_zcu102.bit') or fpgautil (the .hwh beside it); PL0 must read
// ~249975000 Hz. The exit summary splits every conv into pack / in / kernel / out plus PS time between convs.
#include "../hls_kernel/conv_engine.h"
#include "../reference_model/layer_ops.h"
#ifdef Y26_YC
#include "../hls_kernel/ops_engine.h"
#endif
#ifdef Y26_ATTN_INT
#include "../hls_kernel/attention_engine.h"
#include "../hls_kernel/attention2_engine.h"
#endif
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <climits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// The bitstream's interface. A host built with other flags packs a layout the silicon does not read.
static_assert(Y26_ACT_WORD == 128 && Y26_WT_WORD == 64 && Y26_OUT_WORD == 256,
              "port widths must match the YA128 bitstream - build with board_host/build_board_host.sh");
#if !defined(Y26_YSTRIDE_PAD) || !defined(Y26_WT_TAPMAJOR) || !defined(Y26_OCPACK)
#error "not the shipping flag set - build with board_host/build_board_host.sh"
#endif
// Y26_HOST_EXACT (build_board_host.sh exact): float dequant, so the SIM trunk must equal the CPU trunk bit for bit -
// the packing test. It is never the silicon's arithmetic, so it is SIM-only.
#if (!defined(Y26_FX_DEQUANT) || !defined(Y26_SILU_LUT)) && !(defined(Y26_HOST_EXACT) && defined(Y26_BOARD_SIM))
#error "the bitstream is FX_DEQUANT + SILU_LUT - build with board_host/build_board_host.sh"
#endif
// The C-model reads the byte image through these types; that is only valid if they ARE the bytes.
static_assert(sizeof(y26_xw_t) == 16 && sizeof(y26_ww_t) == 8 && sizeof(y26_yw_t) == 32,
              "ap_uint storage is not the raw port word");

#define Y26_HOST_YS(ow) ((((ow) + Y26_EPI_WIDE - 1) / Y26_EPI_WIDE) * Y26_EPI_WIDE)

namespace {
using clk = std::chrono::steady_clock;
double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
size_t al(size_t n) { return (n + 4095) & ~size_t(4095); }   // every buffer 4 KB aligned

// First-fit heap over the buffer's byte offsets, coalescing on free. Weights take from the top (never freed),
// everything else from the bottom, so the permanent blocks do not fragment the activations.
struct Arena {
    std::mutex m;
    std::map<size_t, size_t> free_;                // offset -> bytes
    size_t used = 0, peak = 0;
    std::atomic<int> spilled{0};                   // tensors that did not fit and went to the plain heap
    size_t take(size_t n, bool top) {
        n = al(n);
        std::lock_guard<std::mutex> g(m);
        auto hit = free_.end();
        if (top) { for (auto it = free_.rbegin(); it != free_.rend(); ++it) if (it->second >= n) { hit = std::prev(it.base()); break; } }
        else     { for (auto it = free_.begin(); it != free_.end(); ++it) if (it->second >= n) { hit = it; break; } }
        if (hit == free_.end()) return SIZE_MAX;
        const size_t off = hit->first, rest = hit->second - n;
        free_.erase(hit);
        size_t r = off;
        if (top) { if (rest) free_[off] = rest; r = off + rest; }
        else if (rest) free_[off + n] = rest;
        used += n; if (used > peak) peak = used;
        return r;
    }
    void give(size_t off, size_t n) {
        n = al(n);
        std::lock_guard<std::mutex> g(m);
        auto it = free_.emplace(off, n).first;
        auto nx = std::next(it);
        if (nx != free_.end() && it->first + it->second == nx->first) { it->second += nx->second; free_.erase(nx); }
        if (it != free_.begin()) { auto pv = std::prev(it); if (pv->first + pv->second == it->first) { pv->second += it->second; free_.erase(it); } }
        used -= n;
    }
};
Arena& arena();

struct Stats {
    // Per conv: pack X, pack weights (first use only), in (host -> kernel), kernel, out (kernel -> Tensor).
    struct Row { std::string name; double px, pw, in, kern, out, wall; int check; };  // check: -1 off, 0 exact, >0 rows differ
    std::vector<Row> rows, prev;                   // prev: the last finished frame of a frame loop
    std::map<std::string, Y26Prof::E> prevprof;
    clk::time_point first, last, pfirst, plast;
    std::mutex m;
    ~Stats() {
        if (rows.empty() && !prev.empty()) { rows.swap(prev); first = pfirst; last = plast; Y26Prof::tab() = prevprof; }
        if (rows.empty()) return;
        Row t{"", 0, 0, 0, 0, 0, 0, 0};
        int nchk = 0, nbad = 0;
        std::printf("\n[board] %-30s %8s %8s %8s %9s %8s %9s  %s\n", "conv", "pack X", "pack W", "in", "kernel", "out",
                    "conv ms", "check");
        for (const Row& r : rows) {
            std::printf("[board] %-30s %8.3f %8.3f %8.3f %9.3f %8.3f %9.3f  %s\n", r.name.c_str(), r.px, r.pw, r.in,
                        r.kern, r.out, r.wall, r.check < 0 ? "" : r.check == 0 ? "BIT-EXACT" : "MISMATCH");
            t.px += r.px; t.pw += r.pw; t.in += r.in; t.kern += r.kern; t.out += r.out; t.wall += r.wall;
            if (r.check >= 0) { ++nchk; nbad += r.check > 0; }
        }
        const double span = ms(first, last);
        std::printf("[board] %zu kernel convs: kernel %.3f ms, incl. host pack/copy %.3f ms\n", rows.size(), t.kern, t.wall);
        std::printf("[board] split: pack X %.1f | pack W %.1f | in %.1f | kernel %.1f | out %.1f | PS between convs %.1f"
                    " | first->last conv %.1f ms%s\n", t.px, t.pw, t.in, t.kern, t.out, span - t.wall, span,
                    y26_async() ? "  (async: convs overlap, so 'between' = span - sum of conv walls)" : "");
        std::printf("[board] buffer: peak %.1f MB in use, %d tensors spilled to the heap\n", arena().peak / 1048576.0,
                    arena().spilled.load());
        if (nchk) std::printf("[board] CHECK: %d convs vs C-model, %d MISMATCH\n", nchk, nbad);
        double ptot = 0;   // PS profile: self ms per trunk op; the kernel-conv row is the split above
        for (const auto& e : Y26Prof::tab()) ptot += e.second.ms;
        for (const auto& e : Y26Prof::tab())
            std::printf("[prof] %-30s %5d calls %9.2f ms %5.1f%%\n", e.first.c_str(), e.second.n, e.second.ms,
                        100 * e.second.ms / ptot);
        std::printf("[prof] %-30s %15.2f ms\n", "trunk total", ptot);
    }
} g_stats;

// Byte offsets of the seven kernel buffers in one image.
// yc: the code output (-DY26_YC). ynone / ycnone: nobody reads that output.
struct Layout { size_t x, wt, wsc, bias, stp, lo, y, end; int yq = 0; size_t qs = 0, qlo = 0, qst = 0;
                size_t yc = 0; bool ynone = false, ycnone = false; };

void run_cmodel(uint8_t* b, const Layout& L, const ConvW& c, int H, int W) {
    y26_conv_top(reinterpret_cast<const y26_xw_t*>(b + L.x), reinterpret_cast<const y26_ww_t*>(b + L.wt),
                 reinterpret_cast<const float*>(b + L.wsc), reinterpret_cast<const float*>(b + L.bias),
                 reinterpret_cast<const float*>(b + L.stp), reinterpret_cast<const float*>(b + L.lo),
                 reinterpret_cast<y26_yw_t*>(b + L.y),
                 H, W, c.oc, c.ic, c.kh, c.kw, c.sh, c.sw, c.ph, c.pw, c.groups, c.act, c.perch ? 1 : 0,
                 c.sa, c.lo
#ifdef Y26_YQ8
                 , L.yq, reinterpret_cast<const float*>(b + L.qs), reinterpret_cast<const float*>(b + L.qlo),
                 reinterpret_cast<const float*>(b + L.qst)
#endif
#ifdef Y26_YC
                 , reinterpret_cast<y26_cw_t*>(b + L.yc), L.yq, reinterpret_cast<const float*>(b + L.qs),
                 reinterpret_cast<const float*>(b + L.qlo)
#endif
                 );
}
}  // namespace

#ifdef Y26_BOARD_SIM
namespace {
// The buffer is plain host memory; the C-model is the kernel.
struct Buf {
    std::vector<uint64_t> mem;                     // uint64_t: 8 B aligned for the port-word views
    uint8_t* host = nullptr;
    size_t size = 0;
    Buf() {
        const char* s = std::getenv("Y26_BUF_SIZE");
        size = s ? std::strtoull(s, nullptr, 0) : (size_t)64 << 20;
        mem.assign(size / 8, 0);
        host = reinterpret_cast<uint8_t*>(mem.data());
    }
    std::mutex km;                                 // one kernel
    static constexpr bool coh = false;
    static constexpr uint64_t hole = 0;
    void to_dev(size_t, size_t) {}
    void for_write(size_t, size_t) {}
    void from_dev(size_t, size_t) {}
    double run(const Layout& L, const ConvW& c, int H, int W) {
        std::lock_guard<std::mutex> g(km);
        const auto t = clk::now();
        run_cmodel(host, L, c, H, W);
        return ms(t, clk::now());
    }
#ifdef Y26_ATTN_INT
    std::mutex am;                                 // one attention block
    void attn(size_t x, size_t y, int nh, int n, AttnInt c) {
        std::lock_guard<std::mutex> g(am);
        y26_attn_top(reinterpret_cast<const ap_uint<128>*>(host + x), reinterpret_cast<ap_uint<128>*>(host + y), nh, n, c.m, c.g);
    }
    void attn2(size_t x, size_t d, size_t par, int nh, int n, int w, AttnInt c) {
        std::lock_guard<std::mutex> g(am);
        auto p = [&](size_t o) { return reinterpret_cast<ap_uint<128>*>(host + o); };
        y26_attn2_top(p(x), p(d), p(par), nh, n, w, c.m, c.g);
    }
#endif
#ifdef Y26_YC
    std::mutex om;                                 // one ops block
    void ops(size_t sa, size_t sb, size_t dst, size_t par, int C, int units, int amode, int bmode) {
        std::lock_guard<std::mutex> g(om);
        auto p = [&](size_t o) { return reinterpret_cast<ap_uint<128>*>(host + o); };
        y26_ops_top(p(sa), p(sb), p(dst), p(par), C, units, amode, bmode);
    }
#endif
};
}  // namespace
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {
const uint64_t REG_BASE = 0x80000000ull;   // .hwh: y26_conv_top s_axi_control, 64 KB

void barrier() {
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

std::string slurp(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "r");
    if (!f) throw std::runtime_error("cannot read " + p);
    char buf[64] = {};
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    return std::string(buf, n);
}
struct Buf {
    volatile uint32_t* reg = nullptr;
    uint8_t* map = nullptr;          // the DMA buffer as mapped
    uint8_t* host = nullptr;         // where packing writes: map itself (cached) or mirror (uncached /dev/mem)
    std::vector<uint64_t> mirror;
    uint64_t phys = 0;
    size_t size = 0;
    std::string ub;                  // u-dma-buf sysfs dir when CACHED; empty = uncached /dev/mem
    std::mutex km, sm;               // the kernel; the sysfs sync sequence
    int fo = -1, fs = -1, fdr = -1, fdev = -1, fcpu = -1;   // sync_offset/_size/_direction/_for_device/_for_cpu
    bool cmd = false;                // Y26_SYNC_CMD=1: one packed write per sync
    bool coh = false;                // Y26_COHERENT=1 (cached buffer only): X/Y ride the coherent HPC ports, no sync
    uint64_t hole = 0;               // Y26_YC_HOLE: where an output nobody reads is pointed (0 = into the buffer)
    Buf() {
        if (const char* h = std::getenv("Y26_YC_HOLE")) hole = std::strtoull(h, nullptr, 0);
        // Y26_UDMABUF=<name>: a cached DMA buffer. The HP ports are not cache-coherent, so every run is bracketed by
        // cache maintenance (sync()). Default: Y26_BUF_PHYS/Y26_BUF_SIZE through uncached O_SYNC /dev/mem.
        const char* u = std::getenv("Y26_UDMABUF");
        int fd;
        if (u && *u) {
            for (const char* cls : {"/sys/class/u-dma-buf/", "/sys/class/udmabuf/"})
                if (FILE* f = std::fopen((std::string(cls) + u + "/phys_addr").c_str(), "r")) { std::fclose(f); ub = std::string(cls) + u; break; }
            if (ub.empty()) throw std::runtime_error(std::string("Y26_UDMABUF=") + u + ": no sysfs entry (u-dma-buf module loaded?)");
            phys = std::strtoull(slurp(ub + "/phys_addr").c_str(), nullptr, 0);
            size = std::strtoull(slurp(ub + "/size").c_str(), nullptr, 0);
            fd = open((std::string("/dev/") + u).c_str(), O_RDWR);    // no O_SYNC: cached mapping
            if (fd < 0) throw std::runtime_error(std::string("open /dev/") + u + " failed");
        } else {
            const char* p = std::getenv("Y26_BUF_PHYS");
            const char* s = std::getenv("Y26_BUF_SIZE");
            // No default window: a guessed physical address is a write into memory Linux owns.
            if (!p || !s) throw std::runtime_error("set Y26_UDMABUF, or Y26_BUF_PHYS and Y26_BUF_SIZE to a reserved DDR window");
            phys = std::strtoull(p, nullptr, 0);
            size = std::strtoull(s, nullptr, 0);
            fd = open("/dev/mem", O_RDWR | O_SYNC);
            if (fd < 0) throw std::runtime_error("open /dev/mem failed (root?)");
        }
        if (phys & 4095) throw std::runtime_error("buffer phys must be 4 KB aligned");
        // HP0-3 see DDR_LOW 0..0x7FFFFFFF and DDR_HIGH 0x8_0000_0000..0xF_FFFF_FFFF (.hwh).
        const bool lo = phys + size <= 0x80000000ull;
        const bool hi = phys >= 0x800000000ull && phys + size <= 0x1000000000ull;
        if (!size || !(lo || hi)) throw std::runtime_error("buffer is not in DDR the HP ports can reach");
        if (!ub.empty()) {
            auto opn = [&](const char* a) {
                const int f = open((ub + "/" + a).c_str(), O_WRONLY);
                if (f < 0) throw std::runtime_error("cannot open " + ub + "/" + a);
                return f;
            };
            fo = opn("sync_offset"); fs = opn("sync_size"); fdr = opn("sync_direction");
            fdev = opn("sync_for_device"); fcpu = opn("sync_for_cpu");
            const char* sc = std::getenv("Y26_SYNC_CMD");
            cmd = sc && std::strcmp(sc, "1") == 0;
            // Only with a bitstream that puts gmem_act/out on coherent HPC ports, with CCI snooping on. Weights stay on
            // non-coherent ports and keep their sync. Check once with Y26_BOARD_CHECK=.
            const char* co = std::getenv("Y26_COHERENT");
            coh = co && std::strcmp(co, "1") == 0;
        }
        void* d = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, ub.empty() ? (off_t)phys : 0);
        close(fd);
        const int fm = open("/dev/mem", O_RDWR | O_SYNC);                 // registers are always Device memory
        if (fm < 0) throw std::runtime_error("open /dev/mem failed (root?)");
        void* r = mmap(nullptr, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, fm, (off_t)REG_BASE);
        close(fm);
        if (r == MAP_FAILED || d == MAP_FAILED) throw std::runtime_error("mmap failed");
        reg = static_cast<volatile uint32_t*>(r);
        map = static_cast<uint8_t*>(d);
        // u-dma-buf maps pages on fault: take all the faults here, once.
        if (!ub.empty()) for (size_t i = 0; i < size; i += 4096) (void)*reinterpret_cast<volatile const uint8_t*>(map + i);
        if (ub.empty()) { mirror.assign(size / 8, 0); host = reinterpret_cast<uint8_t*>(mirror.data()); }
        else host = map;
        std::printf("[board] buffer %s: phys 0x%llx, %zu B\n", ub.empty() ? "/dev/mem (uncached, host mirror)" : ub.c_str(),
                    (unsigned long long)phys, size);
    }
    // u-dma-buf cache maintenance on [off, off+n). dir: 1 = to device (clean), 2 = from device (invalidate).
    // The sysfs files stay open (one pwrite each). Y26_SYNC_CMD=1: one write of u-dma-buf's packed command
    // "0x<offset:32><size&~15 | dir<<2 | 1>", only for a driver that parses it (check with Y26_BOARD_CHECK=.).
    static void wr(int f, const char* s) {
        const ssize_t k = (ssize_t)std::strlen(s);
        if (pwrite(f, s, k, 0) != k) throw std::runtime_error("u-dma-buf sync write failed");
    }
    void sync(bool for_device, size_t off, size_t n, int dir) {
        char b[40];
        std::lock_guard<std::mutex> g(sm);
        const unsigned long long sz = (n + 15) & ~15ull;
        if (cmd && off < (1ull << 32) && sz < (1ull << 32) && off + sz <= size) {
            std::snprintf(b, sizeof b, "0x%08llX%08llX", (unsigned long long)off, sz | (unsigned)dir << 2 | 1);
            wr(for_device ? fdev : fcpu, b);
            return;
        }
        std::snprintf(b, sizeof b, "%llu", (unsigned long long)off); wr(fo, b);
        std::snprintf(b, sizeof b, "%llu", (unsigned long long)n);   wr(fs, b);
        std::snprintf(b, sizeof b, "%d", dir);                        wr(fdr, b);
        wr(for_device ? fdev : fcpu, "1");
    }
    // What the host wrote into [off, off+n) becomes what the kernel reads. Cached: clean the lines out to DDR.
    // Uncached: plain 64-bit volatile stores - never memcpy on Device memory (glibc may use DC ZVA or
    // unaligned accesses there, and both fault).
    void to_dev(size_t off, size_t n) {
        if (!n) return;
        if (!ub.empty()) { sync(true, off, n, 1); return; }
        const uint64_t* s = reinterpret_cast<const uint64_t*>(host + off);
        volatile uint64_t* t = reinterpret_cast<volatile uint64_t*>(map + off);
        for (size_t i = 0; i < n / 8; ++i) t[i] = s[i];
    }
    // [off, off+n) is about to be written by the kernel. Cached: clean it first - the output tensor reuses freed
    // memory, and a dirty line left there could later be evicted OVER the kernel's Y.
    void for_write(size_t off, size_t n) { if (n && !ub.empty()) sync(true, off, n, 2); }
    // What the kernel wrote into [off, off+n) becomes visible to the host. Cached: invalidate first, so no
    // stale (or speculatively fetched) line hides it.
    void from_dev(size_t off, size_t n) {
        if (!n) return;
        if (!ub.empty()) { sync(false, off, n, 2); return; }
        uint64_t* t = reinterpret_cast<uint64_t*>(host + off);
        const volatile uint64_t* s = reinterpret_cast<const volatile uint64_t*>(map + off);
        for (size_t i = 0; i < n / 8; ++i) t[i] = s[i];
    }
    void w32(unsigned off, uint32_t v) { reg[off / 4] = v; }
    void w64(unsigned off, uint64_t v) { w32(off, (uint32_t)v); w32(off + 4, (uint32_t)(v >> 32)); }
    void wf(unsigned off, float f) { uint32_t u; std::memcpy(&u, &f, 4); w32(off, u); }
    // Returns kernel time (ap_start -> ap_done) in ms.
    double run(const Layout& L, const ConvW& c, int H, int W) {
        std::lock_guard<std::mutex> g(km);
        if (!(reg[0] & 4)) throw std::runtime_error("kernel not idle before " + c.name);
        w64(0x10, phys + L.x);    w64(0x1c, phys + L.wt);   w64(0x28, phys + L.wsc); w64(0x34, phys + L.bias);
        w64(0x40, phys + L.stp);  w64(0x4c, phys + L.lo);   w64(0x58, L.ynone && hole ? hole : phys + L.y);
        w32(0x64, H);          w32(0x6c, W);          w32(0x74, c.oc);      w32(0x7c, c.ic);
        w32(0x84, c.kh);       w32(0x8c, c.kw);       w32(0x94, c.sh);      w32(0x9c, c.sw);
        w32(0xa4, c.ph);       w32(0xac, c.pw);       w32(0xb4, c.groups);  w32(0xbc, c.act);
        w32(0xc4, c.perch ? 1 : 0);  wf(0xcc, c.sa);  wf(0xd4, c.lo);
#ifdef Y26_YQ8
        // Offsets of the yq/qs/qlo/qst registers, from the yq build's driver header (xy26_conv_top_hw.h).
        enum { Y26_REG_YQ = 0xdc, Y26_REG_QS = 0xe4, Y26_REG_QLO = 0xf0, Y26_REG_QST = 0xfc };
        // yq is written on every run: registers persist, and a float conv after a coded one must not inherit yq=1.
        w32(Y26_REG_YQ, (uint32_t)L.yq);
        w64(Y26_REG_QS, phys + L.qs);  w64(Y26_REG_QLO, phys + L.qlo);  w64(Y26_REG_QST, phys + L.qst);
#endif
#ifdef Y26_YC
        // Offsets of the Yc/yq/q_k/q_b registers, from the direct-codes build's driver header (xy26_conv_top_hw.h).
        enum { Y26_REG_YC = 0xdc, Y26_REG_YQ = 0xe8, Y26_REG_QK = 0xf0, Y26_REG_QB = 0xfc };
        w64(Y26_REG_YC, L.ycnone && hole ? hole : phys + L.yc);
        w32(Y26_REG_YQ, (uint32_t)L.yq);           // every run: registers persist
        w64(Y26_REG_QK, phys + L.qs);  w64(Y26_REG_QB, phys + L.qlo);
#endif
        barrier();
        static std::unordered_map<const ConvW*, double> seen;   // this conv's kernel ms last frame (under km)
        const auto it = seen.find(&c);
        const auto t0 = clk::now();
        w32(0x00, 1);                                  // ap_start
        // Async: another branch can use this core while the PL works, so sleep through 70% of a >1 ms conv, then spin.
        // Serial runs and the first frame spin throughout.
        if (it != seen.end() && it->second > 1.0 && y26_async())
            std::this_thread::sleep_for(std::chrono::microseconds((long)(it->second * 700)));
        while (!(reg[0] & 2))                          // ap_done (clear-on-read)
            if (ms(t0, clk::now()) > 5000) throw std::runtime_error("kernel timeout (5 s) on " + c.name);
        const double k = ms(t0, clk::now());
        seen[&c] = k;
        barrier();
        return k;
    }
#ifdef Y26_ATTN_INT
    // y26_attn_top s_axi_control, 64 KB (.hwh); register offsets from its driver header (xy26_attn_top_hw.h).
    static constexpr uint64_t ATTN_REG_BASE = 0x80010000ull;
    volatile uint32_t* areg = nullptr;
    std::mutex am;
    void attn(size_t x, size_t y, int nh, int n, AttnInt c) {
        std::lock_guard<std::mutex> g(am);
        if (!areg) {
            const int fm = open("/dev/mem", O_RDWR | O_SYNC);
            if (fm < 0) throw std::runtime_error("open /dev/mem failed (root?)");
            void* r = mmap(nullptr, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, fm, (off_t)ATTN_REG_BASE);
            close(fm);
            if (r == MAP_FAILED) throw std::runtime_error("mmap of the attention block failed");
            areg = static_cast<volatile uint32_t*>(r);
        }
        if (!(areg[0] & 4)) throw std::runtime_error("attention block not idle");
        auto w = [&](unsigned off, uint64_t v) { areg[off / 4] = (uint32_t)v; };
        w(0x10, phys + x);  w(0x14, (phys + x) >> 32);
        w(0x1c, phys + y);  w(0x20, (phys + y) >> 32);
        w(0x28, nh);  w(0x30, n);  w(0x38, c.m);  w(0x40, c.g);
        barrier();
        const auto t0 = clk::now();
        w(0x00, 1);                                    // ap_start
        while (!(areg[0] & 2))                         // ap_done (clear-on-read)
            if (ms(t0, clk::now()) > 5000) throw std::runtime_error("attention block timeout (5 s)");
        barrier();
    }
    // The v2 block has its own address: a host that drives the first block on a v2 bitstream faults instead of running.
    static constexpr uint64_t ATTN2_REG_BASE = 0x80040000ull;
    volatile uint32_t* a2reg = nullptr;
    void attn2(size_t x, size_t d, size_t par, int nh, int n, int wd, AttnInt c) {
        std::lock_guard<std::mutex> g(am);
        if (!a2reg) {
            const int fm = open("/dev/mem", O_RDWR | O_SYNC);
            if (fm < 0) throw std::runtime_error("open /dev/mem failed (root?)");
            void* r = mmap(nullptr, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, fm, (off_t)ATTN2_REG_BASE);
            close(fm);
            if (r == MAP_FAILED) throw std::runtime_error("mmap of the attention block failed");
            a2reg = static_cast<volatile uint32_t*>(r);
        }
        if (!(a2reg[0] & 4)) throw std::runtime_error("attention block not idle");
        auto w = [&](unsigned off, uint64_t v) { a2reg[off / 4] = (uint32_t)v; };
        auto a = [&](unsigned off, size_t o) { w(off, phys + o);  w(off + 4, (phys + o) >> 32); };
        a(0x10, x);  a(0x1c, d);  a(0x28, par);
        w(0x34, nh);  w(0x3c, n);  w(0x44, wd);  w(0x4c, c.m);  w(0x54, c.g);
        barrier();
        const auto t0 = clk::now();
        w(0x00, 1);                                    // ap_start
        while (!(a2reg[0] & 2))                        // ap_done (clear-on-read)
            if (ms(t0, clk::now()) > 5000) throw std::runtime_error("attention block timeout (5 s)");
        barrier();
    }
#endif
#ifdef Y26_YC
    // y26_ops_top s_axi_control, 64 KB (.hwh); register offsets from its driver header (xy26_ops_top_hw.h).
    static constexpr uint64_t OPS_REG_BASE = 0x80030000ull;
    volatile uint32_t* oreg = nullptr;
    std::mutex om;
    void ops(size_t sa, size_t sb, size_t dst, size_t par, int C, int units, int amode, int bmode) {
        std::lock_guard<std::mutex> g(om);
        if (!oreg) {
            const int fm = open("/dev/mem", O_RDWR | O_SYNC);
            if (fm < 0) throw std::runtime_error("open /dev/mem failed (root?)");
            void* r = mmap(nullptr, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, fm, (off_t)OPS_REG_BASE);
            close(fm);
            if (r == MAP_FAILED) throw std::runtime_error("mmap of the ops block failed");
            oreg = static_cast<volatile uint32_t*>(r);
        }
        if (!(oreg[0] & 4)) throw std::runtime_error("ops block not idle");
        auto w = [&](unsigned off, uint64_t v) { oreg[off / 4] = (uint32_t)v; };
        auto a = [&](unsigned off, size_t o) { w(off, phys + o);  w(off + 4, (phys + o) >> 32); };
        a(0x10, sa);  a(0x1c, sb);  a(0x28, dst);  a(0x34, par);
        w(0x40, C);  w(0x48, units);  w(0x50, amode);  w(0x58, bmode);
        barrier();
        const auto t0 = clk::now();
        w(0x00, 1);                                    // ap_start
        while (!(oreg[0] & 2))                         // ap_done (clear-on-read)
            if (ms(t0, clk::now()) > 5000) throw std::runtime_error("ops block timeout (5 s)");
        barrier();
    }
#endif
};
}  // namespace
#endif

namespace {
Buf& buf() { static Buf b; return b; }
Arena& arena() {
    static Arena* a = [] { auto* r = new Arena; r->free_[0] = buf().size & ~size_t(4095); return r; }();   // leaked: tensors may outlive statics
    return *a;
}
// One arena block, released on scope exit (a conv's X codes, or its scratch Y).
struct Region {
    size_t off = SIZE_MAX, n;
    explicit Region(size_t bytes, const std::string& who) : n(bytes) {
        if (n && (off = arena().take(n, false)) == SIZE_MAX) throw std::runtime_error(who + ": DMA buffer full - enlarge it");
    }
    ~Region() { if (off != SIZE_MAX) arena().give(off, n); }
    Region(const Region&) = delete;
};

// Weights + scales of one conv, packed once at the top of the arena on first use, keyed by the ConvW.
struct WSlot { size_t wt, wsc, bias, stp, lo, qs, qlo, qst; };
std::unordered_map<const ConvW*, WSlot> g_wslots;
std::mutex g_wmu;

// -DY26_YQ8 code-mode edges: a conv whose output feeds exactly one kernel conv runs with yq=1. The kernel writes
// that consumer's uint8 codes into Y, the host narrows the slots to bytes in place, and the consumer copies
// them into X instead of quantizing. Edges are read off yolo26_network.cpp; a coded tensor reaching any other
// consumer throws (y26_take_codes). Enabled by y26_board_weights() (frame loop) AND Y26_YQ=1.
const Weights* g_W = nullptr;
// -DY26_YC (direct codes) uses the same table: the kernel writes the codes into the X block (Codes::xr) of one conv
// that reads the tensor (cons), at plane `off`, and no host pass touches them. Any conv that reads those channels
// through cons's quantizer takes the block, or a run of its planes, as its X. fl: the float output is kept too,
// for readers on another grid and for the host's adds. The entry lives as long as the tensor's storage.
struct Codes { const ConvW* cons; int off; std::shared_ptr<Region> xr; bool fl = false;
               bool add = false;                   // a residual branch coded for a sum on codes (add_bo), not cons's codes yet
               int by = 0; };                      // who wrote these bytes: 0 the kernel, 1 the ARM (and cleaned them), 2 a PL side block
long g_xkern = 0, g_xcode = 0, g_xall = 0;         // conv input bytes this frame: the kernel's as written / made by the ARM from codes / all
long g_xops = 0;                                    // ... / made by the PL side blocks (ops, attention)
long g_ia_n = 0, g_ia_1 = 0, g_ia_3 = 0;           // Y26_YC_CHECK: sums on codes, off the float sum's code by more than 1 / 3
long g_up_n = 0, g_up_1 = 0, g_up_3 = 0;           // Y26_YC_CHECK: upsamples from codes, off the float's code by more than 1 / 3
long g_yc_n = 0, g_yc_diff = 0, g_yc_far = 0;      // Y26_YC_CHECK totals this frame (under g_cmu)
int g_fuse_checked = 0;                             // fused sources verified by Y26_FUSE_CHECK (exit summary)
int g_ahead_checked = 0;                            // prepacked / sibling-shared / uint8-table sources verified
std::unordered_map<const float*, Codes> g_codes;   // coded tensor storage -> its one consumer
int g_ncoded = 0;                                   // edges coded this frame (frame line)
std::mutex g_cmu;
bool yc_check() { static const bool v = [] { const char* e = std::getenv("Y26_YC_CHECK"); return e && *e == '1'; }(); return v; }
// This frame's X block of a conv. Every conv that writes codes into it and its prepack get the same block; the
// conv takes it out when it runs.
std::map<std::pair<int, const ConvW*>, std::shared_ptr<Region>> g_xb;   // under g_cmu
std::shared_ptr<Region> xblock(const ConvW& c, size_t HW, bool take = false) {
    std::lock_guard<std::mutex> g(g_cmu);
    const auto key = std::make_pair(y26_frame(), &c);
    const auto it = g_xb.find(key);
    if (take) {
        if (it == g_xb.end()) return nullptr;
        std::shared_ptr<Region> r = std::move(it->second);
        g_xb.erase(it);
        return r;
    }
    if (it != g_xb.end()) return it->second;
    return g_xb[key] = std::make_shared<Region>(al((size_t)c.ic * HW), c.name);
}
// A residual branch as the kernel codes it for a sum on codes: at the reader's scale k = ssc/step, from offset bo.
// bo covers SiLU's minimum but does not reach below the reader's own zero.
float add_bo(const ConvW& q, int ch) {
    const float k = q.ssc[ch] / q.step_of(ch);
    return -std::min(std::ceil(0.2785f * k), std::ceil(-q.lo_of(ch) / q.step_of(ch)));
}
#ifdef Y26_YC
bool attn2_on() { static const bool v = [] { const char* e = std::getenv("Y26_ATTN2"); return e && *e == '1'; }(); return v; }
bool ops_on() { static const bool v = [] { const char* e = std::getenv("Y26_OPS"); return e && *e == '1'; }(); return v; }
// The ops block's words for input channels [ch0, ch0 + C) of c made from the codes of s's input channels from soff
// on: s's codes on c's grid; sum: with the kernel's branch codes (add_bo) to be added. In the buffer for good, made
// at first use. SIZE_MAX: an operand does not fit the block.
std::map<std::tuple<const ConvW*, int, const ConvW*, int, bool>, size_t> g_opar;
std::mutex g_opm;
size_t ops_par(const ConvW& c, int ch0, const ConvW& s, int soff, int C, bool sum) {
    std::lock_guard<std::mutex> g(g_opm);
    const auto key = std::make_tuple(&c, ch0, &s, soff, sum);
    const auto it = g_opar.find(key);
    if (it != g_opar.end()) return it->second;
    std::vector<OpsPar> p(C);
    bool ok = C <= Y26_OPS_CMAX;
    for (int k = 0; k < C && ok; ++k) {
        const int ic = ch0 + k, js = soff + k;
        const double u = (double)c.ssc[ic] / c.step_of(ic) / s.ssc[js];
        const double A = s.step_of(js) * u * 65536.;
        const double Bc = (s.lo_of(js) * u - (double)c.lo_of(ic) / c.step_of(ic) + (sum ? add_bo(c, ic) : 0.f)) * 65536.;
        ok = A >= 0 && A < Y26_OPS_A_MAX - 1 && std::fabs(Bc) < 2147483000.;
        if (ok) p[k] = {(uint32_t)std::llround(A), (int32_t)std::llround(Bc), {0, 0}};
    }
    size_t off = SIZE_MAX;
    if (ok) {
        const size_t n = al((size_t)C * sizeof(OpsPar));
        if ((off = arena().take(n, true)) == SIZE_MAX) throw std::runtime_error(c.name + ": no room for the ops block's words - enlarge the DMA buffer");
        std::memset(buf().host + off, 0, n);
        std::memcpy(buf().host + off, p.data(), (size_t)C * sizeof(OpsPar));
        buf().to_dev(off, n);
    }
    return g_opar[key] = off;
}
long g_oc_calls = 0, g_oc_n = 0, g_oc_bad = 0;     // Y26_BOARD_CHECK: ops block calls / codes compared / differing (under g_cmu)
// One call of the ops block. Y26_BOARD_CHECK: its result against y26_ops_ref on the same bytes.
void ops_run(Buf& B, size_t sa, size_t sb, size_t dst, size_t par, int C, int units, int amode, int bmode) {
    static const char* sel = std::getenv("Y26_BOARD_CHECK");
    if (!(sel && *sel)) { B.ops(sa, sb, dst, par, C, units, amode, bmode);  return; }
    const size_t pn = (size_t)units * Y26_OPS_UNIT, n = (size_t)C * pn;
    const size_t na = (size_t)C * y26_ops_src_bytes(pn, amode), nb = bmode ? (size_t)C * y26_ops_src_bytes(pn, bmode == 2) : 0;
    B.from_dev(sa, na);
    if (nb) B.from_dev(sb, nb);
    const std::vector<uint8_t> a(B.host + sa, B.host + sa + na), b(B.host + sb, B.host + sb + nb);   // b may be the destination
    std::vector<uint8_t> ref(n);
    y26_ops_ref(a.data(), b.data(), ref.data(), reinterpret_cast<const OpsPar*>(B.host + par), C, units, amode, bmode);
    B.ops(sa, sb, dst, par, C, units, amode, bmode);
    B.from_dev(dst, n);
    long bad = 0;
    for (size_t i = 0; i < n; ++i) bad += ref[i] != B.host[dst + i];
    if (bad) std::fprintf(stderr, "[board] CHECK ops block C=%d units=%d amode=%d bmode=%d: %ld of %zu codes differ\n", C, units, amode, bmode, bad, n);
    std::lock_guard<std::mutex> g(g_cmu);
    ++g_oc_calls;  g_oc_n += (long)n;  g_oc_bad += bad;
}
// The words of a plain copy: A = 1, B = 0 on every channel.
size_t ops_ident() {
    static const size_t off = [] {
        const size_t n = al((size_t)Y26_OPS_CMAX * sizeof(OpsPar)), o = arena().take(n, true);
        if (o == SIZE_MAX) throw std::runtime_error("no room for the ops block's words - enlarge the DMA buffer");
        OpsPar* p = reinterpret_cast<OpsPar*>(buf().host + o);
        for (int i = 0; i < Y26_OPS_CMAX; ++i) p[i] = {65536u, 0, {0, 0}};
        buf().to_dev(o, n);
        return o;
    }();
    return off;
}
#endif
// Input channels [a0, a0 + n) of a and [b0, b0 + n) of b go through the same quantizer, byte for byte.
static bool grid_eq(const ConvW& a, int a0, const ConvW& b, int b0, int n) {
    if (a0 < 0 || b0 < 0 || a0 + n > a.ic || b0 + n > b.ic) return false;
    if (std::memcmp(a.ssc.data() + a0, b.ssc.data() + b0, (size_t)n * 4)) return false;
    for (int i = 0; i < n; ++i) {
        const float l0 = a.lo_of(a0 + i), l1 = b.lo_of(b0 + i), s0 = a.step_of(a0 + i), s1 = b.step_of(b0 + i);
        if (std::memcmp(&l0, &l1, 4) || std::memcmp(&s0, &s1, 4)) return false;
    }
    return true;
}
// producer -> (consumer, 0 = its first input source / 1 = its last)
// Shared edges (-DY26_YC only): the consumer has other sources, or other convs (same) read the tensor's last
// channels too. fl: the tensor is also read as float. A reader in `same` on another grid makes the edge fl.
// with: the edge feeds a sum on codes (the kernel codes the branch with add_bo); the producer of the sum's other
// input. flif: the floats are needed only while that producer's sum is not made on codes.
// up: the floats are kept only for this conv's 2x upsample of the tensor (its first source).
struct YqEdge { std::string q; int tail; bool shared = false, fl = false; std::vector<std::string> same; std::string with, flif, up; };
const std::unordered_map<std::string, YqEdge>& yq_edges() {
    static const std::unordered_map<std::string, YqEdge> E = [] {
        std::unordered_map<std::string, YqEdge> e;
        auto ed = [&](const std::string& p, const std::string& q, int tail) { e[p] = {q, tail}; };
        auto sh = [&](const std::string& p, const std::string& q, int tail, bool f, std::vector<std::string> same) {
            e[p] = {q, tail, true, f, std::move(same)};
        };
        for (const char* l : {"2", "4"})                              // t -> cv2's head; b = t's tail -> m.0.cv1, and the add
            sh(std::string(l) + ".cv1.conv", std::string(l) + ".cv2.conv", 0, true, {std::string(l) + ".m.0.cv1.conv"});
        for (const char* l : {"6", "8", "13", "16", "19"}) {          // t -> cv2's head; b -> the C3k's cv1 and cv2
            const std::string m = std::string(l) + ".m.0";
            sh(std::string(l) + ".cv1.conv", std::string(l) + ".cv2.conv", 0, false, {m + ".cv1.conv", m + ".cv2.conv"});
            sh(m + ".cv1.conv", m + ".m.0.cv1.conv", 0, true, {});    // and the bottleneck's residual add
        }
        sh("4.cv2.conv", "16.cv1.conv", 1, false, {"5.conv"});        // x4: 16.cv1's last source, and 5.conv
        sh("6.cv2.conv", "13.cv1.conv", 1, false, {"7.conv"});        // x6: 13.cv1's last source, and 7.conv
        sh("16.cv2.conv", "17.conv", 0, false, {"23.one2one_cv2.0.0.conv", "23.one2one_cv3.0.0.0.conv"});
        sh("19.cv2.conv", "20.conv", 0, false, {"23.one2one_cv2.1.0.conv", "23.one2one_cv3.1.0.0.conv"});
        ed("0.conv", "1.conv", 0);      ed("1.conv", "2.cv1.conv", 0);  ed("2.cv2.conv", "3.conv", 0);
        ed("3.conv", "4.cv1.conv", 0);  ed("5.conv", "6.cv1.conv", 0);  ed("7.conv", "8.cv1.conv", 0);
        // NOT 8.cv2 -> 9.cv1: SPPF's residual add(y, x8) reads x8 as float too.
        for (const char* b : {"2.m.0", "4.m.0", "22.m.0.0"}) ed(std::string(b) + ".cv1.conv", std::string(b) + ".cv2.conv", 0);
        for (const char* l : {"6", "8", "13", "16", "19"}) {          // C3k2 deep: m.0 is a C3k
            const std::string m = std::string(l) + ".m.0";
            for (const char* k : {".m.0", ".m.1"}) ed(m + k + ".cv1.conv", m + k + ".cv2.conv", 0);
            ed(m + ".cv2.conv", m + ".cv3.conv", 1);                  // cat(m chain, cv2)
            ed(m + ".cv3.conv", std::string(l) + ".cv2.conv", 1);     // cat(t, mb)
        }
        for (const char* p : {"10.m.0", "22.m.0.1"}) ed(std::string(p) + ".ffn.0.conv", std::string(p) + ".ffn.1.conv", 0);
        ed("17.conv", "19.cv1.conv", 0);  ed("20.conv", "22.cv1.conv", 0);
        for (int i = 0; i < 3; ++i) {
            const std::string b = "23.one2one_cv2." + std::to_string(i), k = "23.one2one_cv3." + std::to_string(i);
            ed(b + ".0.conv", b + ".1.conv", 0);  ed(b + ".1.conv", b + ".2", 0);
            ed(k + ".0.0.conv", k + ".0.1.conv", 0);  ed(k + ".0.1.conv", k + ".1.0.conv", 0);
            ed(k + ".1.0.conv", k + ".1.1.conv", 0);  ed(k + ".1.1.conv", k + ".2", 0);
        }
#ifdef Y26_YC
        if (attn2_on())                                               // qkv's codes are the attention block's input
            for (const char* p : {"10.m.0", "22.m.0.1"}) ed(std::string(p) + ".attn.qkv.conv", std::string(p) + ".attn.qkv.conv@attn", 0);
        auto sum = [&](const std::string& p, const std::string& q, int tail, const std::string& with) {
            e[p] = {q, tail, true, false, {}, with, ""};
            e[with].flif = p;
        };
        for (const char* l : {"2", "4"})                              // cv2's last source = b + m.0.cv2(..)
            sum(std::string(l) + ".m.0.cv2.conv", std::string(l) + ".cv2.conv", 1, std::string(l) + ".cv1.conv");
        for (const char* l : {"6", "8", "13", "16", "19"}) {
            const std::string m = std::string(l) + ".m.0";
            sum(m + ".m.0.cv2.conv", m + ".m.1.cv1.conv", 0, m + ".cv1.conv");   // a + m.0's branch: m.1.cv1's input
            sum(m + ".m.1.cv2.conv", m + ".cv3.conv", 0, m + ".m.0.cv2.conv");   // that + m.1's branch: cv3's first source
            sh(m + ".cv2.conv", m + ".cv3.conv", 1, false, {});      // cv3's last source
            sh(m + ".cv3.conv", std::string(l) + ".cv2.conv", 1, false, {});     // cv2's last source
        }
        sh("17.conv", "19.cv1.conv", 0, false, {});  sh("13.cv2.conv", "19.cv1.conv", 1, true, {});   // x13: 16.cv1 upsamples its floats
        sh("20.conv", "22.cv1.conv", 0, false, {});  sh("10.cv2.conv", "22.cv1.conv", 1, true, {});   // x10: 13.cv1 upsamples its floats
        e["13.cv2.conv"].up = "16.cv1.conv";  e["10.cv2.conv"].up = "13.cv1.conv";
        sh("8.cv2.conv", "9.cv1.conv", 0, true, {});                  // SPPF's residual add reads x8's floats
        sh("22.cv1.conv", "22.cv2.conv", 0, true, {"22.m.0.0.cv1.conv"});      // the bottleneck's add reads b's floats
        sh("22.cv2.conv", "23.one2one_cv2.2.0.conv", 0, false, {"23.one2one_cv3.2.0.0.conv"});
#endif
        return e;
    }();
    return E;
}
// Readers that are not convs, as the quantizers of their input channels: the attention block ("<qkv>@attn").
std::unordered_map<std::string, std::unique_ptr<ConvW>> g_pseudo;
const ConvW& yq_conv(const std::string& n) {
    const auto it = g_W->conv.find(n);
    if (it != g_W->conv.end()) return it->second;
    const auto ip = g_pseudo.find(n);
    if (ip == g_pseudo.end()) throw std::runtime_error("YQ edge table: no conv " + n);
    return *ip->second;
}
// c's edge when its operands fit the kernel: the entry, the consumer q and c's first plane in q's input.
const YqEdge* yq_entry(const ConvW& c, const ConvW*& qp, int& off) {
    if (!g_W) return nullptr;
    const auto it = yq_edges().find(c.name);
    if (it == yq_edges().end() || it->second.q.empty()) return nullptr;
    const YqEdge& e = it->second;
    const ConvW& q = yq_conv(e.q);
    if (!(q.quant && q.asym)) return nullptr;                       // consumer runs on the CPU in float
    off = e.tail ? q.ic - c.oc : 0;
    if (off < 0 || off + c.oc > q.ic) throw std::runtime_error("YQ edge " + c.name + " -> " + q.name + ": channels");
#ifdef Y26_YC
    if (!e.shared && q.ic != c.oc) return nullptr;                  // X has other sources too: the host packs it
    for (int o = 0; o < c.oc; ++o) {                                // y26_qc's operand ranges (conv_engine.h)
        const float k = q.ssc[off + o] / q.step_of(off + o);
        const float b = e.with.empty() ? q.lo_of(off + o) / q.step_of(off + o) : add_bo(q, off + o);
        if (!(k >= 1.f && k < 65536.f && b >= -256.f && b <= 0.f)) return nullptr;
    }
#else
    if (e.shared) return nullptr;
#endif
    qp = &q;
    return &e;
}
// p's output is held as codes: its edge holds and, for a sum on codes, so does the sum's other input.
bool yq_coded(const std::string& p) {
    const ConvW* q = nullptr;
    int off = 0;
    const YqEdge* e = yq_entry(yq_conv(p), q, off);
    return e && (e->with.empty() || yq_coded(e->with));
}
#ifdef Y26_YC
// May the PL write planes [ch0, ch0 + n) of q's block? Planes of 20x20 maps are not whole cache lines, so a run of
// them is the PL's only when it is whole lines, or when the ARM writes nothing in that block: every input channel
// of q arrives as codes.
bool pl_run(const ConvW& q, int ch0, int n, size_t HW) {
    if (((size_t)ch0 * HW) % 64 == 0 && ((size_t)n * HW) % 64 == 0) return true;
    static std::map<const ConvW*, bool> full;
    static std::mutex m;
    std::lock_guard<std::mutex> g(m);
    const auto it = full.find(&q);
    if (it != full.end()) return it->second;
    int cov = 0;
    for (const auto& e : yq_edges())
        if (e.second.q == q.name && yq_coded(e.first)) cov += yq_conv(e.first).oc;
    return full[&q] = cov == q.ic;
}
#endif
// fl: the float output is kept too. add: the edge feeds a sum on codes.
const ConvW* yq_edge(const ConvW& c, int& off, bool* fl = nullptr, bool* add = nullptr) {
    const ConvW* q = nullptr;
    const YqEdge* e = yq_entry(c, q, off);
    if (!e || !yq_coded(c.name)) return nullptr;
    bool f = e->fl && !(!e->flif.empty() && yq_coded(e->flif));
#ifdef Y26_YC
    if (f && !e->up.empty() && ops_on()) {                          // the ops block upsamples the codes instead
        const ConvW& u = yq_conv(e->up);
        f = !(u.quant && u.asym) || ops_par(u, 0, *q, off, c.oc, false) == SIZE_MAX;
    }
#endif
    for (const std::string& n : e->same) {                          // a reader on another grid reads the floats
        const ConvW& r = yq_conv(n);
        f |= !(r.quant && r.asym && grid_eq(*q, off + c.oc - r.ic, r, 0, r.ic));
    }
    if (fl) *fl = f;
    if (add) *add = !e->with.empty();
    return q;
}
// The consumer side: true when t arrived as codes for (c, channel ch0) - claimed, so it is used once.
bool y26_take_codes(const Tensor* t, const ConvW& c, int ch0, std::shared_ptr<Region>* xr = nullptr) {
    std::lock_guard<std::mutex> g(g_cmu);
    const auto it = g_codes.find(t->d.data());
    if (it == g_codes.end()) return false;
    if (it->second.cons != &c || it->second.off != ch0 || (it->second.xr && !xr))
        throw std::runtime_error("YQ: a coded tensor for " + it->second.cons->name + " reached " + c.name);
    if (xr) *xr = std::move(it->second.xr);
    g_codes.erase(it);
    return true;
}
#ifdef Y26_YC
// The consumer side of direct codes: the block holding t's codes when they are c's input channels
// [ch0, ch0 + t->C) as c quantizes them; plane = the block's plane with t's first channel.
std::shared_ptr<Region> y26_codes(const Tensor* t, const ConvW& c, int ch0, int& plane, int& by) {
    std::lock_guard<std::mutex> g(g_cmu);
    const auto it = g_codes.find(t->d.data());
    if (it == g_codes.end()) return nullptr;
    const Codes& k = it->second;
    if (!k.add && grid_eq(*k.cons, k.off, c, ch0, t->C)) { plane = k.off; by = k.by; return k.xr; }
    if (!k.fl) throw std::runtime_error("YC: a tensor coded for " + k.cons->name + " with no float copy reached " + c.name);
    return nullptr;
}
// t's codes already are c's input planes from ch0 on, in c's own block.
bool y26_in_place(const Tensor* t, const ConvW& c, int ch0, const std::shared_ptr<Region>& blk, int& by) {
    std::lock_guard<std::mutex> g(g_cmu);
    const auto it = g_codes.find(t->d.data());
    if (it == g_codes.end()) return false;
    const Codes& k = it->second;
    if (k.add || k.cons != &c || k.off != ch0 || k.xr != blk) return false;
    by = k.by;
    return true;
}
// The sum a + b on codes. b: the branch, coded by the kernel with add_bo into c's planes from ch0 on (the sum
// replaces it there); a: codes of any conv's grid.
// bya, byb: who wrote a's and the branch's codes (Codes::by). par: the ops block's words when it makes the sum.
struct IAdd { const uint8_t* qa; const ConvW* ca; int offa, ch0, C, bya, byb; size_t par; };
bool y26_add_codes(const Tensor* a, const Tensor* b, const ConvW& c, int ch0, const std::shared_ptr<Region>& blk, size_t HW, IAdd& r);
// t's codes, for the ops block to upsample 2x into c's input channels from ch0 on: where they are, the block's words
// and its source layout. False: the upsample is made from t's floats.
bool y26_up_codes(const Tensor* t, const ConvW& c, int ch0, size_t& sa, size_t& par, int& amode) {
    if (!ops_on() || (t->W != 20 && t->W != 40) || ((size_t)t->H * t->W) % Y26_OPS_UNIT) return false;
    std::lock_guard<std::mutex> g(g_cmu);
    const auto it = g_codes.find(t->d.data());
    if (it == g_codes.end() || it->second.add) return false;
    const Codes& k = it->second;
    if ((par = ops_par(c, ch0, *k.cons, k.off, t->C, false)) == SIZE_MAX) return false;
    sa = k.xr->off + (size_t)k.off * t->H * t->W;
    amode = t->W == 20 ? 2 : 4;
    return true;
}
// c is about to read t's floats (a fused source).
void y26_floats(const Tensor* t, const ConvW& c) {
    std::lock_guard<std::mutex> g(g_cmu);
    const auto it = g_codes.find(t->d.data());
    if (it != g_codes.end() && !it->second.fl)
        throw std::runtime_error("YC: a tensor coded for " + it->second.cons->name + " with no float copy is a fused source of " + c.name);
}
#endif

const WSlot& weights_slot(Buf& B, const ConvW& c) {
    std::lock_guard<std::mutex> g(g_wmu);
    auto it = g_wslots.find(&c);
    if (it != g_wslots.end()) return it->second;
    const int icpg = c.ic / c.groups, ktap = c.kh * c.kw, wstr = Y26_WSTRIDE(icpg, ktap);
    const size_t tot = al((size_t)c.oc * wstr) + 2 * al((size_t)c.oc * 4) + 2 * al((size_t)c.ic * 4) + 3 * al((size_t)c.oc * 4);
    WSlot s;
    s.wt   = arena().take(tot, true);
    if (s.wt == SIZE_MAX) throw std::runtime_error(c.name + ": no room for weights - enlarge the DMA buffer");
    s.wsc  = s.wt   + al((size_t)c.oc * wstr);
    s.bias = s.wsc  + al((size_t)c.oc * 4);
    s.stp  = s.bias + al((size_t)c.oc * 4);
    s.lo   = s.stp  + al((size_t)c.ic * 4);
    s.qs   = s.lo   + al((size_t)c.ic * 4);
    s.qlo  = s.qs   + al((size_t)c.oc * 4);
    s.qst  = s.qlo  + al((size_t)c.oc * 4);
    const size_t end = s.qst + al((size_t)c.oc * 4);
    std::memset(B.host + s.wt, 0, end - s.wt);       // tap-stride pads, absent bias, non-perch step/lo: zero, as the tb
    int8_t* Wt = reinterpret_cast<int8_t*>(B.host + s.wt);   // tb: pW permutation
    for (int o = 0; o < c.oc; ++o)
        for (int il = 0; il < icpg; ++il)
            for (int t = 0; t < ktap; ++t)
                Wt[(size_t)o * wstr + Y26_WIDX(icpg, ktap, il, t)] = (int8_t)(int)c.w[((size_t)o * icpg + il) * ktap + t];
    std::memcpy(B.host + s.wsc, c.wsc.data(), c.wsc.size() * 4);
    if (!c.b.empty()) std::memcpy(B.host + s.bias, c.b.data(), c.b.size() * 4);
    if (c.perch) { std::memcpy(B.host + s.stp, c.sa_v.data(), c.sa_v.size() * 4);
                   std::memcpy(B.host + s.lo,  c.lo_v.data(), c.lo_v.size() * 4); }
    int qoff = 0;
    bool qadd = false;
    if (const ConvW* q = yq_edge(c, qoff, nullptr, &qadd)) {        // the consumer's quantizer for these channels
        float* qs = reinterpret_cast<float*>(B.host + s.qs);
        float* ql = reinterpret_cast<float*>(B.host + s.qlo);
        float* qt = reinterpret_cast<float*>(B.host + s.qst);
#ifdef Y26_YC
        (void)qt;                                   // y26_qc: k = ssc/step, b = lo/step
        for (int o = 0; o < c.oc; ++o) {
            qs[o] = q->ssc[qoff + o] / q->step_of(qoff + o);
            ql[o] = qadd ? add_bo(*q, qoff + o) : q->lo_of(qoff + o) / q->step_of(qoff + o);
        }
#else
        (void)qadd;
        for (int o = 0; o < c.oc; ++o) { qs[o] = q->ssc[qoff + o]; ql[o] = q->lo_of(qoff + o); qt[o] = 1.f / q->step_of(qoff + o); }
#endif
    }
    B.to_dev(s.wt, end - s.wt);
    return g_wslots.emplace(&c, s).first->second;
}
#ifdef Y26_YC
// Where the kernel's codes go when no conv reads them: one permanent block, never synced, never read.
size_t code_scrap(size_t n) {
    static const size_t cap = (size_t)4 << 20;     // 640x640 frames: the largest conv writes 1.6 MB of codes
    static const size_t off = [] {
        const size_t o = arena().take(cap, true);
        if (o == SIZE_MAX) throw std::runtime_error("no room for the code scrap block - enlarge the DMA buffer");
        return o;
    }();
    if (n > cap) throw std::runtime_error("a conv writes more codes than the scrap block holds");
    return off;
}
#endif
}  // namespace

// Fused sources (Src in layer_ops.h) straight to this conv's codes; each equals quant_plane() of the
// materialized op, byte for byte (Y26_FUSE_CHECK=1 compares them).
static void quant_sum(uint8_t* d, const float* a, const float* b, size_t n, float s, float lo, float st) {
    float t[4096];                                 // the residual sum, one cache-resident block at a time
    for (size_t i = 0; i < n; i += 4096) {
        const size_t m = std::min<size_t>(4096, n - i);
        for (size_t j = 0; j < m; ++j) t[j] = a[i + j] + b[i + j];
        quant_plane(d + i, t, m, s, lo, st);       // exact per element, so the block split cannot matter
    }
}
#ifdef Y26_YC
// d = clamp(round(A*a + Bc) + d): a's codes rescaled to the reader's grid, plus the branch's.
__attribute__((optimize("tree-vectorize", "vect-cost-model=dynamic")))
static void add_plane(uint8_t* __restrict d, const uint8_t* __restrict a, size_t n, float A, float Bc) {
    for (size_t i = 0; i < n; ++i) {
        float r = std::nearbyint((float)a[i] * A + Bc) + (float)d[i];
        r = r > 255.f ? 255.f : (r < 0.f ? 0.f : r);
        d[i] = (uint8_t)(int)r;
    }
}
// Y26_YC_CHECK: the sum's codes d against the code of the float sum fa + fb.
static void add_chk(const uint8_t* d, const ConvW& c, int ic, size_t HW, const float* fa, const float* fb) {
    long n1 = 0, n3 = 0;
    for (size_t i = 0; i < HW; ++i) {
        const int e = std::abs((int)d[i] - (int)q_u8(fa[i] + fb[i], c.ssc[ic], c.lo_of(ic), c.step_of(ic)));
        n1 += e > 1;  n3 += e > 3;
    }
    std::lock_guard<std::mutex> g(g_cmu);
    g_ia_n += (long)HW;  g_ia_1 += n1;  g_ia_3 += n3;
}
// Plane k of the sum r, into c's input plane ic. With Y26_YC_CHECK, fa/fb are the two float planes.
static void add_ic(uint8_t* d, const IAdd& r, int k, const ConvW& c, int ic, size_t HW, const float* fa, const float* fb) {
    const int ja = r.offa + k;
    const float kk = c.ssc[ic] / c.step_of(ic), ua = kk / r.ca->ssc[ja];
    add_plane(d, r.qa + (size_t)k * HW, HW, r.ca->step_of(ja) * ua, r.ca->lo_of(ja) * ua + add_bo(c, ic) - c.lo_of(ic) / c.step_of(ic));
    if (fa) add_chk(d, c, ic, HW, fa, fb);
}
namespace {
bool y26_add_codes(const Tensor* a, const Tensor* b, const ConvW& c, int ch0, const std::shared_ptr<Region>& blk, size_t HW, IAdd& r) {
    std::lock_guard<std::mutex> g(g_cmu);
    const auto ib = g_codes.find(b->d.data());
    if (ib == g_codes.end() || !ib->second.add) return false;
    const Codes& kb = ib->second;
    const auto ia = g_codes.find(a->d.data());
    if (kb.cons != &c || kb.off != ch0 || kb.xr != blk || ia == g_codes.end() || ia->second.add)
        throw std::runtime_error("YC: the branch coded for a sum in " + kb.cons->name + " met another sum (" + c.name + ")");
    const Codes& ka = ia->second;
    r = {buf().host + ka.xr->off + (size_t)ka.off * HW, ka.cons, ka.off, ch0, a->C, ka.by, kb.by, SIZE_MAX};
    if (ops_on() && HW % Y26_OPS_UNIT == 0 && kb.by != 1 && pl_run(c, ch0, a->C, HW)) r.par = ops_par(c, ch0, *ka.cons, ka.off, a->C, true);
    return true;
}
}  // namespace
#endif
static void quant_up2x(uint8_t* d, const float* a, int h, int w, float s, float lo, float st) {
    thread_local std::vector<uint8_t> q;
    q.resize((size_t)h * w);
    quant_plane(q.data(), a, q.size(), s, lo, st); // a quarter of the elements the upsampled map would have
    const size_t W2 = 2 * (size_t)w;
    for (int y = 0; y < h; ++y) {
        uint8_t* r0 = d + 2 * (size_t)y * W2;
        for (int x = 0; x < w; ++x) r0[2 * x] = r0[2 * x + 1] = q[(size_t)y * w + x];
        std::memcpy(r0 + W2, r0, W2);
    }
}
// maxpool(5,1,2) as max over rows then columns. Max commutes with a monotone q, so pooling the codes == coding
// the pool. Codes are >= 0, so zero padding gives the clipped window's max and every loop is branch-free.
__attribute__((optimize("tree-vectorize", "vect-cost-model=dynamic")))
static void pool_pass(uint8_t* __restrict d, uint8_t* __restrict r, uint8_t* __restrict p, int H, int W) {
    for (int y = 0; y < H; ++y) {                  // p = 0 0 row 0 0
        std::memcpy(p + 2, d + (size_t)y * W, W);
        uint8_t* ry = r + (size_t)y * W;
        for (int x = 0; x < W; ++x)
            ry[x] = std::max(std::max(std::max(p[x], p[x + 1]), std::max(p[x + 2], p[x + 3])), p[x + 4]);
    }
    for (int y = 0; y < H; ++y) {
        uint8_t* dy = d + (size_t)y * W;
        const int y0 = std::max(0, y - 2), y1 = std::min(H - 1, y + 2);
        std::memcpy(dy, r + (size_t)y0 * W, W);
        for (int yy = y0 + 1; yy <= y1; ++yy) {
            const uint8_t* ry = r + (size_t)yy * W;
            for (int x = 0; x < W; ++x) dy[x] = std::max(dy[x], ry[x]);
        }
    }
}
static void quant_pool(uint8_t* d, const float* a, int H, int W, int k, float s, float lo, float st) {
    CSIM_HOST_ASSERT(s > 0 && st > 0, "fused maxpool needs a monotone quantizer (ssc > 0, step > 0)");
    const size_t n = (size_t)H * W;
    quant_plane(d, a, n, s, lo, st);
    thread_local std::vector<uint8_t> r, p;
    r.resize(n);
    p.assign((size_t)W + 4, 0);
    for (int it = 0; it < k; ++it) pool_pass(d, r.data(), p.data(), H, W);
}

// OpenMP only for big inputs: the quantizer is vectorized, so a parallel region costs more than a small conv's
// whole pack. Y26_PACK_MT=<elements> sets the threshold (tune on the board).
static size_t pack_mt() {
    static const size_t v = [] {
        const char* e = std::getenv("Y26_PACK_MT");
        const size_t v = e && *e ? std::strtoull(e, nullptr, 0) : ((size_t)1 << 20);
#ifdef _OPENMP
        std::printf("[board] OpenMP %d threads; pack X threads when X >= %zu elements\n", omp_get_max_threads(), v);
#else
        std::printf("[board] OpenMP OFF (built without -fopenmp): pack X and attention run on ONE core\n");
#endif
        return v;
    }();
    return v;
}

// X packed ahead, X shared by siblings, uint8 input.
// Same quantizer on every input channel, byte for byte.
static bool same_quant(const ConvW& a, const ConvW& b) {
    if (a.ic != b.ic || a.ssc.size() != b.ssc.size() || std::memcmp(a.ssc.data(), b.ssc.data(), a.ssc.size() * 4)) return false;
    for (int i = 0; i < a.ic; ++i) {
        const float l0 = a.lo_of(i), l1 = b.lo_of(i), s0 = a.step_of(i), s1 = b.step_of(i);
        if (std::memcmp(&l0, &l1, 4) || std::memcmp(&s0, &s1, 4)) return false;
    }
    return true;
}
// One PLAIN source packed ahead into its conv's X (y26_board_prepack), on its own thread when async.
// The conv waits on `done` before its run.
struct Y26Pre {
    const ConvW* c; const float* d; int C, ch0; size_t HW;
    std::shared_ptr<Region> xr;
    bool kc = false;                               // these planes already hold the codes (direct codes)
    int by = 0;                                    // and who wrote them (Codes::by)
    std::future<void> done;
};
Pre y26_board_prepack(const Tensor* t, const ConvW& c, bool last) {
    if (!(c.quant && c.asym)) return nullptr;      // a CPU conv
    Y26_PROF("prepack X (issue)");
    auto p = std::make_shared<Y26Pre>();
    p->c = &c;  p->d = t->d.data();  p->C = t->C;  p->HW = (size_t)t->H * t->W;
    p->ch0 = last ? c.ic - t->C : 0;
    CSIM_HOST_ASSERT(p->ch0 >= 0 && p->ch0 + t->C <= c.ic, "prepack " + c.name + ": source wider than the conv");
#ifdef Y26_YC
    p->xr = xblock(c, p->HW);
    if (y26_in_place(t, c, p->ch0, p->xr, p->by)) {   // t's codes are in c's X block already: nothing to pack
        p->kc = true;
        p->done = std::async(std::launch::deferred, [] {});
        return p;
    }
    y26_floats(t, c);
    const bool coded = false;
#else
    p->xr.reset(new Region(al((size_t)c.ic * p->HW), c.name));
    const bool coded = y26_take_codes(t, c, p->ch0);   // the source arrived as this conv's codes (yq)
#endif
    const bool bg = y26_async();
    Y26Pre* q = p.get();                           // p owns q and waits on done before it dies
    q->done = std::async(bg ? std::launch::async : std::launch::deferred, [q, coded, bg] {
        Y26_PROF("prepack X (planes)");
        uint8_t* X = buf().host + q->xr->off + (size_t)q->ch0 * q->HW;
        if (coded) { std::memcpy(X, q->d, (size_t)q->C * q->HW); return; }
        const ConvW& c = *q->c;
        // A background thread packs serially: an OpenMP team per short-lived thread costs more than it saves.
#pragma omp parallel for schedule(static) if (!bg && (size_t)q->C * q->HW >= pack_mt())
        for (int k = 0; k < q->C; ++k) {
            const int ic = q->ch0 + k;
            quant_plane(X + (size_t)k * q->HW, q->d + (size_t)k * q->HW, q->HW, c.ssc[ic], c.lo_of(ic), c.step_of(ic));
        }
    });
    return p;
}
// Sibling convs reading the same tensor through the same quantizer: the first to arrive packs X, the other
// reuses it. Checked at run time (same_quant, same source).
struct SharedX {
    const ConvW* c; const float* d;
    std::shared_ptr<Region> xr;
    std::promise<void> pr;
    std::shared_future<void> ready;                // set once X is packed and cleaned
};
std::map<std::pair<int, const float*>, std::shared_ptr<SharedX>> g_sx;   // (pair, source): frames in flight never meet
std::mutex g_sxm;
static int sibling_pair(const std::string& n) {
    static const std::unordered_map<std::string, int> P = [] {
        std::unordered_map<std::string, int> m;
        int i = 0;
        for (const char* l : {"6", "8", "13", "16", "19"}) {          // C3k: cv1(x), cv2(x)
            m[std::string(l) + ".m.0.cv1.conv"] = i;  m[std::string(l) + ".m.0.cv2.conv"] = i++;
        }
        m["17.conv"] = i;  m["23.one2one_cv2.0.0.conv"] = i++;       // both read x16
        m["20.conv"] = i;  m["23.one2one_cv2.1.0.conv"] = i++;       // both read x19
        return m;
    }();
    const auto it = P.find(n);
    return it == P.end() ? -1 : it->second;
}
// uint8 frames (run_model.cpp, .u8): input storage -> its bytes; 0.conv packs X through a table instead of quantizing.
std::unordered_map<const float*, std::shared_ptr<const std::vector<uint8_t>>> g_u8;
std::mutex g_u8m;
void y26_input_u8(const float* d, std::shared_ptr<const std::vector<uint8_t>> u8) {
    std::lock_guard<std::mutex> g(g_u8m);
    if (u8) g_u8[d] = std::move(u8); else g_u8.erase(d);
}

// Img2Col (Qi et al., JRTIP 2025): a dense kxk conv with ic*kh*kw <= Y26_ICGRP whose input zero point is 0 - only
// 0.conv, the image - runs as a 1x1 conv over the ic*kh*kw shifted planes: 1 MAC step per output pixel, not kh*kw.
// Bit-exact: the integer sums are the same products; a padding tap becomes code 0 (== q_u8(0.f) at lo == 0), and
// the zero-point sum it changes is scaled by wsc*lo == 0. The OIHW weights are already [oc][ic*kh*kw]. Y26_IM2COL=0: off.
const ConvW* im2col_conv(const ConvW& c) {
    static const bool on = [] { const char* e = std::getenv("Y26_IM2COL"); return !(e && std::strcmp(e, "0") == 0); }();
    if (!on || c.groups != 1 || c.perch || c.lo != 0.f || c.kh * c.kw == 1 || c.ic * c.kh * c.kw > Y26_ICGRP) return nullptr;
    static std::mutex m;
    static std::unordered_map<const ConvW*, std::unique_ptr<ConvW>> made;   // stable: weights_slot keys on the pointer
    std::lock_guard<std::mutex> g(m);
    auto& d = made[&c];
    if (!d) {
        d.reset(new ConvW(c));
        d->ic = c.ic * c.kh * c.kw;  d->kh = d->kw = d->sh = d->sw = 1;  d->ph = d->pw = 0;
        d->ssc.clear();
        for (int i = 0; i < d->ic; ++i) d->ssc.push_back(c.ssc[i / (c.kh * c.kw)]);
    }
    return d.get();
}

Tensor y26_board_conv(const std::vector<Src>& xs, const ConvW& c, const Pre& pre) {
    Y26_PROF("kernel conv (pack+PL+unpack)");
    const int H = xs[0].H(), W = xs[0].W();
    struct Plane { const Src* s; int k; int ia; };   // channel k of source s; s == nullptr: its codes are in X; ia >= 0: sum on codes
    std::vector<Plane> src;                        // one plane per input channel: the concat, never built
    std::vector<std::pair<int, const Tensor*>> coded;   // (first channel, source) that arrived as codes
    std::shared_ptr<Region> cx;                    // direct codes: this conv's whole X is a run of planes of another
    int cxp = 0;                                   // conv's block, from plane cxp on
    bool pre_found = false;
#ifdef Y26_YC
    // This conv's own block, when a producer or a prepack made it: X. A source not in it yet is packed into it.
    const std::shared_ptr<Region> blk = xblock(c, (size_t)H * W, true);
    std::vector<IAdd> iadd;
    size_t xkern = 0, xcode = 0, xops = 0;         // X bytes the kernel wrote as they are / the ARM / the ops block made from codes
    struct Up { int ch0, C, amode; size_t sa, par; const Tensor* t; };
    std::vector<Up> ups;                           // sources the ops block upsamples from codes
    bool plw = false;                              // the PL wrote planes of this block that are not whole cache lines
#endif
    for (const Src& t : xs) {
        CSIM_HOST_ASSERT(t.H() == H && t.W() == W, "conv " + c.name + ": concat inputs differ in H,W");
        const int ch0 = (int)src.size();
        const bool p = pre && t.op == Src::PLAIN && t.a->d.data() == pre->d && ch0 == pre->ch0 && t.C() == pre->C;
        pre_found |= p;
        int ia = -1;
        bool up = false;
#ifdef Y26_YC
        const size_t tb = (size_t)t.C() * H * W;
        const bool lines = ((size_t)ch0 * H * W) % 64 == 0 && tb % 64 == 0;
        bool q = false;
        int by = 0;
        IAdd r;
        Up u{ch0, t.C(), 0, 0, 0, t.a};
        auto count = [&](int w) { (w == 1 ? xcode : w == 2 ? xops : xkern) += tb; };
        if (p) { if (pre->kc) { count(pre->by);  plw |= pre->by != 1 && !lines; } }
        else if (blk && t.op == Src::PLAIN) q = y26_in_place(t.a, c, ch0, blk, by);
        else if (blk && t.op == Src::ADD && y26_add_codes(t.a, t.b, c, ch0, blk, (size_t)H * W, r)) {
            ia = (int)iadd.size();
            iadd.push_back(r);
            count(r.par == SIZE_MAX ? 1 : 2);
            plw |= r.par != SIZE_MAX && !lines;
        } else if (!blk && t.op == Src::PLAIN && xs.size() == 1) q = (cx = y26_codes(t.a, c, ch0, cxp, by)) != nullptr;
        else if (t.op == Src::UP2X && (up = y26_up_codes(t.a, c, ch0, u.sa, u.par, u.amode))) { ups.push_back(u);  count(2); }
        if (q) { count(by);  plw |= blk && by != 1 && !lines; }
        if (!p && !q && ia < 0 && !up) { y26_floats(t.a, c);  if (t.b) y26_floats(t.b, c); }
#else
        const bool q = !p && t.op == Src::PLAIN && y26_take_codes(t.a, c, ch0, &cx);
#endif
        if (q) coded.push_back({ch0, t.a});
        for (int k = 0; k < t.C(); ++k) src.push_back({p || q || up ? nullptr : &t, k, ia});
    }
    CSIM_HOST_ASSERT((int)src.size() == c.ic, "conv " + c.name + ": input C mismatch");
    CSIM_HOST_ASSERT(!pre || (pre->c == &c && pre_found), "conv " + c.name + ": prepacked source not among its inputs");
#ifdef Y26_YC
    CSIM_HOST_ASSERT(!pre || pre->xr == blk, "conv " + c.name + ": its prepack is not in its block");
    const Region* xb = blk.get();
#else
    CSIM_HOST_ASSERT(!cx || xs.size() == 1, "conv " + c.name + ": direct codes need a single source");
    const Region* xb = pre ? pre->xr.get() : nullptr;
#endif
    const auto t0 = clk::now();
    Buf& B = buf();
    const int OH = (H + 2 * c.ph - c.kh) / c.sh + 1, OW = (W + 2 * c.pw - c.kw) / c.sw + 1;
    const int YS = Y26_HOST_YS(OW);
    const size_t HW = (size_t)H * W, nx = (size_t)c.ic * HW, ny = (size_t)c.oc * OH * YS * 4;

    const ConvW* i2c = !pre && coded.empty() && xs.size() == 1 && sibling_pair(c.name) < 0 ? im2col_conv(c) : nullptr;
    const WSlot& ws = weights_slot(B, i2c ? *i2c : c);
    const auto t1 = clk::now();

    // Y straight into the output tensor when it lives in the buffer with unpadded rows; else a scratch Y.
    Tensor y(c.oc, OH, OW, NoInit());
    const uint8_t* yp = reinterpret_cast<const uint8_t*>(y.d.data());
    const bool direct = YS == OW && yp >= B.host && yp < B.host + B.size;
    // X: the prepacked region, a sibling's (pack it for both, or reuse it), or this conv's own.
    std::shared_ptr<SharedX> sx;
    bool reuse = false;
    const int pair = !pre && xs.size() == 1 && xs[0].op == Src::PLAIN && coded.empty() ? sibling_pair(c.name) : -1;
    if (pair >= 0) {
        std::lock_guard<std::mutex> g(g_sxm);
        const auto it = g_sx.find({pair, xs[0].a->d.data()});
        if (it == g_sx.end()) {
            sx = std::make_shared<SharedX>();
            sx->c = &c;  sx->d = xs[0].a->d.data();
            sx->xr = std::make_shared<Region>(al(nx), c.name);
            sx->ready = sx->pr.get_future().share();
            g_sx[{pair, xs[0].a->d.data()}] = sx;
        } else {
            reuse = it->second->d == xs[0].a->d.data() && same_quant(*it->second->c, c);
            if (reuse) sx = it->second;
            g_sx.erase(it);                        // a mismatch packs its own X below
        }
    }
    // The packer's promise is kept on every path: an exception before `ready` must not strand its sibling.
    struct Ready {
        SharedX* s; bool set = false;
        void ok() { s->pr.set_value(); set = true; }
        ~Ready() { if (s && !set) s->pr.set_exception(std::make_exception_ptr(std::runtime_error("sibling X pack failed"))); }
    } ready{sx && !reuse ? sx.get() : nullptr};
    const ConvW& kc = i2c ? *i2c : c;              // what the kernel runs
    const int KH = i2c ? OH : H, KW = i2c ? OW : W;
    const size_t nxk = i2c ? (size_t)kc.ic * OH * OW : nx;
    std::vector<uint8_t> q3(i2c ? nx : 0);         // Img2Col: the ic code planes, gathered into X below
    Region xr(xb || sx || cx ? 0 : al(nxk), c.name), yr(direct ? 0 : ny, c.name);
    Layout L;
    L.x = xb ? xb->off : sx ? sx->xr->off : cx ? cx->off + (size_t)cxp * HW : xr.off;
    L.wt = ws.wt;  L.wsc = ws.wsc;  L.bias = ws.bias;  L.stp = ws.stp;  L.lo = ws.lo;
    L.y = direct ? (size_t)(yp - B.host) : yr.off;
    L.end = L.y + ny;
    int qoff = 0;
    bool yfl = false, yadd = false;
    const ConvW* cons = yq_edge(c, qoff, &yfl, &yadd);
#ifdef Y26_YC
    // The kernel's code groups are the consumer's X bytes only when rows are whole groups, and whole cache lines per
    // plane keep the host's cache maintenance of a neighbour off these bytes. Otherwise (20x20 maps) the kernel
    // writes its padded rows into a block of their own and the ARM copies them into the consumer's planes.
    const bool cmp = cons && (OW % Y26_EPI_WIDE || ((size_t)OH * OW) % 64);
    const size_t ncode = (size_t)c.oc * OH * YS;
    std::shared_ptr<Region> cxr;                   // the consumer's X: all its planes, this conv's from plane qoff
    if (cons) cxr = xblock(*cons, (size_t)OH * OW);
    Region cr(cmp ? ncode : 0, c.name);
    L.yc = !cons ? code_scrap(ncode) : cmp ? cr.off : cxr->off + (size_t)qoff * OH * OW;
    L.ynone = cons && !yfl;  L.ycnone = !cons;
    const bool dc = cons != nullptr;               // the codes are kept
    const bool yf = !dc || yfl || yc_check();      // the float output is kept
#else
    const bool yf = true;
#endif
    L.yq = cons ? 1 : 0;  L.qs = ws.qs;  L.qlo = ws.qlo;  L.qst = ws.qst;

    uint8_t* const XK = B.host + L.x;              // tb: pX (uint8 codes); Y26_XPE-packing is LE bytes
    uint8_t* X = i2c ? q3.data() : XK;             // the ic planes as packed; == XK unless Img2Col
    // uint8 frame input: X[ic] = table_ic[k], the table being quant_plane of the very floats the loader made (k/255.f).
    std::shared_ptr<const std::vector<uint8_t>> u8;
    if (xs.size() == 1 && xs[0].op == Src::PLAIN && src[0].s) {
        std::lock_guard<std::mutex> g(g_u8m);
        const auto it = g_u8.find(xs[0].a->d.data());
        if (it != g_u8.end() && it->second->size() == nx) { u8 = it->second; g_u8.erase(it); }
    }
    // SPPF's pooled planes cost ~2 passes x k each on top of the quantize: weigh them into the thread threshold.
    size_t work = nx;
    for (const Src& t : xs) if (t.op == Src::POOL) work += (size_t)t.C() * HW * 2 * t.k;
    const size_t mt = reuse ? SIZE_MAX : pack_mt();
    // Y26_PACK_SCHED=dynamic: hands out 4 planes at a time instead of an even static split (helps when a thread
    // shares its core with another async branch).
    static const bool pack_dyn = [] { const char* e = std::getenv("Y26_PACK_SCHED"); return e && std::strcmp(e, "dynamic") == 0; }();
    auto pack_ic = [&](int ic) {
        const Src* sp = src[ic].s;
        if (!sp || reuse) return;
        const float* a = sp->a->d.data() + (size_t)src[ic].k * sp->a->H * sp->a->W;
        uint8_t* d = X + (size_t)ic * HW;
#ifdef Y26_YC
        if (src[ic].ia >= 0) {
            const IAdd& r = iadd[src[ic].ia];
            const float* fb = sp->b->d.data() + (size_t)src[ic].k * HW;
            if (r.par == SIZE_MAX) add_ic(d, r, src[ic].k, c, ic, HW, yc_check() ? a : nullptr, fb);
            else if (yc_check()) add_chk(d, c, ic, HW, a, fb);
            return;
        }
#endif
        const float s = c.ssc[ic], lo = c.lo_of(ic), st = c.step_of(ic);
        if (u8) {
            uint8_t lut[256];
            for (int k = 0; k < 256; ++k) lut[k] = (uint8_t)(int)q_u8((float)k / 255.f, s, lo, st);
            const uint8_t* u = u8->data() + (size_t)ic * HW;
            for (size_t i = 0; i < HW; ++i) d[i] = lut[u[i]];
            return;
        }
        switch (sp->op) {
        case Src::PLAIN: quant_plane(d, a, HW, s, lo, st); break;
        case Src::ADD:   quant_sum(d, a, sp->b->d.data() + (size_t)src[ic].k * HW, HW, s, lo, st); break;
        case Src::UP2X:  quant_up2x(d, a, sp->a->H, sp->a->W, s, lo, st); break;
        case Src::POOL:  quant_pool(d, a, H, W, sp->k, s, lo, st); break;
        }
    };
#ifdef Y26_YC
    for (const Up& u : ups) {                      // the 2x upsample of codes, onto this conv's grid
        const size_t d = L.x + (size_t)u.ch0 * HW, n = (size_t)u.C * HW;
        B.for_write(d, n);
        { Y26_PROF("ops block (PL)");  ops_run(B, u.sa, u.sa, d, u.par, u.C, (int)(HW / Y26_OPS_UNIT), u.amode, 0); }
        if (!yc_check()) continue;                 // against the codes of the upsampled floats
        B.from_dev(d, n);
        std::vector<uint8_t> q(HW);
        long n1 = 0, n3 = 0;
        for (int k = 0; k < u.C; ++k) {
            const int ic = u.ch0 + k;
            quant_up2x(q.data(), u.t->d.data() + (size_t)k * (HW / 4), H / 2, W / 2, c.ssc[ic], c.lo_of(ic), c.step_of(ic));
            for (size_t i = 0; i < HW; ++i) {
                const int e = std::abs((int)q[i] - (int)B.host[d + (size_t)k * HW + i]);
                n1 += e > 1;  n3 += e > 3;
            }
        }
        std::lock_guard<std::mutex> g(g_cmu);
        g_up_n += (long)n;  g_up_1 += n1;  g_up_3 += n3;
    }
    for (const IAdd& r : iadd) {
        const size_t d = L.x + (size_t)r.ch0 * HW, n = (size_t)r.C * HW;
        if (r.par != SIZE_MAX) {                   // the ops block makes the sum where the branch's codes are
            { Y26_PROF("ops block (PL)");  ops_run(B, (size_t)(r.qa - B.host), d, d, r.par, r.C, (int)(HW / Y26_OPS_UNIT), 0, 1); }
            if (yc_check()) B.from_dev(d, n);
            continue;
        }
        if (r.byb != 1) B.from_dev(d, n);          // what the PL wrote there becomes visible to the ARM
        if (r.bya != 1) B.from_dev((size_t)(r.qa - B.host), n);
    }
#endif
    if (pack_dyn) {
#pragma omp parallel for schedule(dynamic, 4) if (work >= mt)
        for (int ic = 0; ic < c.ic; ++ic) pack_ic(ic);
    } else {
#pragma omp parallel for schedule(static) if (work >= mt)
        for (int ic = 0; ic < c.ic; ++ic) pack_ic(ic);
    }
    if (pre) { Y26_PROF("prepack X (wait)"); pre->done.get(); }
    if (reuse) { Y26_PROF("sibling X (wait)"); sx->ready.get(); }
    static const bool fuse_check = [] { const char* e = std::getenv("Y26_FUSE_CHECK"); return e && *e == '1'; }();
    if (fuse_check) {                              // every fused plane == quant_plane(the materialized op); every
        std::vector<uint8_t> q(HW);                // prepacked / shared / table plane == quant_plane(its tensor)
        int ch0 = 0;
        for (const Src& t : xs) {
            const bool fused = t.op != Src::PLAIN && src[ch0].ia < 0 && src[ch0].s;
            const bool ahead = !fused && ((pre && !pre->kc && t.a->d.data() == pre->d) || reuse || u8);
            if (fused || ahead) {
                const Tensor m = fused ? y26_materialize(t) : Tensor();
                const float* f = fused ? m.d.data() : t.a->d.data();
                for (int k = 0; k < t.C(); ++k) {
                    const int ic = ch0 + k;
                    quant_plane(q.data(), f + (size_t)k * HW, HW, c.ssc[ic], c.lo_of(ic), c.step_of(ic));
                    if (std::memcmp(q.data(), X + (size_t)ic * HW, HW) != 0)
                        throw std::runtime_error("Y26_FUSE_CHECK: " + c.name + " input channel " + std::to_string(ic) + " differs");
                }
                std::lock_guard<std::mutex> g(g_stats.m);
                ++(fused ? g_fuse_checked : g_ahead_checked);
            }
            ch0 += t.C();
        }
    }
#ifndef Y26_YC
    for (const auto& e : coded)                    // already this conv's codes (yq), channel-major
        std::memcpy(X + (size_t)e.first * HW, e.second->d.data(), (size_t)e.second->C * HW);
#endif
    if (i2c) {                                     // XK[(ic*kh+ky)*kw+kx][oy][ox] = X[ic][oy*sh+ky-ph][ox*sw+kx-pw], else 0
        Y26_PROF("im2col gather");
        for (int ic = 0; ic < c.ic; ++ic)
            for (int ky = 0; ky < c.kh; ++ky)
                for (int kx = 0; kx < c.kw; ++kx) {
                    uint8_t* d = XK + (size_t)((ic * c.kh + ky) * c.kw + kx) * OH * OW;
                    const uint8_t* sp = X + (size_t)ic * HW;
                    for (int oy = 0; oy < OH; ++oy, d += OW) {
                        const int iy = oy * c.sh + ky - c.ph;
                        if (iy < 0 || iy >= H) { std::memset(d, 0, OW); continue; }
                        const uint8_t* r = sp + (size_t)iy * W;
                        for (int ox = 0; ox < OW; ++ox) {
                            const int ix = ox * c.sw + kx - c.pw;
                            d[ox] = ix >= 0 && ix < W ? r[ix] : 0;
                        }
                    }
                }
    }
    bool armx = !reuse && !cx;                     // the ARM wrote some of X here: clean it before the kernel reads
#ifdef Y26_YC
    if (blk) {                                     // planes already holding their codes were cleaned by their writer
        armx = pre && !pre->kc;
        for (const Plane& pl : src) armx |= pl.s && !(pl.ia >= 0 && iadd[pl.ia].par != SIZE_MAX);
        if (armx && plw) throw std::runtime_error("conv " + c.name + ": the ARM and the PL both write planes that share cache lines");
    }
#endif
    if (armx) std::memset(XK + nxk, 0, al(nxk) - nxk);   // only the port-word tail needs zeros
    const auto t2 = clk::now();
    // Cached buffer: X (and the weights, at first use) are cleaned before the kernel reads them; Y's range is
    // cleaned before the kernel writes it, so no dirty line can later be evicted over Y. A reused sibling X was
    // cleaned by its packer.
    // Direct codes: the kernel wrote X, so there is nothing to clean; and of this conv's outputs only the codes are
    // kept, so only their block is cleaned.
    if (!B.coh) {                                  // coherent X/Y: the CCI snoops, nothing to maintain
        if (armx) B.to_dev(L.x, al(nxk));          // the kernel-written planes hold no dirty line: untouched
        if (yf) B.for_write(L.y, ny);
    }
#ifdef Y26_YC
    if (dc) B.for_write(L.yc, ncode);
#endif
    if (sx && !reuse) ready.ok();
    const auto t3 = clk::now();

    int check = -1;
#ifndef Y26_BOARD_SIM
    static const char* sel = std::getenv("Y26_BOARD_CHECK");
    std::vector<uint64_t> ref;
    if (sel && *sel && c.name.find(sel) != std::string::npos) {   // inputs + every weight slot, before the run
#ifdef Y26_YC
        if (xkern) B.from_dev(L.x, nxk);           // X came from the kernel: make the host see it
#endif
        ref.assign(B.size / 8, 0);                 // CHECK forces serial (y26_async), so the image is stable
        std::memcpy(ref.data(), B.host, B.size);
    }
#endif
    const double kern = B.run(L, kc, KH, KW);
    const auto t4 = clk::now();
    if (!B.coh && yf) B.from_dev(L.y, ny);
    const auto t4b = clk::now();

    const float* Yf = reinterpret_cast<const float*>(B.host + L.y);
#ifndef Y26_BOARD_SIM
    if (!ref.empty()) {
        uint8_t* rb = reinterpret_cast<uint8_t*>(ref.data());
        run_cmodel(rb, L, kc, KH, KW);
        const float* g = reinterpret_cast<const float*>(rb + L.y);
        check = 0;
        const bool yvoid = !yf && B.hole;          // the float output went nowhere
        if (!yf && !yvoid) B.from_dev(L.y, ny);
        for (int o = 0; o < c.oc && !yvoid; ++o)
            for (int r = 0; r < OH; ++r) {
                const size_t i = ((size_t)o * OH + r) * YS;
                check += std::memcmp(Yf + i, g + i, (size_t)OW * 4) != 0;   // rows that differ
            }
#ifdef Y26_YC
        if (dc) {                                  // and the codes, row by row
            B.from_dev(L.yc, ncode);
            for (size_t row = 0; row < (size_t)c.oc * OH; ++row)
                check += std::memcmp(B.host + L.yc + row * YS, rb + L.yc + row * YS, (size_t)OW) != 0;
        }
#endif
    }
#endif
#ifdef Y26_YC
    // The ops block copies the 20-wide rows out of their padding when it may write the consumer's planes.
    const bool plc = cmp && ops_on() && OW == 20 && YS == 24 && ((size_t)OH * OW) % Y26_OPS_UNIT == 0 && c.oc <= Y26_OPS_CMAX &&
                     pl_run(*cons, qoff, c.oc, (size_t)OH * OW);
    if (cmp && !plc) B.from_dev(L.yc, ncode);
    if (dc && yc_check() && !B.hole) {             // each code against the host quantizer of the kernel's own float
        if (!cmp || plc) B.from_dev(L.yc, ncode);
        std::vector<uint8_t> q(OW);
        long nd = 0, nf = 0;
        for (size_t row = 0; row < (size_t)c.oc * OH; ++row) {
            const int ch = qoff + (int)(row / OH);
            if (yadd) quant_plane(q.data(), Yf + row * YS, OW, cons->ssc[ch] / cons->step_of(ch), add_bo(*cons, ch), 1.f);
            else quant_plane(q.data(), Yf + row * YS, OW, cons->ssc[ch], cons->lo_of(ch), cons->step_of(ch));
            const uint8_t* k = B.host + L.yc + row * YS;
            for (int i = 0; i < OW; ++i) { nd += q[i] != k[i]; nf += q[i] > k[i] + 1 || k[i] > q[i] + 1; }
        }
        std::lock_guard<std::mutex> g(g_cmu);
        g_yc_n += (long)c.oc * OH * OW;  g_yc_diff += nd;  g_yc_far += nf;
    }
    if (plc) {
        const size_t d = cxr->off + (size_t)qoff * OH * OW;
        B.for_write(d, (size_t)c.oc * OH * OW);
        Y26_PROF("ops block (PL)");
        ops_run(B, L.yc, L.yc, d, ops_ident(), c.oc, OH * OW / Y26_OPS_UNIT, 1, 0);
    } else if (cmp) {                              // the padded code rows -> the consumer's planes, cleaned here
        uint8_t* d = B.host + cxr->off + (size_t)qoff * OH * OW;
        for (size_t row = 0; row < (size_t)c.oc * OH; ++row) std::memcpy(d + row * OW, B.host + L.yc + row * YS, (size_t)OW);
        B.to_dev(cxr->off + (size_t)qoff * OH * OW, (size_t)c.oc * OH * OW);
    }
#endif
    const auto t5 = clk::now();
#ifdef Y26_YC
    {
        std::lock_guard<std::mutex> g(g_cmu);
        if (dc) { g_codes[y.d.data()] = {cons, qoff, cxr, yfl, yadd, cmp ? (plc ? 2 : 1) : 0};  ++g_ncoded; }
        g_xall += (long)nx;  g_xkern += (long)xkern;  g_xcode += (long)xcode;  g_xops += (long)xops;
    }
    if (!yf) {                                     // the output tensor only names the block: its floats are not valid
#ifdef Y26_BOARD_SIM
        std::fill(y.d.begin(), y.d.end(), std::nanf(""));   // a float reader of this tensor shows in the detections
#endif
    } else if (!direct)                            // scratch Y: unpack the padded rows
        for (int o = 0; o < c.oc; ++o)
            for (int r = 0; r < OH; ++r)
                std::memcpy(&y.d[((size_t)o * OH + r) * OW], Yf + ((size_t)o * OH + r) * YS, (size_t)OW * 4);
#else
    if (cons) {
        // Code slots -> the consumer's X-plane bytes, in the output tensor's own storage. In place when direct
        // (byte i is written after slot i is read, and every later slot sits at byte >= 4i).
        uint8_t* d = reinterpret_cast<uint8_t*>(y.d.data());
        const uint32_t* s = reinterpret_cast<const uint32_t*>(B.host + L.y);
        for (size_t row = 0; row < (size_t)c.oc * OH; ++row) {
            const int ch = qoff + (int)(row / OH);
            const float qs = cons->ssc[ch], ql = cons->lo_of(ch), qt = cons->step_of(ch);
            y26_yq_plane(d + row * OW, s + row * YS, OW, qs, ql, qt);
        }
        std::lock_guard<std::mutex> g(g_cmu);
        g_codes[y.d.data()] = {cons, qoff, nullptr};
        ++g_ncoded;
    } else if (!direct)                            // scratch Y: unpack the padded rows
    for (int o = 0; o < c.oc; ++o)
        for (int r = 0; r < OH; ++r)
            std::memcpy(&y.d[((size_t)o * OH + r) * OW], Yf + ((size_t)o * OH + r) * YS, (size_t)OW * 4);
#endif
    const auto t6 = clk::now();
    // out = invalidate/copy-back + unpack. Do not time CHECK runs (the C-model is inside the conv total).
    std::lock_guard<std::mutex> g(g_stats.m);
    if (g_stats.rows.empty() || t0 < g_stats.first) g_stats.first = t0;
    if (g_stats.rows.empty() || t6 > g_stats.last) g_stats.last = t6;
    g_stats.rows.push_back({c.name, ms(t1, t2), ms(t0, t1), ms(t2, t3), kern, ms(t4, t4b) + ms(t5, t6), ms(t0, t6), check});
    return y;
}

// Tensor storage (tensor_types.h NoInitAlloc): from the DMA buffer, so conv outputs need no unpack copy.
void* y26_tensor_alloc(size_t bytes) {
    const size_t off = arena().take(bytes, false);
    if (off != SIZE_MAX) return buf().host + off;
    ++arena().spilled;                             // full: plain heap; a conv writing into it takes the copy path
    return ::operator new(bytes);
}
#ifdef Y26_ATTN_INT
// X: per [head][n] one 128-byte record {q[32], k[32], v[64]}; Y: J as int32, [head][n][64] (attention_engine.cpp).
void y26_board_attn(const int8_t* q, const int8_t* k, const int8_t* v, int32_t* J, int NH, int N, int KD, int VD, AttnInt c) {
    Y26_PROF("attention block (pack+unpack)");
    if (KD != Y26_ATTN_KD || VD != Y26_ATTN_VD || NH > Y26_ATTN_NH || N > Y26_ATTN_NMAX)
        throw std::runtime_error("attention: not a shape the PL block takes");
    Buf& B = buf();
    const size_t P = (size_t)NH * N, nx = P * 128, ny = P * VD * 4;
    Region xr(nx, "attention X"), yr(ny, "attention Y");
    uint8_t* X = B.host + xr.off;
    for (size_t i = 0; i < P; ++i) {
        std::memcpy(X + i * 128, q + i * KD, KD);
        std::memcpy(X + i * 128 + 32, k + i * KD, KD);
        std::memcpy(X + i * 128 + 64, v + i * VD, VD);
    }
    B.to_dev(xr.off, nx);
    B.for_write(yr.off, ny);
    { Y26_PROF("attention block (PL)"); B.attn(xr.off, yr.off, NH, N, c); }
    B.from_dev(yr.off, ny);
    std::memcpy(J, B.host + yr.off, ny);

    static const char* sel = std::getenv("Y26_BOARD_CHECK");
    if (sel && *sel) {
        std::vector<int32_t> ref(P * VD);
        for (int h = 0; h < NH; ++h)
            y26_attn_int_head(q + (size_t)h * N * KD, k + (size_t)h * N * KD, v + (size_t)h * N * VD, &ref[(size_t)h * N * VD],
                              N, KD, VD, c);
        long bad = 0;
        for (size_t i = 0; i < ref.size(); ++i) bad += ref[i] != J[i];
        std::printf("[board] CHECK attention: %zu J vs C-model, %ld differ -> %s\n", ref.size(), bad, bad ? "MISMATCH" : "BIT-EXACT");
    }
}
#endif

bool y26_board_attn2(const Tensor& x, const ConvW& qkv, const ConvW& proj, const ConvW& pe, const AttnFQ& fq, Tensor& out) {
#if defined(Y26_ATTN_INT) && defined(Y26_YC)
    if (!g_W || !attn2_on()) return false;
    const int C = x.C, N = x.H * x.W, NH = std::max(C / 64, 1), VD = C / NH, KD = VD / 2;
    auto centred = [](const FQ& f) { return f.on && std::fabs(f.lo + 128.f * f.step) < 1e-3f * f.step; };
    if (KD != Y26_ATTN_KD || VD != Y26_ATTN_VD || NH > Y26_ATTN_NH || N > Y26_ATTN_NMAX || N % 16 || qkv.oc != NH * (2 * KD + VD) ||
        NH * VD != C || !(centred(fq.q) && centred(fq.k) && centred(fq.v_mm) && fq.sm.on && fq.sm.lo == 0.f) || !(proj.quant && proj.asym) ||
        proj.ic != C || pe.quant || pe.groups != C || pe.kh != 3 || pe.kw != 3 || pe.sh != 1 || pe.sw != 1 || pe.ph != 1 || pe.pw != 1 || pe.act != 0)
        throw std::runtime_error("attention v2: " + qkv.name + " is not a module the block takes");
    int off = 0;
    const ConvW* a = yq_edge(qkv, off);
    if (!a || a->name != qkv.name + "@attn") throw std::runtime_error("attention v2: " + qkv.name + "'s codes do not fit the kernel's quantizer");
    Buf& B = buf();
    // The block's words, made once: pe's taps and bias and the J scale in proj's code units.
    static std::map<const ConvW*, std::pair<size_t, std::vector<AttnPar>>> made;
    static std::mutex mm;
    size_t par;
    const std::vector<AttnPar>* pp;
    {
        std::lock_guard<std::mutex> g(mm);
        const auto it = made.find(&proj);
        if (it != made.end()) { par = it->second.first;  pp = &it->second.second; }
        else {
            std::vector<AttnPar> p(C);
            const double osc = (double)fq.sm.step * fq.v_mm.step;
            for (int c = 0; c < C; ++c) {
                const double g = (double)proj.ssc[c] / proj.step_of(c);
                const double m1 = osc * g * std::ldexp(1.0, Y26_ATTN_OF);
                const double b = ((pe.b.empty() ? 0. : (double)pe.b[c]) * g - (double)proj.lo_of(c) / proj.step_of(c)) * std::ldexp(1.0, Y26_ATTN_OF);
                double wmax = 0;
                for (int t = 0; t < 9; ++t) {
                    const double w = (double)pe.w[(size_t)c * 9 + t] * (pe.has_bn ? pe.s[c] : 1.f) * fq.v_mm.step * g * (1 << Y26_ATTN_WF);
                    wmax = std::max(wmax, std::fabs(w));
                    p[c].w[t] = (int32_t)std::llround(std::max(-2147483000., std::min(2147483000., w)));
                }
                if (!(m1 >= 0 && m1 < (1 << 26) - 1 && std::fabs(b) < std::ldexp(1.0, 39) - 1 && wmax < (1 << 17) - 1))
                    throw std::runtime_error("attention v2: " + proj.name + " channel " + std::to_string(c) + " does not fit the block: m1 " +
                                             std::to_string(m1) + ", largest tap " + std::to_string(wmax) + ", B " + std::to_string(b));
                p[c].m1 = (uint32_t)std::llround(m1);  p[c].b = std::llround(b);
            }
            std::vector<uint32_t> pk((size_t)NH * Y26_ATTN2_PW * 4);
            y26_attn2_pack(p.data(), NH, VD, pk.data());
            const size_t n = al(pk.size() * 4);
            if ((par = arena().take(n, true)) == SIZE_MAX) throw std::runtime_error("no room for the attention block's words - enlarge the DMA buffer");
            std::memset(B.host + par, 0, n);
            std::memcpy(B.host + par, pk.data(), pk.size() * 4);
            B.to_dev(par, n);
            pp = &(made[&proj] = std::make_pair(par, std::move(p))).second;
        }
    }
    const Tensor q = y26_board_conv({&x}, qkv);    // its codes: the block's input planes
    std::shared_ptr<Region> xr;
    {
        std::lock_guard<std::mutex> g(g_cmu);
        const auto it = g_codes.find(q.d.data());
        if (it == g_codes.end() || it->second.cons != a || it->second.add) throw std::runtime_error("attention v2: " + qkv.name + " left no codes");
        xr = it->second.xr;
    }
    xblock(*a, (size_t)N, true);                   // nothing else reads that block
    const std::shared_ptr<Region> dr = xblock(proj, (size_t)N);
    const AttnInt ai = {(uint32_t)std::llround(std::ldexp((double)fq.q.step * fq.k.step / std::log(2.0), 28)),
                        (uint32_t)std::llround(65536.0 / fq.sm.step)};
    const size_t nd = (size_t)C * N;
    B.for_write(dr->off, nd);
    { Y26_PROF("attention block (PL)");  B.attn2(xr->off, dr->off, par, NH, N, x.W, ai); }
    static const char* sel = std::getenv("Y26_BOARD_CHECK");
    if (sel && *sel) {
        B.from_dev(xr->off, (size_t)qkv.oc * N);
        B.from_dev(dr->off, nd);
        std::vector<uint8_t> ref(nd);
        y26_attn2_ref(B.host + xr->off, ref.data(), pp->data(), NH, N, x.W, KD, VD, ai);
        long bad = 0;
        for (size_t i = 0; i < nd; ++i) bad += ref[i] != B.host[dr->off + i];
        std::printf("[board] CHECK attention v2 %s: %zu codes vs C-model, %ld differ -> %s\n", qkv.name.c_str(), nd, bad, bad ? "MISMATCH" : "BIT-EXACT");
    }
    Tensor o(C, x.H, x.W, NoInit());               // names proj's input planes; it has no floats
#ifdef Y26_BOARD_SIM
    std::fill(o.d.begin(), o.d.end(), std::nanf(""));
#endif
    {
        std::lock_guard<std::mutex> g(g_cmu);
        g_codes[o.d.data()] = {&proj, 0, dr, false, false, 2};
    }
    out = y26_board_conv({&o}, proj);
    return true;
#else
    (void)x; (void)qkv; (void)proj; (void)pe; (void)fq; (void)out;
    return false;
#endif
}

// A channel slice of a coded tensor is coded: the same block, c0 planes on. True when x has no floats to copy.
bool y26_board_slice(const Tensor& x, Tensor& y, int c0) {
#ifdef Y26_YC
    std::lock_guard<std::mutex> g(g_cmu);
    const auto it = g_codes.find(x.d.data());
    if (it == g_codes.end()) return false;
    const Codes k = it->second;                    // a copy: the insert below may rehash
    g_codes[y.d.data()] = {k.cons, k.off + c0, k.xr, k.fl, k.add, k.by};
    if (k.fl || yc_check()) return false;
#ifdef Y26_BOARD_SIM
    std::fill(y.d.begin(), y.d.end(), std::nanf(""));
#endif
    return true;
#else
    (void)x; (void)y; (void)c0;
    return false;
#endif
}
// A residual add whose branch the kernel coded for it: the sum is made on the codes, in the reader's block.
bool y26_board_add(const Tensor& a, const Tensor& b, Tensor& y) {
#ifdef Y26_YC
    Codes kb;
    {
        std::lock_guard<std::mutex> g(g_cmu);
        const auto ia = g_codes.find(a.d.data()), ib = g_codes.find(b.d.data());
        if (ib == g_codes.end() || !ib->second.add) {          // a float add
            if ((ia != g_codes.end() && !ia->second.fl) || (ib != g_codes.end() && !ib->second.fl))
                throw std::runtime_error("YC: an add reads a tensor that is held as codes only");
            return false;
        }
        kb = ib->second;
    }
    Y26_PROF("add on codes");
    const size_t HW = (size_t)a.H * a.W;
    const ConvW& c = *kb.cons;
    IAdd r;
    y26_add_codes(&a, &b, c, kb.off, kb.xr, HW, r);
    Buf& B = buf();
    const size_t d0 = kb.xr->off + (size_t)kb.off * HW, n = (size_t)a.C * HW;
    const bool chk = yc_check(), pl = r.par != SIZE_MAX;
    if (pl) {                                      // the ops block makes the sum where the branch's codes are
        { Y26_PROF("ops block (PL)");  ops_run(B, (size_t)(r.qa - B.host), d0, d0, r.par, a.C, (int)(HW / Y26_OPS_UNIT), 0, 1); }
        if (chk) {
            B.from_dev(d0, n);
            for (int k = 0; k < a.C; ++k)
                add_chk(B.host + d0 + (size_t)k * HW, c, kb.off + k, HW, a.d.data() + (size_t)k * HW, b.d.data() + (size_t)k * HW);
        }
    } else {
        if ((d0 % 64 || n % 64) && r.byb != 1)
            throw std::runtime_error("add for " + c.name + ": the ARM and the PL both write planes that share cache lines");
        if (r.byb != 1) B.from_dev(d0, n);
        if (r.bya != 1) B.from_dev((size_t)(r.qa - B.host), n);
#pragma omp parallel for schedule(static) if (n >= pack_mt())
        for (int k = 0; k < a.C; ++k)
            add_ic(B.host + d0 + (size_t)k * HW, r, k, c, kb.off + k, HW, chk ? a.d.data() + (size_t)k * HW : nullptr, b.d.data() + (size_t)k * HW);
        B.to_dev(d0, n);
    }
    {
        std::lock_guard<std::mutex> g(g_cmu);
        g_codes[y.d.data()] = {kb.cons, kb.off, kb.xr, false, false, pl ? 2 : 1};
    }
    if (chk) return false;                         // the float sum too: later checks read it
#ifdef Y26_BOARD_SIM
    std::fill(y.d.begin(), y.d.end(), std::nanf(""));
#endif
    return true;
#else
    (void)a; (void)b; (void)y;
    return false;
#endif
}
void y26_tensor_free(void* p, size_t bytes) {
#ifdef Y26_YC
    { std::lock_guard<std::mutex> g(g_cmu);  g_codes.erase(static_cast<const float*>(p)); }
#endif
    uint8_t* q = static_cast<uint8_t*>(p);
    Buf& B = buf();
    if (q >= B.host && q < B.host + B.size) arena().give((size_t)(q - B.host), bytes);
    else ::operator delete(p);
}
void y26_board_weights(const Weights& W) {
#ifdef Y26_YC
    const char* off = std::getenv("Y26_YC");
    if (!(off && std::strcmp(off, "0") == 0)) g_W = &W;
#ifdef Y26_ATTN_INT
    static std::mutex pm;                          // the pseudo readers are made once: readers keep the pointers
    std::lock_guard<std::mutex> pg(pm);
    if (g_W && attn2_on())                         // per head: KD channels of q (scaled), KD of k, VD of v
        for (const auto& a : W.attn) {
            const auto iq = W.conv.find(a.first + ".qkv.conv");
            if (!a.second.on || iq == W.conv.end()) continue;
            const ConvW& qkv = iq->second;
            if (g_pseudo.count(qkv.name + "@attn")) continue;
            const int nh = std::max(qkv.ic / 64, 1), hd = qkv.ic / nh, kd = hd / 2;
            std::unique_ptr<ConvW> p(new ConvW());
            p->name = qkv.name + "@attn";
            p->ic = p->oc = qkv.oc;  p->kh = p->kw = p->sh = p->sw = p->groups = 1;  p->ph = p->pw = 0;  p->act = 0;
            p->quant = p->asym = p->perch = true;
            for (int ch = 0; ch < qkv.oc; ++ch) {
                const int d = ch % (2 * kd + hd);
                const FQ& g = d < kd ? a.second.q : d < 2 * kd ? a.second.k : a.second.v_mm;
                p->ssc.push_back(d < kd ? 1.f / std::sqrt((float)kd) : 1.f);
                p->sa_v.push_back(g.step);  p->lo_v.push_back(g.lo);
            }
            g_pseudo[p->name] = std::move(p);
        }
#endif
    return;
#endif
    const char* e = std::getenv("Y26_YQ");
    if (!(e && std::strcmp(e, "1") == 0)) return;
#ifdef Y26_YQ8
    g_W = &W;
#else
    throw std::runtime_error("Y26_YQ=1 needs a -DY26_YQ8 build (and the sol_YQ8 bitstream)");
#endif
}
// Frame loop (run_model.cpp): one split line per frame; keep this frame's rows + profile for the exit table.
void y26_frame_end(int frame) {
    {
        std::lock_guard<std::mutex> g(g_cmu);
        // Overlapped frames: another frame's tensor may be pending here.
        static const char* ie = std::getenv("Y26_INFLIGHT");
        static const bool ovl = ie && std::atoi(ie) > 1;
#ifdef Y26_YC
        (void)ovl;                                         // direct codes live and die with their tensor
        if (g_W) std::printf("[board] frame %d: %d code-mode edges; conv input bytes: %ld the kernel's as written, %ld made by the PL side blocks, %ld made by the ARM from codes, of %ld\n",
                             frame, g_ncoded, g_xkern, g_xops, g_xcode, g_xall);
        g_xkern = g_xcode = g_xall = g_xops = 0;
        if (g_oc_calls) std::printf("[board] frame %d: CHECK ops block: %ld calls, %ld codes vs C-model, %ld differ -> %s\n",
                                    frame, g_oc_calls, g_oc_n, g_oc_bad, g_oc_bad ? "MISMATCH" : "BIT-EXACT");
        g_oc_calls = g_oc_n = g_oc_bad = 0;
        if (g_up_n) std::printf("[board] frame %d: Y26_YC_CHECK %ld upsampled codes, %ld off the float's code by more than 1, %ld by more than 3\n",
                                frame, g_up_n, g_up_1, g_up_3);
        g_up_n = g_up_1 = g_up_3 = 0;
        for (auto it = g_xb.begin(); it != g_xb.end();) it = it->first.first == frame ? g_xb.erase(it) : std::next(it);   // a conv that never ran
        if (g_ia_n) std::printf("[board] frame %d: Y26_YC_CHECK %ld sums on codes, %ld off the float sum's code by more than 1, %ld by more than 3\n",
                                frame, g_ia_n, g_ia_1, g_ia_3);
        g_ia_n = g_ia_1 = g_ia_3 = 0;
#else
        if (!g_codes.empty() && !ovl)                      // a coded tensor nobody consumed: the edge table is wrong
            throw std::runtime_error("YQ: " + std::to_string(g_codes.size()) + " coded tensors unconsumed at frame end (first for " +
                                     g_codes.begin()->second.cons->name + ")");
        if (g_W) std::printf("[board] frame %d: %d code-mode edges\n", frame, g_ncoded);
#endif
        g_ncoded = 0;
        if (g_yc_n) std::printf("[board] frame %d: Y26_YC_CHECK %ld direct codes, %ld differ from the host quantizer, %ld by more than 1\n",
                                frame, g_yc_n, g_yc_diff, g_yc_far);
        g_yc_n = g_yc_diff = g_yc_far = 0;
    }
    {
        std::lock_guard<std::mutex> g(g_sxm);      // a sibling that never came (a skipped branch): free its X
        g_sx.clear();
    }
    if (g_fuse_checked) {                          // Y26_FUSE_CHECK: reached only if every fused plane matched
        std::printf("[board] frame %d: Y26_FUSE_CHECK %d fused sources byte-identical\n", frame, g_fuse_checked);
        std::printf("[board] frame %d: Y26_FUSE_CHECK %d prepacked/shared/table sources byte-identical\n", frame, g_ahead_checked);
        g_fuse_checked = g_ahead_checked = 0;
    }
    std::lock_guard<std::mutex> g(g_stats.m);
    double px = 0, pw = 0, in = 0, k = 0, out = 0, wall = 0;
    for (const auto& r : g_stats.rows) { px += r.px; pw += r.pw; in += r.in; k += r.kern; out += r.out; wall += r.wall; }
    const double span = g_stats.rows.empty() ? 0 : ms(g_stats.first, g_stats.last);
    std::printf("[board] frame %d: pack X %.1f | pack W %.1f | in %.1f | kernel %.1f | out %.1f | PS between convs %.1f"
                " | first->last conv %.1f ms\n", frame, px, pw, in, k, out, span - wall, span);
    g_stats.prev.swap(g_stats.rows);
    g_stats.rows.clear();
    g_stats.pfirst = g_stats.first; g_stats.plast = g_stats.last;
    std::lock_guard<std::mutex> p(Y26Prof::mu());
    g_stats.prevprof.swap(Y26Prof::tab());
    Y26Prof::tab().clear();
}
// Async unless Y26_ASYNC=0, or a CHECK run (which snapshots the whole buffer per conv).
bool y26_async() {
    static const bool on = [] {
        const char* a = std::getenv("Y26_ASYNC");
        const char* k = std::getenv("Y26_BOARD_CHECK");
        return !(a && std::strcmp(a, "0") == 0) && !(k && *k);
    }();
    return on;
}
