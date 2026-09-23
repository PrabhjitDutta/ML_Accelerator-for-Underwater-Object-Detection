// board_host.cpp - ZCU102 host for the shipping YA128 kernel (bringup/ya128_250_h, impl_2).
//
// Hooks the single conv choke point: with -DY26_BOARD, conv2d() (layer_ops.h) sends every
// SmoothQuant integer conv (c.quant && c.asym - the tb gates' eligibility) here. The rest of the
// trunk - FP32 attn.pe convs, attention, upsample, concat, head - stays on the A53s, unchanged.
// Entry point is the existing run_model.cpp:  ./y26_board <weights_dir> <input.bin> <dump_dir>
//   or the frame loop:  ls frames/*.bin | ./y26_board <weights_dir> - <out_dir>   (steady state from frame 1 on:
//   weights packed and buffer pages faulted once; one split line per frame, the exit table is the last frame's)
//
// The host->DDR layout is the tb's (conv_engine_tb.cpp), written as the raw byte image the m_axi ports
// read: X = uint8 codes, ic*H*W flat; Wt = int8 codes at Y26_WIDX() inside Y26_WSTRIDE() per oc;
// wsc/bias/step_v/lo_v = float; Y = float at the padded row stride YS.
//
// ONE buffer, one heap (Arena): every Tensor's storage (NoInitAlloc -> y26_tensor_alloc), each conv's X codes
// (freed after its run), and every conv's weights + scales (from the top, packed ONCE on first use, never
// freed). The kernel writes Y straight into the output tensor when that tensor is in the buffer and its rows
// are unpadded (YS == OW) - no unpack copy; else into a scratch Y that is unpacked. A full buffer falls back
// to the plain heap (still correct, just copies); the summary prints the peak so the buffer can be sized.
// Async (default; Y26_ASYNC=0 or Y26_BOARD_CHECK turns it off): the trunk runs independent branches on
// threads (yolo26_network.cpp fork); the kernel is one mutex, so one branch's PS work overlaps another's conv.
//
// Backends:
//   -DY26_BOARD_SIM  : the buffer is host memory and the C-model y26_conv_top() reads it, reinterpreted as
//                      port words. So a SIM run checks everything here except the silicon and the bus.
//   default (board)  : needs root, the bitstream loaded, PL0 at 250 MHz, and a DDR buffer the kernel owns:
//     Y26_UDMABUF=udmabuf0     PREFERRED. Cached u-dma-buf buffer (ikwzm u-dma-buf module); packed in place,
//                              cache-synced per run. e.g. insmod u-dma-buf.ko udmabuf0=67108864.
//     Y26_BUF_PHYS/_SIZE       Fallback, no module: O_SYNC /dev/mem, UNCACHED. Packs into a host mirror and
//                              copies X / Y through Device memory every conv - expect it to dominate.
//   Y26_BOARD_CHECK=<substr> (board only): convs whose name contains <substr> are ALSO run through the
//                      C-model and compared bit-for-bit ('.' = all; the whole frame's C-model is ~9 s on
//                      the PC, expect a few x that on the A53s).
//   Y26_PACK_MT=<elements>  pack X uses OpenMP threads when a conv's X has >= this many elements (default 1048576,
//                      the PC optimum; 5 convs). Board: sweep 1048576 / 262144 / 65536 in the frame loop, keep the
//                      lowest `pack X` on frames >= 1. The first conv prints the OpenMP thread count - "OpenMP OFF"
//                      means the build had no -fopenmp and pack X + attention are single-core.
//   Y26_PACK_SCHED=dynamic  pack X's OpenMP schedule (default static). Sweep it with Y26_PACK_MT when async is on.
//   Y26_COHERENT=1     ONLY with overlay_coh/ (ps-notes 7): X/Y on coherent HPC ports, their cache syncs skipped.
//
// Board run (root; y26_zcu102.bit + .hwh side by side, bringup/ya128_250_h/overlay):
//   python3 -c "from pynq import Overlay; Overlay('y26_zcu102.bit')"   # or fpgautil -b; sets PL0
//   PL0 must read ~249975000 Hz (the .hwh's 249.975 MHz), else the timings are not the design's.
//   Buffer: >= 64 MB (PC peak 50 MB: tensors + X + weights; the summary prints it) - smaller still runs, with copies. A /dev/mem window must be memory Linux
//   does NOT own - a device-tree reserved-memory node with no-map, e.g. 64 MB at 0x60000000. Then:
//   YOLO26_HEADS_ONLY=1 Y26_UDMABUF=udmabuf0 ./y26_board weights input.bin dumps
//   YOLO26_HEADS_ONLY=1 Y26_BUF_PHYS=0x60000000 Y26_BUF_SIZE=0x4000000 ./y26_board weights input.bin dumps
//   YOLO26_HEADS_ONLY=1 writes only the 3 o2o head maps (not ~45 MB of per-layer debug dumps) and skips the
//   duplicate layer-10 attention tap (100 kernel convs, not 102). Add Y26_BOARD_CHECK=. on the first run.
//   The exit summary splits every conv into pack / in / kernel / out, and the PS time between convs.
#include "../hls_kernel/conv_engine.h"
#include "../reference_model/layer_ops.h"
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
struct Layout { size_t x, wt, wsc, bias, stp, lo, y, end; int yq = 0; size_t qs = 0, qlo = 0, qst = 0; };

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
    void to_dev(size_t, size_t) {}
    void for_write(size_t, size_t) {}
    void from_dev(size_t, size_t) {}
    double run(const Layout& L, const ConvW& c, int H, int W) {
        std::lock_guard<std::mutex> g(km);
        const auto t = clk::now();
        run_cmodel(host, L, c, H, W);
        return ms(t, clk::now());
    }
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
    Buf() {
        // Y26_UDMABUF=<name> (e.g. udmabuf0, from the u-dma-buf kernel module): a CACHED DMA buffer. The HP
        // ports are not cache-coherent, so every run is bracketed by explicit cache maintenance (sync()).
        // Default: Y26_BUF_PHYS/Y26_BUF_SIZE through O_SYNC /dev/mem - uncached, needs no module.
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
            // ONLY with the bd.tcl `coh` bitstream (gmem_act/out on HPC0/1, AxCACHE 1111, AxPROT 010) and CCI
            // snooping on. Weights stay on non-coherent HP1/HP3 and keep their sync. Prove it once with Y26_BOARD_CHECK=.
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
        // u-dma-buf maps page by page on fault: take all ~16K faults here, once, not inside the frames.
        if (!ub.empty()) for (size_t i = 0; i < size; i += 4096) (void)*reinterpret_cast<volatile const uint8_t*>(map + i);
        if (ub.empty()) { mirror.assign(size / 8, 0); host = reinterpret_cast<uint8_t*>(mirror.data()); }
        else host = map;
        std::printf("[board] buffer %s: phys 0x%llx, %zu B\n", ub.empty() ? "/dev/mem (uncached, host mirror)" : ub.c_str(),
                    (unsigned long long)phys, size);
    }
    // u-dma-buf cache maintenance on [off, off+n). dir: 1 = to device (clean), 2 = from device (invalidate).
    // The sysfs files stay open: one pwrite each (fopen/fprintf/fclose per value was ~1,200 opens a frame).
    // Y26_SYNC_CMD=1: ONE write, u-dma-buf's packed command "0x<offset:32><size&~15 | dir<<2 | 1>" (size rounded
    // up to 16 B - cache ops are line-granular anyway). Only for a driver that parses it: an old one would sync
    // its stored range instead - run Y26_BOARD_CHECK=. with it once before trusting it.
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
        w64(0x40, phys + L.stp);  w64(0x4c, phys + L.lo);   w64(0x58, phys + L.y);
        w32(0x64, H);          w32(0x6c, W);          w32(0x74, c.oc);      w32(0x7c, c.ic);
        w32(0x84, c.kh);       w32(0x8c, c.kw);       w32(0x94, c.sh);      w32(0x9c, c.sw);
        w32(0xa4, c.ph);       w32(0xac, c.pw);       w32(0xb4, c.groups);  w32(0xbc, c.act);
        w32(0xc4, c.perch ? 1 : 0);  wf(0xcc, c.sa);  wf(0xd4, c.lo);
#ifdef Y26_YQ8
        // PROVISIONAL until checked against sol_YQ8's xy26_conv_top_hw.h (args appended after lo, Vitis layout).
        enum { Y26_REG_YQ = 0xdc, Y26_REG_QS = 0xe4, Y26_REG_QLO = 0xf0, Y26_REG_QST = 0xfc };
        // HW lever B registers - offsets from the sol_YQ8 driver header (xy26_conv_top_hw.h). yq is written on EVERY
        // run: registers persist, and a float conv after a coded one must not inherit yq=1.
        w32(Y26_REG_YQ, (uint32_t)L.yq);
        w64(Y26_REG_QS, phys + L.qs);  w64(Y26_REG_QLO, phys + L.qlo);  w64(Y26_REG_QST, phys + L.qst);
#endif
        barrier();
        static std::unordered_map<const ConvW*, double> seen;   // this conv's kernel ms last frame (under km)
        const auto it = seen.find(&c);
        const auto t0 = clk::now();
        w32(0x00, 1);                                  // ap_start
        // Async: another branch can use this core while the PL works - sleep through 70% of a >1 ms conv (Linux
        // oversleeps ~0.1 ms), then spin. Serial runs and frame 1 spin throughout.
        if (it != seen.end() && it->second > 1.0 && y26_async())
            std::this_thread::sleep_for(std::chrono::microseconds((long)(it->second * 700)));
        while (!(reg[0] & 2))                          // ap_done (clear-on-read)
            if (ms(t0, clk::now()) > 5000) throw std::runtime_error("kernel timeout (5 s) on " + c.name);
        const double k = ms(t0, clk::now());
        seen[&c] = k;
        barrier();
        return k;
    }
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

// Weights + scales of one conv, packed once at the top of the arena on first use. Keyed by the ConvW, which
// lives in Weights for the whole run (a conv called twice - the full-dump layer-10 tap - hits the cache).
struct WSlot { size_t wt, wsc, bias, stp, lo, qs, qlo, qst; };
std::unordered_map<const ConvW*, WSlot> g_wslots;
std::mutex g_wmu;

// ---- HW lever B (-DY26_YQ8): code-mode edges ------------------------------------------------------------------
// A conv whose output feeds exactly ONE kernel conv and nothing else (51 edges) runs with yq=1: the kernel writes that
// consumer's uint8 code (its ssc/lo/step for those channels) into each Y slot, the host narrows the slots to bytes
// in the output tensor's own storage, and the consumer copies them into X instead of quantizing. The edges below
// are read off yolo26_network.cpp (2026-09-22); anything else read by an add/slice/concat/maxpool/upsample/decode
// stays float. A coded tensor reaching any consumer but its own throws (y26_take_codes). The frame's o2o bytes vs
// reference/ are the end-to-end check. Enabled only by y26_board_weights() (frame loop) AND Y26_YQ=1.
const Weights* g_W = nullptr;
struct Codes { const ConvW* cons; int off; };
int g_fuse_checked = 0;                             // fused sources verified by Y26_FUSE_CHECK (exit summary)
int g_ahead_checked = 0;                            // prepacked / sibling-shared / uint8-table sources verified (§13)
std::unordered_map<const float*, Codes> g_codes;   // coded tensor storage -> its one consumer
int g_ncoded = 0;                                   // edges coded this frame (frame line)
std::mutex g_cmu;
// producer -> (consumer, 0 = its first input source / 1 = its last)
const ConvW* yq_edge(const ConvW& c, int& off) {
    static const std::unordered_map<std::string, std::pair<std::string, int>> E = [] {
        std::unordered_map<std::string, std::pair<std::string, int>> e;
        auto ed = [&](const std::string& p, const std::string& q, int tail) { e[p] = {q, tail}; };
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
        return e;
    }();
    if (!g_W) return nullptr;
    const auto it = E.find(c.name);
    if (it == E.end()) return nullptr;
    const auto jt = g_W->conv.find(it->second.first);
    if (jt == g_W->conv.end()) throw std::runtime_error("YQ edge table: no conv " + it->second.first);
    const ConvW& q = jt->second;
    if (!(q.quant && q.asym)) return nullptr;                       // consumer runs on the CPU in float
    off = it->second.second ? q.ic - c.oc : 0;
    if (off < 0 || off + c.oc > q.ic) throw std::runtime_error("YQ edge " + c.name + " -> " + q.name + ": channels");
    return &q;
}
// The consumer side: true when t arrived as codes for (c, channel ch0) - claimed, so it is used once.
bool y26_take_codes(const Tensor* t, const ConvW& c, int ch0) {
    std::lock_guard<std::mutex> g(g_cmu);
    const auto it = g_codes.find(t->d.data());
    if (it == g_codes.end()) return false;
    if (it->second.cons != &c || it->second.off != ch0)
        throw std::runtime_error("YQ: a coded tensor for " + it->second.cons->name + " reached " + c.name);
    g_codes.erase(it);
    return true;
}

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
    if (const ConvW* q = yq_edge(c, qoff)) {        // the consumer's quantizer for these channels
        float* qs = reinterpret_cast<float*>(B.host + s.qs);
        float* ql = reinterpret_cast<float*>(B.host + s.qlo);
        float* qt = reinterpret_cast<float*>(B.host + s.qst);
        for (int o = 0; o < c.oc; ++o) { qs[o] = q->ssc[qoff + o]; ql[o] = q->lo_of(qoff + o); qt[o] = q->step_of(qoff + o); }
    }
    B.to_dev(s.wt, end - s.wt);
    return g_wslots.emplace(&c, s).first->second;
}
}  // namespace

// Fused sources (Src, layer_ops.h; ps-notes §10) straight to this conv's codes. Each equals quant_plane() of the
// materialized op, byte for byte (Y26_FUSE_CHECK=1 compares every fused plane against exactly that).
static void quant_sum(uint8_t* d, const float* a, const float* b, size_t n, float s, float lo, float st) {
    float t[4096];                                 // the residual sum, one cache-resident block at a time
    for (size_t i = 0; i < n; i += 4096) {
        const size_t m = std::min<size_t>(4096, n - i);
        for (size_t j = 0; j < m; ++j) t[j] = a[i + j] + b[i + j];
        quant_plane(d + i, t, m, s, lo, st);       // exact per element, so the block split cannot matter
    }
}
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
// maxpool(5,1,2) = max over rows then columns, windows clipped at the edges: the -inf pad never wins, the centre is
// always inside. Max commutes with a monotone q, so pooling the codes == coding the pool. Codes are >= 0, so a row
// padded with 0s gives the clipped window's max - every loop below is branch-free over x and vectorizes (§13).
__attribute__((optimize("vect-cost-model=dynamic")))
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

// Threads only for big inputs: the quantizer is vectorized, so a parallel region's start-up costs more than a
// small conv's whole pack (PC A/B, 2026-09-21). 1M threads only 5 convs = 28% of pack X's 27.6M elements/frame;
// the A53 pays far more per element than the PC, so its break-even is lower. Y26_PACK_MT=<elements> sets the
// threshold - sweep 1048576 / 262144 / 65536 on the board (frame >= 1 pack X) and keep the fastest.
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

// ---- §13 levers: X packed ahead, X shared by siblings, uint8 input ----------------------------------------------
// Same quantizer on every input channel, byte for byte.
static bool same_quant(const ConvW& a, const ConvW& b) {
    if (a.ic != b.ic || a.ssc.size() != b.ssc.size() || std::memcmp(a.ssc.data(), b.ssc.data(), a.ssc.size() * 4)) return false;
    for (int i = 0; i < a.ic; ++i) {
        const float l0 = a.lo_of(i), l1 = b.lo_of(i), s0 = a.step_of(i), s1 = b.step_of(i);
        if (std::memcmp(&l0, &l1, 4) || std::memcmp(&s0, &s1, 4)) return false;
    }
    return true;
}
// One PLAIN source packed ahead into its conv's X (y26_board_prepack). The X region is taken at once; the planes are
// packed on their own thread when async (deferred to the conv otherwise). The conv waits on `done` before its run.
struct Y26Pre {
    const ConvW* c; const float* d; int C, ch0; size_t HW;
    std::unique_ptr<Region> xr;
    std::future<void> done;
};
Pre y26_board_prepack(const Tensor* t, const ConvW& c, bool last) {
    if (!(c.quant && c.asym)) return nullptr;      // a CPU conv
    Y26_PROF("prepack X (issue)");
    auto p = std::make_shared<Y26Pre>();
    p->c = &c;  p->d = t->d.data();  p->C = t->C;  p->HW = (size_t)t->H * t->W;
    p->ch0 = last ? c.ic - t->C : 0;
    CSIM_HOST_ASSERT(p->ch0 >= 0 && p->ch0 + t->C <= c.ic, "prepack " + c.name + ": source wider than the conv");
    p->xr.reset(new Region(al((size_t)c.ic * p->HW), c.name));
    const bool coded = y26_take_codes(t, c, p->ch0);   // HW-B: the source arrived as this conv's codes
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
// Sibling convs reading the same tensor through the same quantizer (manifest step/lo + .ssc.bin equal, ps-notes
// §11): the first to arrive packs X, the other runs on that X. Checked at run time (same_quant, same source).
struct SharedX {
    const ConvW* c; const float* d;
    std::shared_ptr<Region> xr;
    std::promise<void> pr;
    std::shared_future<void> ready;                // set once X is packed and cleaned
};
std::unordered_map<int, std::shared_ptr<SharedX>> g_sx;
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

Tensor y26_board_conv(const std::vector<Src>& xs, const ConvW& c, const Pre& pre) {
    Y26_PROF("kernel conv (pack+PL+unpack)");
    const int H = xs[0].H(), W = xs[0].W();
    struct Plane { const Src* s; int k; };         // channel k of source s; s == nullptr: arrived as codes (HW-B)
    std::vector<Plane> src;                        // one plane per input channel: the concat, never built
    std::vector<std::pair<int, const Tensor*>> coded;   // (first channel, source) that arrived as codes
    bool pre_found = false;
    for (const Src& t : xs) {
        CSIM_HOST_ASSERT(t.H() == H && t.W() == W, "conv " + c.name + ": concat inputs differ in H,W");
        const bool p = pre && t.op == Src::PLAIN && t.a->d.data() == pre->d && (int)src.size() == pre->ch0 && t.C() == pre->C;
        pre_found |= p;
        const bool q = !p && t.op == Src::PLAIN && y26_take_codes(t.a, c, (int)src.size());
        if (q) coded.push_back({(int)src.size(), t.a});
        for (int k = 0; k < t.C(); ++k) src.push_back({p || q ? nullptr : &t, k});
    }
    CSIM_HOST_ASSERT((int)src.size() == c.ic, "conv " + c.name + ": input C mismatch");
    CSIM_HOST_ASSERT(!pre || (pre->c == &c && pre_found), "conv " + c.name + ": prepacked source not among its inputs");
    const auto t0 = clk::now();
    Buf& B = buf();
    const int OH = (H + 2 * c.ph - c.kh) / c.sh + 1, OW = (W + 2 * c.pw - c.kw) / c.sw + 1;
    const int YS = Y26_HOST_YS(OW);
    const size_t HW = (size_t)H * W, nx = (size_t)c.ic * HW, ny = (size_t)c.oc * OH * YS * 4;

    const WSlot& ws = weights_slot(B, c);
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
        const auto it = g_sx.find(pair);
        if (it == g_sx.end()) {
            sx = std::make_shared<SharedX>();
            sx->c = &c;  sx->d = xs[0].a->d.data();
            sx->xr = std::make_shared<Region>(al(nx), c.name);
            sx->ready = sx->pr.get_future().share();
            g_sx[pair] = sx;
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
    Region xr(pre || sx ? 0 : al(nx), c.name), yr(direct ? 0 : ny, c.name);
    Layout L;
    L.x = pre ? pre->xr->off : sx ? sx->xr->off : xr.off;
    L.wt = ws.wt;  L.wsc = ws.wsc;  L.bias = ws.bias;  L.stp = ws.stp;  L.lo = ws.lo;
    L.y = direct ? (size_t)(yp - B.host) : yr.off;
    L.end = L.y + ny;
    int qoff = 0;
    const ConvW* cons = yq_edge(c, qoff);
    L.yq = cons ? 1 : 0;  L.qs = ws.qs;  L.qlo = ws.qlo;  L.qst = ws.qst;

    uint8_t* X = B.host + L.x;                     // tb: pX (uint8 codes); Y26_XPE-packing is LE bytes
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
    // Y26_PACK_SCHED=dynamic: static splits ic evenly, so a thread sharing its core with another async branch
    // holds the whole region; dynamic hands out 4 planes at a time around it. Board: sweep with Y26_PACK_MT.
    static const bool pack_dyn = [] { const char* e = std::getenv("Y26_PACK_SCHED"); return e && std::strcmp(e, "dynamic") == 0; }();
    auto pack_ic = [&](int ic) {
        const Src* sp = src[ic].s;
        if (!sp || reuse) return;
        const float* a = sp->a->d.data() + (size_t)src[ic].k * sp->a->H * sp->a->W;
        uint8_t* d = X + (size_t)ic * HW;
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
            const bool fused = t.op != Src::PLAIN;
            const bool ahead = !fused && ((pre && t.a->d.data() == pre->d) || reuse || u8);
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
    for (const auto& e : coded)                    // HW lever B: already this conv's codes, channel-major
        std::memcpy(X + (size_t)e.first * HW, e.second->d.data(), (size_t)e.second->C * HW);
    if (!reuse) std::memset(X + nx, 0, al(nx) - nx);   // only the port-word tail needs zeros
    const auto t2 = clk::now();
    // Cached buffer: X (and the weights, at first use) are cleaned before the kernel reads them; Y's range is
    // cleaned before the kernel writes it, so no dirty line can later be evicted over Y. A reused sibling X was
    // cleaned by its packer.
    if (!B.coh) {                                  // coherent X/Y: the CCI snoops, nothing to maintain
        if (!reuse) B.to_dev(L.x, al(nx));
        B.for_write(L.y, ny);
    }
    if (sx && !reuse) ready.ok();
    const auto t3 = clk::now();

    int check = -1;
#ifndef Y26_BOARD_SIM
    static const char* sel = std::getenv("Y26_BOARD_CHECK");
    std::vector<uint64_t> ref;
    if (sel && *sel && c.name.find(sel) != std::string::npos) {   // inputs + every weight slot, before the run
        ref.assign(B.size / 8, 0);                 // CHECK forces serial (y26_async), so the image is stable
        std::memcpy(ref.data(), B.host, B.size);
    }
#endif
    const double kern = B.run(L, c, H, W);
    const auto t4 = clk::now();
    if (!B.coh) B.from_dev(L.y, ny);
    const auto t4b = clk::now();

    const float* Yf = reinterpret_cast<const float*>(B.host + L.y);
#ifndef Y26_BOARD_SIM
    if (!ref.empty()) {
        uint8_t* rb = reinterpret_cast<uint8_t*>(ref.data());
        run_cmodel(rb, L, c, H, W);
        const float* g = reinterpret_cast<const float*>(rb + L.y);
        check = 0;
        for (int o = 0; o < c.oc; ++o)
            for (int r = 0; r < OH; ++r) {
                const size_t i = ((size_t)o * OH + r) * YS;
                check += std::memcmp(Yf + i, g + i, (size_t)OW * 4) != 0;   // rows that differ
            }
    }
#endif
    const auto t5 = clk::now();
    if (cons) {
        // Code slots -> the consumer's X-plane bytes, in the output tensor's own storage. In place when direct
        // (byte i is written after slot i is read, and every later slot sits at byte >= 4i).
        uint8_t* d = reinterpret_cast<uint8_t*>(y.d.data());
        const uint32_t* s = reinterpret_cast<const uint32_t*>(B.host + L.y);
        for (size_t row = 0; row < (size_t)c.oc * OH; ++row)
            for (int w = 0; w < OW; ++w) d[row * OW + w] = (uint8_t)s[row * YS + w];
        std::lock_guard<std::mutex> g(g_cmu);
        g_codes[y.d.data()] = {cons, qoff};
        ++g_ncoded;
    } else if (!direct)                            // scratch Y: unpack the padded rows
    for (int o = 0; o < c.oc; ++o)
        for (int r = 0; r < OH; ++r)
            std::memcpy(&y.d[((size_t)o * OH + r) * OW], Yf + ((size_t)o * OH + r) * YS, (size_t)OW * 4);
    const auto t6 = clk::now();
    // out = invalidate/copy-back + unpack. The CHECK C-model (t4b->t5) is only in the conv total - don't time CHECK runs.
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
void y26_tensor_free(void* p, size_t bytes) {
    uint8_t* q = static_cast<uint8_t*>(p);
    Buf& B = buf();
    if (q >= B.host && q < B.host + B.size) arena().give((size_t)(q - B.host), bytes);
    else ::operator delete(p);
}
void y26_board_weights(const Weights& W) {
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
        if (!g_codes.empty())                      // a coded tensor nobody consumed: the edge table is wrong
            throw std::runtime_error("YQ: " + std::to_string(g_codes.size()) + " coded tensors unconsumed at frame end (first for " +
                                     g_codes.begin()->second.cons->name + ")");
        if (g_W) std::printf("[board] frame %d: %d code-mode edges\n", frame, g_ncoded);
        g_ncoded = 0;
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
