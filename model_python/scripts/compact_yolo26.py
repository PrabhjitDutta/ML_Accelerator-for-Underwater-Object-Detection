                      
"""Physically remove pruned (all-zero) channels from a weights dir.

Walks the trunk wiring (mirrors yolo26_network.cpp) with per-channel provenance, so residual adds union
their operands' live sets and every conv learns which input/output channels to keep. Slices weights,
per-OC scales, per-IC smooth scales and bias; writes the compacted weights dir plus compaction_map.json.
Only channels whose output is identically zero and that no residual add revives are dropped (lossless).
attn.qkv inputs stay full width: attention() derives num_heads/head_dim from them.

  python model_python/scripts/compact_yolo26.py [weights_dir]
"""
import os
import sys
import json
import shutil
import numpy as np
import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "/workspace/ckarfa/projects/UOD/final_models/pruned50/yolo26s_urpc2018_pruned50_fp32.pt"

                                                                                                      
class Compactor:
    """Walks the trunk carrying per-channel provenance to solve channel liveness.
    
    prov[i] = list of (conv, oc) summed into channel i. keep_out[conv] starts as the nonzero-filter mask and
    is unioned wherever a residual add revives a channel; keep_in = kept(input tensor)."""

                                                                                                    
                                                                                                      
                                                                                                      
                                                                                                 
                                                
    FOLD_PAIRS = {f"{root}.{i}.{j}.0.conv": f"{root}.{i}.{j}.1.conv"
                  for root in ("23.cv3", "23.one2one_cv3") for i in range(3) for j in range(2)}

    def __init__(self, nz, groups, fold=False):
        self.nz = nz
        self.groups = groups
        self.fold = fold
        self.conv_in = {}                                                                                 
        self.splits = {}                                                                              
        self.oc = {n: len(m) for n, m in nz.items()}
                                                                                                    
                                                                                                
                                                                          
        self.folded = {n for n in nz if fold and n in self.FOLD_PAIRS} if fold else set()
        self.keep_out = {n: (np.zeros_like(m) if n in self.folded else m.copy())
                         for n, m in nz.items()}                            

                      
    def conv(self, name, x):
        self.conv_in[name] = x
        if name in self.folded:
            assert len(x) == self.oc[name], (name, len(x), self.oc[name])
            self.keep_out[name] |= (self.nz[name] & self.kept(x))
        elif self.groups[name] > 1:
                                                                                                 
                                                                               
             
                                                                                                     
                                                                                                    
                                                                                                      
                                                                                              
            assert len(x) == self.oc[name], (name, len(x), self.oc[name])
            m = self.keep_out[name] | self.kept(x)
            self.keep_out[name] = m
            for o in np.flatnonzero(m):                                                            
                for (c, oo) in x[o]:
                    self.keep_out[c][oo] = True
        return [[(name, o)] for o in range(self.oc[name])]

    @staticmethod
    def concat(ts):
        out = []
        for t in ts:
            out += t
        return out

    @staticmethod
    def sl(t, a, n):
        return t[a:a + n]

    def add(self, A, B):
        assert len(A) == len(B), (len(A), len(B))
        return [A[i] + B[i] for i in range(len(A))]                                                   

                        
    def kept(self, t):
        return np.array([any(self.keep_out[c][o] for (c, o) in ch) for ch in t], dtype=bool)

    def force_full(self, t):
        for ch in t:
            for (c, o) in ch:
                self.keep_out[c][o] = True

    def apply_add_union(self, A, B):
        """A residual add revives channel i if live in EITHER operand -> both producers must keep it."""
        for i in range(len(A)):
            live = any(self.keep_out[c][o] for (c, o) in A[i] + B[i])
            if live:
                for (c, o) in A[i] + B[i]:
                    self.keep_out[c][o] = True

                                            
    def bottleneck(self, p, x, addr):
        y = self.conv(p + ".cv1.conv", x)
        y = self.conv(p + ".cv2.conv", y)
        if addr:
            self.apply_add_union(x, y)
            return self.add(x, y)
        return y

    def c3k(self, p, x):
        a = self.conv(p + ".cv1.conv", x)
        a = self.bottleneck(p + ".m.0", a, True)
        a = self.bottleneck(p + ".m.1", a, True)
        b = self.conv(p + ".cv2.conv", x)
        return self.conv(p + ".cv3.conv", self.concat([a, b]))

    def c3k2(self, i, x, mode):
        s = str(i)
        t = self.conv(s + ".cv1.conv", x)
        c = len(t) // 2
        self.splits[i] = c
        a, b = self.sl(t, 0, c), self.sl(t, c, len(t) - c)
        mb = self.bottleneck(s + ".m.0", b, True) if mode == 0 else self.c3k(s + ".m.0", b)
        return self.conv(s + ".cv2.conv", self.concat([a, b, mb]))

    def psablock(self, p, x):
                                                                                                     
                                                                                                      
                                                                                                      
        self.force_full(x)
        self.conv(p + ".attn.qkv.conv", x)                                          
        self.conv(p + ".attn.pe.conv", x)                                                        
        attn = self.conv(p + ".attn.proj.conv", x)                                 
        self.apply_add_union(x, attn)
        x2 = self.add(x, attn)
        f = self.conv(p + ".ffn.0.conv", x2)                                                        
        f = self.conv(p + ".ffn.1.conv", f)
        self.apply_add_union(x2, f)                                                         
        return self.add(x2, f)

    def c2psa(self, i, x):
        s = str(i)
        t = self.conv(s + ".cv1.conv", x)
        c = len(t) // 2
        self.splits[i] = c
        a, b = self.sl(t, 0, c), self.sl(t, c, len(t) - c)
        pb = self.psablock(s + ".m.0", b)
        return self.conv(s + ".cv2.conv", self.concat([a, pb]))

    def c3k2_attn(self, i, x):
        s = str(i)
        t = self.conv(s + ".cv1.conv", x)
        c = len(t) // 2
        self.splits[i] = c
        a, b = self.sl(t, 0, c), self.sl(t, c, len(t) - c)
        mb = self.bottleneck(s + ".m.0.0", b, True)
        mb = self.psablock(s + ".m.0.1", mb)
        return self.conv(s + ".cv2.conv", self.concat([a, b, mb]))

    def sppf(self, i, x):
        s = str(i)
        y0 = self.conv(s + ".cv1.conv", x)
        cc = self.concat([y0, y0, y0, y0])                                        
        y = self.conv(s + ".cv2.conv", cc)
        if len(y) == len(x):                                                  
            self.apply_add_union(y, x)
            return self.add(y, x)
        return y

    def head_box(self, root, i, f):
        p = f"{root}.{i}"
        y = self.conv(p + ".0.conv", f)
        y = self.conv(p + ".1.conv", y)
        return self.conv(p + ".2", y)

    def head_cls(self, root, i, f):
        p = f"{root}.{i}"
        y = self.conv(p + ".0.0.conv", f)
        y = self.conv(p + ".0.1.conv", y)
        y = self.conv(p + ".1.0.conv", y)
        y = self.conv(p + ".1.1.conv", y)
        return self.conv(p + ".2", y)

    def run(self):
        inp = [[("__input__", o)] for o in range(3)]
        self.keep_out["__input__"] = np.ones(3, bool)
        x0 = self.conv("0.conv", inp)
        x1 = self.conv("1.conv", x0)
        x2 = self.c3k2(2, x1, 0)
        x3 = self.conv("3.conv", x2)
        x4 = self.c3k2(4, x3, 0)
        x5 = self.conv("5.conv", x4)
        x6 = self.c3k2(6, x5, 1)
        x7 = self.conv("7.conv", x6)
        x8 = self.c3k2(8, x7, 1)
        x9 = self.sppf(9, x8)
        x10 = self.c2psa(10, x9)
        x12 = self.concat([x10, x6])                                             
        x13 = self.c3k2(13, x12, 1)
        x15 = self.concat([x13, x4])
        x16 = self.c3k2(16, x15, 1)
        x17 = self.conv("17.conv", x16)
        x18 = self.concat([x17, x13])
        x19 = self.c3k2(19, x18, 1)
        x20 = self.conv("20.conv", x19)
        x21 = self.concat([x20, x10])
        x22 = self.c3k2_attn(22, x21)
                                                                                             
                                                                           
        dumped = {"0": x0, "1": x1, "2": x2, "3": x3, "4": x4, "5": x5, "6": x6, "7": x7,
                  "8": x8, "9": x9, "10": x10, "11": x10, "12": x12, "13": x13, "14": x13,
                  "15": x15, "16": x16, "17": x17, "18": x18, "19": x19, "20": x20,
                  "21": x21, "22": x22}
        for i, f in enumerate((x16, x19, x22)):
            for root in ("23.cv2", "23.cv3", "23.one2one_cv2", "23.one2one_cv3"):
                (self.head_box if "cv2" in root else self.head_cls)(root, i, f)
        return dumped

                                                                                                      
def nonzero_masks(wdir, manifest_name, names):
    """bool[oc] per conv: an output filter is live iff not all-zero (dead filters == pruned channels)."""
    nz = {}
    for n in names:
        w = np.fromfile(os.path.join(wdir, f"{n}.w.bin"), dtype="<f4")
        oc = _oc_from_manifest(wdir, manifest_name, n)
        w = w.reshape(oc, -1)
        nz[n] = (np.abs(w).sum(axis=1) > 0)
    return nz

def _read_manifest(wdir):
    for mn in ("manifest_sq.txt", "manifest_int8.txt", "manifest.txt"):
        p = os.path.join(wdir, mn)
        if os.path.exists(p):
            rows = [ln.split() for ln in open(p) if ln.strip()]
            return mn, rows
    raise FileNotFoundError(f"no manifest in {wdir}")

def _oc_from_manifest(wdir, mn, name):
    for r in open(os.path.join(wdir, mn)):
        f = r.split()
        if f and f[0] == name:
            return int(f[1])
    raise KeyError(name)

def _act(v, act):
    return v / (1.0 + np.exp(-v)) if int(act) == 1 else v                                  

def _fold_dw_constants(wdir, mn, row_by, comp, keep_out):
    """Fold each dropped dw channel's constant act(bias) into the consuming 1x1's bias.
    Returns {conv_name: new dense bias array}.
    """
    out = {}
    for dw, pw in Compactor.FOLD_PAIRS.items():
        if dw not in keep_out:
            continue
        dropped = np.flatnonzero(~keep_out[dw])
        if dropped.size == 0:
            continue
        rd, rp = row_by[dw], row_by[pw]
                                                                                         
        C = _act(np.fromfile(os.path.join(wdir, f"{dw}.b.bin"), dtype="<f4")[dropped].astype(np.float64),
                 rd[11])
        oc_p, ic_p = int(rp[1]), int(rp[2])
        w = np.fromfile(os.path.join(wdir, f"{pw}.w.bin"), dtype="<f4").reshape(oc_p, ic_p)[:, dropped]
        b = np.fromfile(os.path.join(wdir, f"{pw}.b.bin"), dtype="<f4").astype(np.float64)
        if mn == "manifest_sq.txt" and int(rp[10]) == 1:
                                                                                                   
                                                                                          
            sa, lo = float(rp[12]), float(rp[13])
            sw = np.fromfile(os.path.join(wdir, f"{pw}.sw.bin"), dtype="<f4").astype(np.float64)
            ssc = np.fromfile(os.path.join(wdir, f"{pw}.ssc.bin"), dtype="<f4").astype(np.float64)[dropped]
            Q = np.clip(np.round((C * ssc - lo) / sa), 0, 255)
            b += sw * (sa * (w * Q).sum(axis=1) + lo * w.sum(axis=1))
        elif mn == "manifest_int8.txt":
            sa = float(rp[12])
            sw = np.fromfile(os.path.join(wdir, f"{pw}.sw.bin"), dtype="<f4").astype(np.float64)
            Q = np.clip(np.round(C / sa), -127, 127)
            b += sa * sw * (w * Q).sum(axis=1)
        else:
                                                                                                  
            contrib = (w * C).sum(axis=1)
            if int(rp[10]) != 0 and mn == "manifest.txt":                                          
                contrib *= np.fromfile(os.path.join(wdir, f"{pw}.s.bin"), dtype="<f4").astype(np.float64)
            b += contrib
        out[pw] = b.astype("<f4")
    return out

def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    fold = "--fold" in sys.argv
    wdir = argv[0] if argv else os.path.join(HERE, "..", "weights")
    wdir = os.path.abspath(wdir)
    outdir = wdir + ("_compact_fold" if fold else "_compact")
    mn, rows = _read_manifest(wdir)
    names = [r[0] for r in rows]
    row_by = {r[0]: r for r in rows}
    nz = nonzero_masks(wdir, mn, names)

                                                                                             
                                                                                                 
                                                                  
    groups = {r[0]: int(r[9]) for r in rows}
    comp = Compactor(nz, groups, fold=fold)
    for it in range(1, 21):
        before = {n: m.copy() for n, m in comp.keep_out.items()}
        dumped = comp.run()
        if all(np.array_equal(before[n], comp.keep_out[n]) for n in before):
            print(f"[compact] liveness fixpoint after {it} pass(es)")
            break
    else:
        raise RuntimeError("liveness did not converge")
    missing = [n for n in names if n not in comp.conv_in]
    assert not missing, f"convs never visited by the graph walk: {missing}"

                                          
    keep_in = {n: comp.kept(comp.conv_in[n]) for n in names}
    keep_out = comp.keep_out
    os.makedirs(outdir, exist_ok=True)
                                                                                               
    bias_override = _fold_dw_constants(wdir, mn, row_by, comp, keep_out) if fold else {}
    if fold:
        nfold = sum(int((~keep_out[dw]).sum()) for dw in Compactor.FOLD_PAIRS if dw in keep_out)
        print(f"[compact] constant-folded {nfold} dw channels into {len(bias_override)} 1x1 biases")

    cmap = {"source": wdir, "manifest": mn, "convs": {}, "splits": comp.splits}
    tot_p0 = tot_p1 = 0
    for n in names:
        r = row_by[n]
        oc, ic, kh, kw = int(r[1]), int(r[2]), int(r[3]), int(r[4])
        groups = int(r[9])
        ko = keep_out[n]
        ki = keep_in[n]
        assert ki.size == ic, (n, ki.size, ic)
        w = np.fromfile(os.path.join(wdir, f"{n}.w.bin"), dtype="<f4").reshape(oc, ic // groups, kh, kw)
        if groups == 1:
            wc = w[ko][:, ki, :, :]
            new_ic = int(ki.sum())
        else:                                                                                      
            assert np.array_equal(ko, ki), f"{n}: depthwise keep_out != keep_in"
            wc = w[ko]
            new_ic = int(ko.sum())
        new_oc = int(ko.sum())
        tot_p0 += w.size
        tot_p1 += wc.size
        wc.astype("<f4").tofile(os.path.join(outdir, f"{n}.w.bin"))
        for suf in ("b", "sw", "s"):                                                    
            p = os.path.join(wdir, f"{n}.{suf}.bin")
            if os.path.exists(p):
                v = bias_override[n] if (suf == "b" and n in bias_override) \
                    else np.fromfile(p, dtype="<f4")
                v[ko].astype("<f4").tofile(os.path.join(outdir, f"{n}.{suf}.bin"))
                                                                                                     
                                                                                                   
                                                                                                        
        for suf in ("ssc", "sa", "lo"):
            p = os.path.join(wdir, f"{n}.{suf}.bin")
            if os.path.exists(p):
                np.fromfile(p, dtype="<f4")[ki].astype("<f4").tofile(
                    os.path.join(outdir, f"{n}.{suf}.bin"))
        cmap["convs"][n] = {"oc": new_oc, "ic": new_ic, "groups": groups,
                            "keep_out": np.flatnonzero(ko).tolist(), "keep_in": np.flatnonzero(ki).tolist()}

                                                                                                     
    _write_manifest(wdir, outdir, mn, rows, cmap)
                                                                       
    comp_splits = {}
    for i, c in comp.splits.items():
        cv1 = f"{i}.cv1.conv"
        ko = keep_out[cv1]
        comp_splits[i] = int(ko[:c].sum())                                                
    cmap["comp_splits"] = comp_splits
                                                                                              
    cmap["dumps"] = {k: np.flatnonzero(comp.kept(t)).tolist() for k, t in dumped.items()}
    json.dump(cmap, open(os.path.join(outdir, "compaction_map.json"), "w"), indent=1)
                                                                                                    
                                                                                                  
                            
    afq = os.path.join(wdir, "attn_fq.txt")
    if os.path.exists(afq):
        shutil.copy(afq, os.path.join(outdir, "attn_fq.txt"))
                                                                                                  
    with open(os.path.join(outdir, "splits.txt"), "w") as f:
        for i in sorted(comp_splits, key=int):
            f.write(f"{i} {comp_splits[i]}\n")

    live_oc = sum(int(keep_out[n].sum()) for n in names)
    all_oc = sum(len(keep_out[n]) for n in names)
    print(f"[compact] {wdir} -> {outdir}")
    print(f"[compact] output channels kept {live_oc}/{all_oc} ({100*(1-live_oc/all_oc):.1f}% dropped); "
          f"conv weights {tot_p1/1e6:.3f}M / {tot_p0/1e6:.3f}M ({100*(1-tot_p1/tot_p0):.1f}% smaller)")
    _verify_lossless(wdir, outdir, mn, rows, keep_in, keep_out, nz, comp.folded)

def _write_manifest(wdir, outdir, mn, rows, cmap):
    out = []
    for r in rows:
        r = list(r)
        c = cmap["convs"][r[0]]
        r[1] = str(c["oc"])
        r[2] = str(c["ic"])
        if int(r[9]) > 1:                                                                              
            r[9] = str(c["oc"])
        out.append(" ".join(r))
    open(os.path.join(outdir, mn), "w").write("\n".join(out) + "\n")

def _verify_lossless(wdir, outdir, mn, rows, keep_in, keep_out, nz, folded=frozenset()):
    """Check every dropped channel's output is identically zero: all-zero filter AND all-zero bias, or
    (depthwise) a dead input channel.
    """
    bad = 0
    for r in rows:
        n, oc, ic, kh, kw, groups = r[0], int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[9])
        w = np.fromfile(os.path.join(wdir, f"{n}.w.bin"), dtype="<f4").reshape(oc, ic // groups, kh, kw)
        b = np.fromfile(os.path.join(wdir, f"{n}.b.bin"), dtype="<f4")
        ko, ki = keep_out[n], keep_in[n]
        if n in folded and not np.array_equal(ko, ki):
            print(f"[verify] FAIL {n}: folded dw keep_out != keep_in"); bad += 1
        elif n in folded:
            continue                                                                                
        if groups > 1 and not np.array_equal(ko, ki):
            print(f"[verify] FAIL {n}: depthwise keep_out != keep_in (oc/ic/groups would disagree)")
            bad += 1
        dead = ~ko
        if dead.any():
            if np.abs(w[dead]).sum() != 0:
                print(f"[verify] FAIL {n}: dropped a nonzero output filter"); bad += 1
            if np.abs(b[dead]).sum() != 0:
                print(f"[verify] FAIL {n}: dropped a filter with nonzero bias (constant channel)"); bad += 1
    print(f"[verify] lossless check: "
          f"{'OK (every dropped channel is identically zero)' if bad == 0 else f'{bad} FAILURES'}")

if __name__ == "__main__":
    main()
