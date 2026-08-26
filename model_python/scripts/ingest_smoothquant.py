                      
"""Phase 3: ingest the *deployed* SmoothQuant INT8 scales from the OpenVINO IR and drive the same
C-sim with them, to reproduce the deployed ~0.7546-mAP model (vs Phase 2's hand-rolled max-abs PTQ).

Unlike Phase 2, SmoothQuant is NOT a drop-in for the symmetric per-tensor `sa`. Verified from the IR
(final_models/pruned50/int8_smoothquant_openvino_model/best.{xml,bin}):
  * every conv input carries a per-INPUT-channel smooth scale `ssc` (nncf_smooth_quant/scale [1,IC,1,1]),
  * activations are ASYMMETRIC uint8 (FakeQuantize levels=256, in_low<0 for 99/100 convs),
  * weights are per-OUTPUT-channel symmetric int8 (i8 Const -> Convert -> Multiply by sw [OC,1,1,1]).

The exact integer W8A8 dataflow a hand-written HLS kernel runs (attention interior stays FP32, like
Phase 2):
    X_s[i]   = X[i] * ssc[i]                                  # per-input-channel smooth pre-scale
    q[i]     = clamp(round((X_s[i] - lo)/step), 0, 255)       # asymmetric uint8, step=(hi-lo)/255
    Y[o]     = (sw[o]*step) * sum_i w_int[o,i]*q[i]  +  bias_eff[o]
    bias_eff[o] = bias[o] + sw[o]*lo*sum_i w_int[o,i]         # zero-point correction folded into bias
`sum w_int*q` is exact int32. This reuses the C-sim's existing dequant shape `sa*wsc[oc]` (sa:=step)
and bias plumbing (b:=bias_eff); the only genuinely new inputs are per-IC `ssc` and per-tensor `lo`.

Produces, mirroring quantize_yolo26.py:
  1. weights_sq/  -- per-conv int8 weights + sw + ssc + bias_eff + manifest_sq.txt (qmode/sa/lo cols).
  2. dumps_sq/<name>_python.bin -- per-layer activations of a PyTorch integer oracle built the SAME
     way, the arithmetic reference the C-sim must match (compare_cosine.py hls/dumps_sq 0.99).

IR node `__module.model.<path>/aten::_convolution/Convolution` maps 1:1 to C-sim conv name `<path>`.
The IR deploys the one2one (end2end) head only, so the ~24 one2many head convs (23.cv2.*/23.cv3.*)
have no IR entry -> exported as FP32 (qmode=0, BN-folded from the .pt) and are unused by the o2o decode.

  conda run -n ueaod python hls/export/ingest_smoothquant.py
"""
import os
import re
import sys
import json
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv

HERE = os.path.dirname(os.path.abspath(__file__))
WSQ = os.path.join(HERE, "..", "weights_sq")
DUMPS_FP32 = os.path.join(HERE, "..", "dumps")
DUMPS_SQ = os.path.join(HERE, "..", "dumps_sq")
sys.path.insert(0, "/workspace/ckarfa/projects/UOD/training/yolo26s")
from yolo26_trunk import Yolo26Trunk              

CKPT = "/workspace/ckarfa/projects/UOD/final_models/pruned50/yolo26s_urpc2018_pruned50_fp32.pt"
IRDIR = "/workspace/ckarfa/projects/UOD/final_models/pruned50/int8_smoothquant_openvino_model"
EPS = 1e-3                                                                                 

_ET = {"i8": np.int8, "u8": np.uint8, "i32": "<i4", "i64": "<i8",
       "f16": "<f2", "f32": "<f4", "f64": "<f8"}


                                                                                                      
class IR:
    """Minimal opset1 IR reader: id->layer, type, name, edge maps, and const-tensor bytes."""

    def __init__(self, xmlpath, binpath):
        root = ET.parse(xmlpath).getroot()
        self.L = {x.get("id"): x for x in root.iter("layer")}
        self.T = {i: x.get("type") for i, x in self.L.items()}
        self.NM = {i: x.get("name") for i, x in self.L.items()}
        self.src = {}                                             
        self.dst = {}                                       
        for e in root.find("edges").iter("edge"):
            fr, fp = e.get("from-layer"), e.get("from-port")
            to, tp = e.get("to-layer"), e.get("to-port")
            self.src[(to, tp)] = (fr, fp)
            self.dst.setdefault(fr, []).append((to, tp))
        self.blob = np.fromfile(binpath, dtype=np.uint8)

    def const(self, lid):
        d = self.L[lid].find("data")
        off, sz = int(d.get("offset")), int(d.get("size"))
        dt = _ET[d.get("element_type")]
        arr = np.frombuffer(self.blob[off:off + sz].tobytes(), dtype=dt)
        shp = d.get("shape", "")
        if shp:
            arr = arr.reshape([int(v) for v in shp.split(",")])
        return arr

    def convs(self):
        return [i for i, t in self.T.items() if t in ("Convolution", "GroupConvolution")]


def cname(irname):
    m = re.match(r"__module\.model\.(.*?)/aten", irname)
    return m.group(1) if m else None


                                                                                                       
                                                                                                    
                                                              
                                                                                                      
                                                                                                    
                                                                                                      
                                                                                                 
                                                                    
                                                                                    
                                                                       
                                                                    
                                                                                               
                                                                                                  
                                                                                                
                                                                                              
ATTN_FQ = {"q": "aten::mul/Multiply_1/fq_output_0",
           "k": "aten::matmul/MatMul/fq_input_1",
           "sm": "aten::softmax/Softmax/fq_output_0",
           "v_mm": "aten::matmul/MatMul_1/fq_input_0",
           "v_pe": "aten::reshape/Reshape/fq_input_0"}
ATTN_KEYS = ["q", "k", "sm", "v_mm", "v_pe"]


def parse_attn_fq(ir):
    """block prefix (e.g. '10.m.0.attn') -> {key: (lo, hi)} for the 5 attention-interior quantizers."""
    out = {}
    suffix2key = {v: k for k, v in ATTN_FQ.items()}
    for lid, t in ir.T.items():
        if t != "FakeQuantize" or ".attn/" not in ir.NM[lid]:
            continue
        nm = ir.NM[lid]
        key = next((k for s, k in suffix2key.items() if nm.endswith(s)), None)
        if key is None:
            continue
        levels = int(ir.L[lid].find("data").get("levels"))
        assert levels == 256, f"{nm}: levels={levels}, expected 256"
        lo = float(ir.const(ir.src[(lid, "1")][0]).reshape(-1)[0])
        hi = float(ir.const(ir.src[(lid, "2")][0]).reshape(-1)[0])
        ol = float(ir.const(ir.src[(lid, "3")][0]).reshape(-1)[0])
        oh = float(ir.const(ir.src[(lid, "4")][0]).reshape(-1)[0])
                                                                                                   
        assert np.isclose(lo, ol) and np.isclose(hi, oh), f"{nm}: in/out ranges differ"
        out.setdefault(nm.split("/")[0].replace("__module.model.", ""), {})[key] = (lo, hi)
    for blk, d in out.items():
        missing = set(ATTN_KEYS) - set(d)
        assert not missing, f"{blk}: missing attention FQ {sorted(missing)}"
    return out


def fq_np(x, lo, hi):
    """OV FakeQuantize with in==out ranges, 256 levels (torch tensor in, torch tensor out)."""
    step = float(np.float32((hi - lo) / 255.0))
    return torch.clamp(torch.round((x - lo) / step), 0.0, 255.0) * step + lo


def parse_conv(ir, cid):
    """Extract (cname, w_int[OC,ICg,KH,KW], sw[OC], ssc[IC]|None, lo, hi, bias[OC], has_fq)."""
    grouped = ir.T[cid] == "GroupConvolution"
                                                                                                       
    wmul = ir.src[(cid, "1")][0]
    assert ir.T[wmul] == "Multiply", f"{cname(ir.NM[cid])}: weight src {ir.T[wmul]}"
    w_src = ir.src[(wmul, "0")][0]
    w_i8 = ir.const(ir.src[(w_src, "0")][0] if ir.T[w_src] == "Convert" else w_src).astype(np.float32)
    sw = ir.const(ir.src[(wmul, "1")][0]).astype(np.float32).reshape(-1)            
    if grouped:                                                                                
        g, ocg, icg, kh, kw = w_i8.shape
        w_i8 = w_i8.reshape(g * ocg, icg, kh, kw)
    OC = w_i8.shape[0]
                                                                                                         
    a_id = ir.src[(cid, "0")][0]
    if ir.T[a_id] == "FakeQuantize":
                                                                                                    
                                                                                                     
                                                                                                     
                                                                                                     
        lo = ir.const(ir.src[(a_id, "1")][0]).astype(np.float32).reshape(-1)
        hi = ir.const(ir.src[(a_id, "2")][0]).astype(np.float32).reshape(-1)
        f0 = ir.src[(a_id, "0")][0]
        ssc = None
        if ir.T[f0] == "Multiply" and "nncf_smooth_quant" in ir.NM[f0]:
            ssc = ir.const(ir.src[(f0, "1")][0]).astype(np.float32).reshape(-1)         
        has_fq = True
    else:                                                                                             
        lo = hi = 0.0
        ssc = None
        has_fq = False
                                                                                         
    add_id = next(t for t, p in ir.dst[cid] if ir.T[t] == "Add")
    bias = ir.const(ir.src[(add_id, "1")][0]).astype(np.float32).reshape(-1)            
    assert bias.shape[0] == OC and sw.shape[0] == OC
    return cname(ir.NM[cid]), w_i8, sw, ssc, lo, hi, bias, has_fq


                                                                                                      
def fold_bn(model):
    """Fold each ultralytics Conv's BN into its Conv2d (bn->Identity). Returns is_silu name->bool."""
    c2w = {id(mod.conv): mod for mod in model.modules() if isinstance(mod, Conv)}
    is_silu = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        w = mod.weight.data
        wrapper = c2w.get(id(mod))
        if wrapper is not None:
            bn = wrapper.bn
            eps = getattr(bn, "eps", EPS)
            scale = bn.weight.data / torch.sqrt(bn.running_var.data + eps)
            mod.weight.data = w * scale.view(-1, 1, 1, 1)
            mod.bias = nn.Parameter(bn.bias.data - bn.running_mean.data * scale)
            is_silu[name] = isinstance(wrapper.act, nn.SiLU)
            wrapper.bn = nn.Identity()
        else:
            if mod.bias is None:
                mod.bias = nn.Parameter(torch.zeros(mod.weight.shape[0]))
            is_silu[name] = False
    return is_silu


                                                                                                      
def make_sq_forward(mod, w_int, sw, ssc, step, lo, bias):
    """Asymmetric-uint8 + per-input-channel-smooth integer conv (matches the C-sim conv2d exactly).

    real[o] = sw[o]*( step * Σ_ik w_int[o,ik]*q[ik]  +  lo * Σ_{ik in-image} w_int[o,ik] ) + bias[o]
    The zero-point (lo) term uses the VALID-tap weight sum (a conv of an all-ones map), so padded
    borders contribute real 0.0 -- exactly like OV -- instead of `lo`. Folding lo into a constant
    per-channel bias (lo*Σ_all w_int) is wrong at borders and its error propagates inward with depth.
    """
    stride, padding, dilation, groups = mod.stride, mod.padding, mod.dilation, mod.groups
    w_d = w_int.double()
    sw_v = torch.from_numpy(sw.astype(np.float32)).view(1, -1, 1, 1)
    bs = torch.from_numpy(bias.astype(np.float32)).view(1, -1, 1, 1)
    smt = torch.from_numpy(ssc.astype(np.float32)).view(1, -1, 1, 1)
                                                                                                    
                                                                                                  
                                                                                                 
                                                                                                  
    step_q = torch.as_tensor(step, dtype=torch.float32).view(1, -1, 1, 1)
    lo_q = torch.as_tensor(lo, dtype=torch.float32).view(1, -1, 1, 1)
    per_channel = step_q.numel() > 1
    if per_channel:
        assert groups == mod.in_channels == mod.out_channels, (
            f"{mod}: per-channel activation range on a non-depthwise conv -- the dequant would need "
            f"a different step per accumulated input channel, which int32 accumulation cannot do")
    step_d = step_q.double()
    lo_d = lo_q.double()

    def fwd(x):
        xs = x * smt                                                                                 
        q = torch.clamp(torch.round((xs - lo_q) / step_q), 0.0, 255.0)                           
        acc_q = F.conv2d(q.double(), w_d, None, stride, padding, dilation, groups)                 
        ones = torch.ones_like(q, dtype=torch.float64)
        acc_w = F.conv2d(ones, w_d, None, stride, padding, dilation, groups)                           
        return (acc_q * step_d + acc_w * lo_d).float() * sw_v + bs                                  
    return fwd


def make_attn_forward(mod, fq):
    """ultralytics Attention.forward with the IR's 5 interior quantizers inserted.

    Two deviations from the stock source, both required to land the quantization boundaries where OV
    puts them (not to change the math):
      - q is scaled BEFORE the matmul and quantized there, instead of scaling the q@k product after.
      - v is quantized TWICE, with different ranges, for its two consumers (matmul vs pe).
    The q@k product and the softmax itself stay float: OV puts no FakeQuantize on the MatMul output,
    and SoftMax is a float op there too, so a fully-integer attention would NOT match the reference.
    """
    nh, kd, hd, scale = mod.num_heads, mod.key_dim, mod.head_dim, mod.scale

    def fwd(x):
        B, C, H, W = x.shape
        N = H * W
        q, k, v = mod.qkv(x).view(B, nh, kd * 2 + hd, N).split([kd, kd, hd], dim=2)
        qs = fq_np(q * scale, *fq["q"])
        ks = fq_np(k, *fq["k"])
        attn = (qs.transpose(-2, -1) @ ks).softmax(dim=-1)
        attn = fq_np(attn, *fq["sm"])
        vm = fq_np(v, *fq["v_mm"])
        vp = fq_np(v, *fq["v_pe"])
        y = (vm @ attn.transpose(-2, -1)).view(B, C, H, W) + mod.pe(vp.reshape(B, C, H, W))
        return mod.proj(y)
    return fwd


def ingest(ckpt=CKPT):
    """Parse the OV IR, fold BN, and install per-conv SmoothQuant/FP32 forwards on a Yolo26Trunk.
    Returns (trunk, export, matched, unmatched): `trunk` is the integer oracle ready to run;
    `export` maps cname -> the weights_sq tensors; matched/unmatched are the conv-name splits."""
    ir = IR(os.path.join(IRDIR, "best.xml"), os.path.join(IRDIR, "best.bin"))

                                                      
    params = {}                                                                 
    for cid in ir.convs():
        nm, w_i8, sw, ssc, lo, hi, bias, has_fq = parse_conv(ir, cid)
        params[nm] = dict(w_i8=w_i8, sw=sw, ssc=ssc, lo=lo, hi=hi, bias=bias, has_fq=has_fq)
    print(f"[sq] parsed {len(params)} IR convs")

    m = YOLO(ckpt)
    trunk = Yolo26Trunk(m.model).float().eval()
    is_silu = fold_bn(trunk.model)
    all_convs = [(n, mod) for n, mod in trunk.model.named_modules() if isinstance(mod, nn.Conv2d)]
    print(f"[sq] model has {len(all_convs)} convs; folded BN")

                                                                                           
    matched = [n for n, _ in all_convs if n in params]
    unmatched = [n for n, _ in all_convs if n not in params]
    assert all(u.startswith(("23.cv2", "23.cv3")) for u in unmatched), \
        f"unexpected unmatched convs (not one2many head): {unmatched}"
    o2o_path = [n for n, _ in all_convs if not n.startswith(("23.cv2", "23.cv3"))]
    assert set(o2o_path) <= set(matched), \
        f"o2o/backbone convs missing IR scales: {sorted(set(o2o_path) - set(matched))}"
    print(f"[sq] matched(SmoothQuant)={len(matched)}  unmatched(FP32 one2many head)={len(unmatched)}")

                                                                                                  
    export = {}                                 
    n_asym = n_fp32 = 0
    for name, mod in all_convs:
        if name not in params:                                                                          
            export[name] = dict(qmode=0, w=mod.weight.data.cpu().numpy().astype(np.float32),
                                b=mod.bias.data.cpu().numpy().astype(np.float32), act=is_silu[name])
            n_fp32 += 1
            continue
        p = params[name]
        w_int = torch.from_numpy(p["w_i8"])
        sw, bias, ssc = p["sw"], p["bias"], p["ssc"]
        if p["has_fq"]:                                                                              
                                                                                                    
                                                                                                   
            step = np.float32((p["hi"] - p["lo"]) / 255.0).reshape(-1)
            lo = np.float32(p["lo"]).reshape(-1)
            if ssc is None:
                ssc = np.ones(mod.in_channels, dtype=np.float32)
            mod.forward = make_sq_forward(mod, w_int, sw, ssc, step, lo, bias)                        
            export[name] = dict(qmode=1, w=p["w_i8"].astype(np.float32), sw=sw.astype(np.float32),
                                ssc=ssc.astype(np.float32), b=bias.astype(np.float32),
                                sa=step, lo=lo, act=is_silu[name])
            n_asym += 1
        else:                                                                                         
            w_deq = (p["w_i8"] * sw.reshape(-1, 1, 1, 1)).astype(np.float32)
            mod.weight.data = torch.from_numpy(w_deq)
            mod.bias = nn.Parameter(torch.from_numpy(bias.astype(np.float32)))
            export[name] = dict(qmode=0, w=w_deq, b=bias.astype(np.float32), act=is_silu[name])
            n_fp32 += 1
    print(f"[sq] installed forwards: SmoothQuant-asym={n_asym}  FP32(one2many head)={n_fp32}")

                                                                                                     
    attn_fq = parse_attn_fq(ir)
    named = dict(trunk.model.named_modules())
    for blk, fq in attn_fq.items():
        assert blk in named, f"IR attention block {blk} has no module in the checkpoint"
        named[blk].forward = make_attn_forward(named[blk], fq)
    print(f"[sq] installed quantized attention on {len(attn_fq)} block(s): {sorted(attn_fq)}")
    return trunk, export, matched, unmatched, attn_fq


def main():
    os.makedirs(WSQ, exist_ok=True)
    os.makedirs(DUMPS_SQ, exist_ok=True)
    trunk, export, matched, unmatched, attn_fq = ingest()
    all_convs = [(n, mod) for n, mod in trunk.model.named_modules() if isinstance(mod, nn.Conv2d)]
    n_asym = sum(1 for e in export.values() if e["qmode"] == 1)
    n_fp32 = sum(1 for e in export.values() if e["qmode"] == 0)

                                                                                                      
    xin = np.fromfile(os.path.join(DUMPS_FP32, "input.bin"), dtype="<f4")
    assert xin.size == 3 * 640 * 640
    x = torch.from_numpy(xin.copy()).view(1, 3, 640, 640)
    feats, dhooks = {}, []

    def mk_dump(name):
        def hook(mod, inp, out):
            feats[name] = out[0] if isinstance(out, (list, tuple)) else out
        return hook

    for i in range(23):
        dhooks.append(trunk.model[i].register_forward_hook(mk_dump(str(i))))
    l10 = trunk.model[10]
    dhooks.append(l10.cv1.register_forward_hook(mk_dump("l10_cv1")))
    dhooks.append(l10.m[0].attn.register_forward_hook(mk_dump("l10_attn")))
    dhooks.append(l10.m[0].register_forward_hook(mk_dump("l10_psablock")))
    with torch.no_grad():
        out = trunk(x)

    def save(name, t, d):
        arr = t.detach().float().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        np.ascontiguousarray(arr, dtype="<f4").tofile(os.path.join(d, f"{name}_python.bin"))

    for name, t in feats.items():
        save(name, t, DUMPS_SQ)
    for i in range(trunk.nl):
        save(f"o2m_{i}", out[i], DUMPS_SQ)
        save(f"o2o_{i}", out[trunk.nl + i], DUMPS_SQ)
    for h in dhooks:
        h.remove()
    print(f"[sq] wrote SmoothQuant oracle dumps -> {DUMPS_SQ}")

                                                                                                       
    lines = []
    for name, mod in all_convs:
        e = export[name]
        e["w"].astype("<f4").tofile(os.path.join(WSQ, f"{name}.w.bin"))
        np.asarray(e["b"], dtype="<f4").tofile(os.path.join(WSQ, f"{name}.b.bin"))
        perch = 0
        if e["qmode"] == 1:
            e["sw"].astype("<f4").tofile(os.path.join(WSQ, f"{name}.sw.bin"))
            e["ssc"].astype("<f4").tofile(os.path.join(WSQ, f"{name}.ssc.bin"))
                                                                                                    
                                                                                                    
            perch = int(e["sa"].size > 1)
            if perch:
                e["sa"].astype("<f4").tofile(os.path.join(WSQ, f"{name}.sa.bin"))
                e["lo"].astype("<f4").tofile(os.path.join(WSQ, f"{name}.lo.bin"))
            sa, lo = float(e["sa"][0]), float(e["lo"][0])
        else:
            sa, lo = 0.0, 0.0
        kh, kw = mod.kernel_size
        lines.append(f"{name} {mod.out_channels} {mod.in_channels} {kh} {kw} "
                     f"{mod.stride[0]} {mod.stride[1]} {mod.padding[0]} {mod.padding[1]} "
                     f"{mod.groups} {e['qmode']} {1 if e['act'] else 0} {sa:.9e} {lo:.9e} {perch}\n")
    with open(os.path.join(WSQ, "manifest_sq.txt"), "w") as f:
        f.writelines(lines)
                                                                                         
    with open(os.path.join(WSQ, "attn_fq.txt"), "w") as f:
        for blk in sorted(attn_fq):
            vals = " ".join(f"{v:.9e}" for k in ATTN_KEYS for v in attn_fq[blk][k])
            f.write(f"{blk} {vals}\n")
    with open(os.path.join(WSQ, "manifest_sq.json"), "w") as f:
        json.dump({"source": IRDIR, "n_convs": len(all_convs), "n_smoothquant": n_asym,
                   "n_fp32_fallback": n_fp32,
                   "attn_fq": {b: {k: list(v) for k, v in d.items()} for b, d in attn_fq.items()},
                   "per_channel_act": sorted(n for n in matched if export[n]["qmode"] == 1
                                             and export[n]["sa"].size > 1),
                   "act_scales": {n: float(export[n]["sa"][0]) for n in matched
                                  if export[n]["qmode"] == 1}},
                  f, indent=1)
    print(f"[sq] exported {len(all_convs)} convs ({n_asym} SmoothQuant + {n_fp32} FP32) -> {WSQ}")


if __name__ == "__main__":
    main()
