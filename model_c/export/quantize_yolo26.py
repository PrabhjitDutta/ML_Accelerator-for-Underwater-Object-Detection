#!/usr/bin/env python3
"""Build the INT8 fake-quant reference: weights_int8/ (BN-folded, per-output-channel symmetric int8
weights, bias, manifest_int8.txt with each conv's input scale sa) and dumps_int8/<name>_python.bin
(oracle activations).

  sw[o]   = max|Wf[o]|/127,  w_int = round(Wf/sw)
  sa      = max|x|/127 (calibrated over N images),  x_int = clamp(round(x/sa), -127, 127)
  real[o] = (sum_i x_int[i]*w_int[o,i]) * (sa*sw[o]) + bias[o]
Integer accumulation is order-independent, so the C++ model matches the oracle exactly.
Attention matmuls stay FP32.

  python model_c/export/quantize_yolo26.py [n_calib]
"""
import os
import sys
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.data.augment import LetterBox

HERE = os.path.dirname(os.path.abspath(__file__))
WINT8 = os.path.join(HERE, "..", "weights_int8")
DUMPS_FP32 = os.path.join(HERE, "..", "dumps")
DUMPS_INT8 = os.path.join(HERE, "..", "dumps_int8")
sys.path.insert(0, "/workspace/ckarfa/projects/UOD/training/yolo26s")
from yolo26_trunk import Yolo26Trunk  # noqa: E402

CKPT = "/workspace/ckarfa/projects/UOD/final_models/pruned50/yolo26s_urpc2018_pruned50_fp32.pt"
VAL_IMAGES = "/workspace/ckarfa/projects/UOD/dataset/urpc 2018/val2018/images"
EPS = 1e-3        # ultralytics BatchNorm2d eps (NOT torch's 1e-5)
QMAX = 127        # symmetric signed INT8

def preprocess(path, size=640):
    im0 = cv2.imread(path)
    lb = LetterBox((size, size), auto=False, stride=32)
    im = lb(image=im0)[:, :, ::-1]                       # BGR->RGB
    im = np.ascontiguousarray(im.transpose(2, 0, 1))     # HWC->CHW
    return torch.from_numpy(im.astype(np.float32) / 255.0).unsqueeze(0)

def fold_bn(model):
    """Fold every ultralytics Conv's BN into its Conv2d (giving the conv a bias, bn->Identity).
    Returns (quant_convs, is_silu): the list of (name, Conv2d) to quantize in named order, and a
    name->bool map of whether that conv is followed by SiLU (Identity/bare convs -> False)."""
    conv2d_to_wrapper = {id(mod.conv): mod for mod in model.modules() if isinstance(mod, Conv)}
    quant_convs, is_silu = [], {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        w = mod.weight.data
        wrapper = conv2d_to_wrapper.get(id(mod))
        if wrapper is not None:
            bn = wrapper.bn
            eps = getattr(bn, "eps", EPS)
            scale = bn.weight.data / torch.sqrt(bn.running_var.data + eps)
            bias = bn.bias.data - bn.running_mean.data * scale
            mod.weight.data = w * scale.view(-1, 1, 1, 1)     # fold BN into weights
            mod.bias = nn.Parameter(bias.clone())
            is_silu[name] = isinstance(wrapper.act, nn.SiLU)
            wrapper.bn = nn.Identity()
        else:
            if mod.bias is None:
                mod.bias = nn.Parameter(torch.zeros(mod.weight.shape[0]))
            is_silu[name] = False                             # bare head-final 1x1: identity act
        quant_convs.append((name, mod))
    return quant_convs, is_silu

def main():
    n_calib = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    os.makedirs(WINT8, exist_ok=True)
    os.makedirs(DUMPS_INT8, exist_ok=True)
    torch.manual_seed(0)

    m = YOLO(CKPT)
    trunk = Yolo26Trunk(m.model).float().eval()
    quant_convs, is_silu = fold_bn(trunk.model)
    print(f"[quant] folded BN; {len(quant_convs)} convs to quantize")

    # pass 1: calibrate per-tensor input-activation ranges
    amax = {name: 0.0 for name, _ in quant_convs}
    handles = []

    def mk_calib(name):
        def pre(mod, inp):
            amax[name] = max(amax[name], float(inp[0].detach().abs().max()))
        return pre

    for name, mod in quant_convs:
        handles.append(mod.register_forward_pre_hook(mk_calib(name)))

    imgs = sorted(glob.glob(os.path.join(VAL_IMAGES, "*.jpg")))[:n_calib]
    assert imgs, f"no calib images in {VAL_IMAGES}"
    with torch.no_grad():
        for i, p in enumerate(imgs):
            trunk(preprocess(p))
            if (i + 1) % 32 == 0:
                print(f"[quant] calibrated {i+1}/{len(imgs)}")
    for h in handles:
        h.remove()
    # pin to float32 so the exact same scale round-trips through the text manifest into the C-sim
    sa = {name: float(np.float32(max(amax[name], 1e-8) / QMAX)) for name, _ in quant_convs}

    # quantize weights per output channel; replace each conv's forward with the integer W8A8 dataflow
    q_int = {}   # name -> (w_int tensor, sw tensor)

    def make_int_forward(mod, w_int, scale, sa_v):
        stride, padding, dilation, groups = mod.stride, mod.padding, mod.dilation, mod.groups
        bias = mod.bias.data.clone().view(1, -1, 1, 1)
        w_d, sc = w_int.double(), scale.view(1, -1, 1, 1)

        def fwd(x):
            xq = torch.clamp(torch.round(x / sa_v), -QMAX, QMAX)     # int8 activation codes
            acc = F.conv2d(xq.double(), w_d, None, stride, padding, dilation, groups)  # exact int32
            return acc.float() * sc + bias                          # dequant + FP32 bias (pre-act)
        return fwd

    for name, mod in quant_convs:
        w = mod.weight.data
        sw = w.abs().amax(dim=(1, 2, 3)).clamp_min(1e-8) / QMAX          # [OC]
        w_int = torch.clamp(torch.round(w / sw.view(-1, 1, 1, 1)), -QMAX, QMAX)
        scale = (sa[name] * sw).float()                                 # [OC] dequant scale
        mod.forward = make_int_forward(mod, w_int, scale, sa[name])
        q_int[name] = (w_int, sw)

    # pass 2: dump per-layer oracle activations
    xin = np.fromfile(os.path.join(DUMPS_FP32, "input.bin"), dtype="<f4")
    assert xin.size == 3 * 640 * 640, xin.size
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
        save(name, t, DUMPS_INT8)
    for i in range(trunk.nl):
        save(f"o2m_{i}", out[i], DUMPS_INT8)
        save(f"o2o_{i}", out[trunk.nl + i], DUMPS_INT8)
    for h in dhooks:
        h.remove()
    print(f"[quant] wrote INT8 oracle dumps -> {DUMPS_INT8}")

    # export weights_int8: .w.bin codes, .sw.bin, .b.bin; manifest_int8.txt: name oc ic kh kw sh sw ph pw
    # groups has_bn(=0) act sa
    lines = []
    for name, mod in quant_convs:
        w_int, sw = q_int[name]
        w_int.cpu().numpy().astype("<f4").tofile(os.path.join(WINT8, f"{name}.w.bin"))
        sw.cpu().numpy().astype("<f4").tofile(os.path.join(WINT8, f"{name}.sw.bin"))
        mod.bias.data.cpu().numpy().astype("<f4").tofile(os.path.join(WINT8, f"{name}.b.bin"))
        kh, kw = mod.kernel_size
        lines.append(f"{name} {mod.out_channels} {mod.in_channels} {kh} {kw} "
                     f"{mod.stride[0]} {mod.stride[1]} {mod.padding[0]} {mod.padding[1]} "
                     f"{mod.groups} 0 {1 if is_silu[name] else 0} {sa[name]:.9e}\n")
    with open(os.path.join(WINT8, "manifest_int8.txt"), "w") as f:
        f.writelines(lines)
    with open(os.path.join(WINT8, "manifest_int8.json"), "w") as f:
        json.dump({"eps": EPS, "qmax": QMAX, "n_calib": len(imgs),
                   "act_scales": {n: float(sa[n]) for n, _ in quant_convs}}, f, indent=1)
    print(f"[quant] exported {len(quant_convs)} int8 convs + manifest -> {WINT8}")
    print(f"[quant] act-scale range: min={min(sa.values()):.3e} max={max(sa.values()):.3e} "
          f"(n_calib={len(imgs)})")

if __name__ == "__main__":
    main()
