                      
"""Export YOLO26s conv weights (BN folded) to per-conv float32 .bin files + a manifest.

Conv with BN: scale[o] = gamma[o]/sqrt(var[o]+eps), bias[o] = beta[o] - mean[o]*scale[o];
writes <name>.w.bin (OC*IC*KH*KW, PyTorch layout), .s.bin, .b.bin; output = conv(x)*scale + bias.
Bare Conv2d (Detect's final 1x1): .w.bin + .b.bin, scale 1.
Activation flags are read from the module, not hard-coded.

  python model_python/scripts/export_weights_yolo26.py
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "weights")
CKPT = "/workspace/ckarfa/projects/UOD/final_models/pruned50/yolo26s_urpc2018_pruned50_fp32.pt"
EPS = 1e-3                                                                  

def fuse_conv_bn(conv_w, gamma, beta, mean, var, eps):
    std = torch.sqrt(var + eps)
    scale = gamma / std
    bias = beta - mean * scale
    return scale.numpy().astype("<f4"), bias.numpy().astype("<f4")

def save_bin(path, arr):
    arr = np.asarray(arr, dtype="<f4")
    arr.tofile(path)
    back = np.fromfile(path, dtype="<f4")
    assert np.array_equal(arr.flatten(), back), f"round-trip failed {path}"
    return arr.nbytes

def act_name(act):
    if isinstance(act, nn.SiLU):
        return "silu"
    if isinstance(act, nn.Identity) or act is None:
        return "identity"
    raise ValueError(f"unexpected activation {type(act)}")

def main():
    os.makedirs(WEIGHTS, exist_ok=True)
    m = YOLO(CKPT)
    model = m.model.model.float().eval()

                                                                                             
    conv2d_to_wrapper = {}
    for mod in model.modules():
        if isinstance(mod, Conv):
            conv2d_to_wrapper[id(mod.conv)] = mod

    manifest = {"eps": EPS, "convs": {}}
    n_folded = n_bare = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        oc, ic_g = mod.weight.shape[0], mod.weight.shape[1]
        kh, kw = mod.kernel_size
        entry = {
            "oc": oc, "ic": mod.in_channels, "kh": kh, "kw": kw,
            "sh": mod.stride[0], "sw": mod.stride[1],
            "ph": mod.padding[0], "pw": mod.padding[1],
            "groups": mod.groups,
        }
        w = mod.weight.detach()                      
        wrapper = conv2d_to_wrapper.get(id(mod))
        if wrapper is not None:
            bn = wrapper.bn
            scale, bias = fuse_conv_bn(w, bn.weight.detach(), bn.bias.detach(),
                                       bn.running_mean.detach(), bn.running_var.detach(),
                                       getattr(bn, "eps", EPS))
            save_bin(os.path.join(WEIGHTS, f"{name}.w.bin"), w.numpy())
            save_bin(os.path.join(WEIGHTS, f"{name}.s.bin"), scale)
            save_bin(os.path.join(WEIGHTS, f"{name}.b.bin"), bias)
            entry["has_bn"] = True
            entry["act"] = act_name(wrapper.act)
            n_folded += 1
        else:
                                                                               
            save_bin(os.path.join(WEIGHTS, f"{name}.w.bin"), w.numpy())
            b = mod.bias.detach().numpy() if mod.bias is not None else np.zeros(oc, "<f4")
            save_bin(os.path.join(WEIGHTS, f"{name}.b.bin"), b)
            entry["has_bn"] = False
            entry["act"] = "identity"
            n_bare += 1
        manifest["convs"][name] = entry

    with open(os.path.join(WEIGHTS, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
                                                                                
                                                                       
    with open(os.path.join(WEIGHTS, "manifest.txt"), "w") as f:
        for name, e in manifest["convs"].items():
            f.write(f"{name} {e['oc']} {e['ic']} {e['kh']} {e['kw']} {e['sh']} {e['sw']} "
                    f"{e['ph']} {e['pw']} {e['groups']} {int(e['has_bn'])} "
                    f"{1 if e['act'] == 'silu' else 0}\n")
    print(f"[export] {len(manifest['convs'])} convs  ({n_folded} BN-folded, {n_bare} bare)")
    print(f"[export] wrote weights + manifest.json -> {WEIGHTS}")
                    
    k0 = next(iter(manifest["convs"]))
    print(f"[export] e.g. {k0}: {manifest['convs'][k0]}")

if __name__ == "__main__":
    main()
