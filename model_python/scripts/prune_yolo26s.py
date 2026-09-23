                      
"""Structured channel pruning of a fine-tuned YOLO26s with torch_pruning.

Workarounds for the Ultralytics DetectionModel:
  1. loaded checkpoints have requires_grad=False, which gives tp an empty graph: re-enable grad.
  2. the Detect decode is untraceable: trace the conv-only Yolo26Trunk (shares the submodules).
  3. tp's grouper loops forever on the attention blocks (layer 10 C2PSA, layer 22 C3k2): replace them
     with channel-preserving pass-throughs while tracing, then restore.
Protected: layer 10, layer 22, Detect (23) and SPPF's output conv. The rest is pruned at one global ratio.
Saves the pickled pruned DetectionModel, so YOLO(path) reloads it without a .yaml.

  python model_python/scripts/prune_yolo26s.py --init <fp32.pt> --ratio 0.3 --out <pruned_init.pt>
"""
import argparse
import datetime
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/yolo26s"))
from yolo26_trunk import Yolo26Trunk

C2PSA_IDX = 10
SPPF_IDX = 9
ATTN_HEAD_IDX = 22                                              
MIN_KEEP = 8                                                        

class PassThrough(nn.Module):
    """Channel-preserving stand-in for C2PSA during tracing; carries ultralytics routing attrs."""
    def __init__(self, f, i):
        super().__init__(); self.f = f; self.i = i
    def forward(self, x):
        return x

class AttnIdentity(nn.Module):
    """Stand-in for a nested Attention submodule during tracing (channel-preserving)."""
    def forward(self, x):
        return x

class RootMagnitudeImportance(tp.importance.Importance):
    """Group importance = L2 magnitude of the group's root conv weights only (tp's GroupMagnitudeImportance
    mis-indexes across YOLO26's concat/split boundaries)."""
    def __call__(self, group, **kwargs):
        for dep, idxs in group:
            layer = dep.target.module
            if isinstance(layer, nn.Conv2d) and \
               getattr(dep.handler, "__name__", "") == "prune_out_channels":
                s = layer.weight.data.flatten(1).norm(dim=1)                          
                return s[torch.tensor(idxs, device=s.device)]
        dep, idxs = group[0]
        s = dep.target.module.weight.data.flatten(1).norm(dim=1)
        return s[torch.tensor(idxs)]

def _neutralize_attn(model):
    """Swap layer-10 C2PSA -> PassThrough and every nested `.attn` -> AttnIdentity for tracing.
    Returns a restore() closure that puts the real modules back."""
    real_c2psa = model.model[C2PSA_IDX]
    model.model[C2PSA_IDX] = PassThrough(real_c2psa.f, real_c2psa.i)
    saved = []
    for name, mod in list(model.named_modules()):
        if name.endswith(".attn") and hasattr(mod, "qkv"):
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            saved.append((parent, mod))
            parent.attn = AttnIdentity()

    def restore():
        model.model[C2PSA_IDX] = real_c2psa
        for parent, mod in saved:
            parent.attn = mod
    return restore

def trunk_macs(model):
    """Conv-only FPGA-workload MACs. Traces with attention neutralized so tp's tracer terminates."""
    restore = _neutralize_attn(model)
    try:
        macs, _ = tp.utils.count_ops_and_params(Yolo26Trunk(model).eval(), torch.randn(1, 3, 640, 640))
    finally:
        restore()
    return macs

def prune(init, ratio, out):
    m = YOLO(str(init))
    model = m.model.eval().float().cpu()
    for p in model.parameters():
        p.requires_grad_(True)                                                  

    c2psa, detect, head22 = model.model[C2PSA_IDX], model.model[-1], model.model[ATTN_HEAD_IDX]
    c2psa_p0 = sum(p.numel() for p in c2psa.parameters())
    det_p0 = sum(p.numel() for p in detect.parameters())
    params0 = sum(p.numel() for p in model.parameters())
    macs0 = trunk_macs(model)
    print(f"[prune] BEFORE: params={params0/1e6:.3f}M  trunk MACs={macs0/1e9:.3f}G "
          f"(C2PSA={c2psa_p0/1e6:.3f}M Detect={det_p0/1e6:.3f}M layer22 protected)")

                                                                                                 
    ignored = [model.model[SPPF_IDX].cv2.conv]
    ignored += [x for x in detect.modules() if isinstance(x, (nn.Conv2d, nn.Linear))]
    ignored += [x for x in head22.modules() if isinstance(x, (nn.Conv2d, nn.Linear))]

    restore = _neutralize_attn(model)                                                                
    trunk = Yolo26Trunk(model).eval()
    pruner = tp.pruner.MetaPruner(
        trunk, torch.randn(1, 3, 640, 640), importance=RootMagnitudeImportance(),
        pruning_ratio=ratio, ignored_layers=ignored, iterative_steps=1,
    )
    ng = len(list(pruner.DG.get_all_groups(ignored_layers=pruner.ignored_layers,
                                           root_module_types=pruner.root_module_types)))
    print(f"[prune] {ng} prunable groups; pruning at ratio={ratio} ...")
    assert ng > 0, "no prunable groups"
    pruner.step()                                                                            
    restore()                                                                          

                                                                           
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 640, 640))
    params1 = sum(p.numel() for p in model.parameters())
    macs1 = trunk_macs(model)
    assert sum(p.numel() for p in c2psa.parameters()) == c2psa_p0, "C2PSA changed!"
    assert sum(p.numel() for p in detect.parameters()) == det_p0, "Detect changed!"
    assert params1 < params0, "no params removed"
    print(f"[prune] forward OK ({type(y).__name__})")
    print(f"[prune] AFTER : params={params1/1e6:.3f}M  trunk MACs={macs1/1e9:.3f}G  "
          f"(-{100*(1-params1/params0):.1f}% params, -{100*(1-macs1/macs0):.1f}% MACs)")

                                                                   
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    for p in model.parameters():
        p.requires_grad_(False)
    ckpt = {
        "model": deepcopy(model).half(), "epoch": -1, "best_fitness": None, "optimizer": None,
        "train_args": dict(getattr(m, "overrides", {}) or {}),
        "date": datetime.datetime.now().isoformat(), "version": ultralytics.__version__,
    }
    torch.save(ckpt, str(out))
    print(f"[prune] saved -> {out}")

                             
    m2 = YOLO(str(out))
    with torch.no_grad():
        _ = m2.model.float().eval()(torch.randn(1, 3, 640, 640))
    print(f"[prune] reload OK via YOLO(): params={sum(p.numel() for p in m2.model.parameters())/1e6:.3f}M")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "final_models/fine_tuned_fp32/yolo26s_urpc2018_fp32.pt"))
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prune(args.init, args.ratio, args.out)

if __name__ == "__main__":
    main()
