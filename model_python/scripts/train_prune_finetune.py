                      
"""Resumable fine-tune of a *pruned* YOLO26s init (accuracy recovery after channel pruning).

Same idempotent/resumable contract as train_ft_sweep.py, but the starting weights are a pruned
checkpoint produced by prune_yolo26s.py (whose channel counts differ from stock yolo26s) rather than
the fixed COCO-pretrained model. A crash-retry loop calls this until it exits 0; it decides what to do
from the on-disk state of the run dir. Gentle recovery recipe (lower lr0 + cosine) since pruning
perturbs the weights and the model must re-heal.

  conda run -n ueaod python training/yolo26s/train_prune_finetune.py \
      --init runs/prune_r30/pruned_init.pt --data <yaml> --name prune_r30 \
      [--epochs 200] [--patience 50] [--lr0 0.005]
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
SWEEP_DATA = ROOT / "dataset/urpc2018_yolo/urpc2018_sweep.yaml"
FULL_DATA = ROOT / "dataset/urpc2018_yolo/urpc2018.yaml"
PROJECT = ROOT / "training/yolo26s/runs"


def is_resumable(p: Path) -> bool:
    """True only if the checkpoint carries live training state (not a stripped/finalized ckpt)."""
    if not p.exists():
        return False
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[prune-ft] could not read {p}: {e}")
        return False
    return ck.get("epoch", -1) != -1 and ck.get("optimizer") is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="pruned init checkpoint from prune_yolo26s.py")
    ap.add_argument("--data", default=None, help="dataset yaml (default: sweep split)")
    ap.add_argument("--name", required=True, help="run dir name under runs/")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr0", type=float, default=0.005)
    args = ap.parse_args()

    data = Path(args.data) if args.data else SWEEP_DATA
    run = PROJECT / args.name
    last = run / "weights" / "last.pt"
    done = run / "DONE"

    if done.exists():
        print(f"[prune-ft:{args.name}] {done} present -> already finished; nothing to do.")
        return

    if is_resumable(last):
        print(f"[prune-ft:{args.name}] resuming from {last}")
        YOLO(str(last)).train(resume=True)
        run.mkdir(parents=True, exist_ok=True); done.touch()
        print(f"PRUNE_FT_DONE {args.name}")
        return

    if last.exists():
        print(f"[prune-ft:{args.name}] {last} finalized (epoch=-1/no optimizer) -> run complete.")
        run.mkdir(parents=True, exist_ok=True); done.touch()
        print(f"PRUNE_FT_DONE {args.name}")
        return

    print(f"[prune-ft:{args.name}] fresh recovery fine-tune: init={Path(args.init).name} data={data.name}")
    model = YOLO(str(args.init))                                                             
    model.train(
        data=str(data), epochs=args.epochs, imgsz=640, batch=16, workers=8, seed=42, device=0,
        amp=True, patience=args.patience, close_mosaic=10, deterministic=False,
        lr0=args.lr0, cos_lr=True, warmup_epochs=5.0,
        project=str(PROJECT), name=args.name, exist_ok=True, val=True, plots=True,
    )
    run.mkdir(parents=True, exist_ok=True); done.touch()
    print(f"PRUNE_FT_DONE {args.name}")


if __name__ == "__main__":
    main()
