                      
"""Idempotent, resumable fine-tune of COCO-pretrained YOLO26s on URPC2018.

Safe to invoke repeatedly (that's the whole point — a crash-retry loop calls this
until it exits 0). Decides what to do from the on-disk state of the run dir:

  1. RUN/DONE exists            -> already finished; exit 0 (no accidental retrain).
  2. last.pt is *resumable*     -> YOLO(last).train(resume=True)  (optimizer + LR intact).
     (resumable == ckpt['epoch'] != -1 AND ckpt['optimizer'] is not None)
  3. last.pt exists but stripped-> epoch==-1 / optimizer==None means ultralytics already
     finalized it => run is complete; touch DONE, exit 0.
     (This guards the gotcha we hit on YOLO26m: train(resume=True) on a stripped ckpt
      silently falls back to a fresh coco8 run.)
  4. otherwise (fresh)          -> YOLO('yolo26s.pt') COCO weights -> fine-tune.

On normal completion it writes RUN/DONE and prints FINETUNE COMPLETE.

  conda run -n ueaod python training/yolo26s/train_ft_resumable.py
"""
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]                    
DATA = ROOT / "dataset/urpc2018_yolo/urpc2018.yaml"
PRETRAINED = ROOT / "training/yolo26s/pretrained/yolo26s_coco.pt"
PROJECT = ROOT / "training/yolo26s/runs"
NAME = "urpc2018_finetune"
RUN = PROJECT / NAME
LAST = RUN / "weights" / "last.pt"
DONE = RUN / "DONE"


def is_resumable(p: Path) -> bool:
    """True only if the checkpoint carries live training state (not a stripped ckpt)."""
    if not p.exists():
        return False
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[ft] could not read {p}: {e}")
        return False
    return ck.get("epoch", -1) != -1 and ck.get("optimizer") is not None


def mark_done():
    RUN.mkdir(parents=True, exist_ok=True)
    DONE.touch()
    print("FINETUNE COMPLETE")


def main():
    if DONE.exists():
        print(f"[ft] {DONE} present -> already finished; nothing to do.")
        return

    if is_resumable(LAST):
        print(f"[ft] resuming from {LAST} (optimizer + LR schedule intact)")
        model = YOLO(str(LAST))
        model.train(resume=True)
        mark_done()
        return

    if LAST.exists():
                                                                                            
        print(f"[ft] {LAST} is finalized (epoch=-1 / no optimizer) -> treating run as complete.")
        mark_done()
        return

                                                                                               
    print(f"[ft] fresh fine-tune from COCO-pretrained {PRETRAINED}")
    model = YOLO(str(PRETRAINED) if PRETRAINED.exists() else "yolo26s.pt")
    model.train(
        data=str(DATA),
        epochs=300,
        imgsz=640,
        batch=16,
        workers=8,
        seed=42,
        device=0,
        amp=True,
        close_mosaic=10,
        deterministic=False,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        val=True,
        plots=True,
    )
    mark_done()


if __name__ == "__main__":
    main()
