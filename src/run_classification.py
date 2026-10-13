"""Train + evaluate the downstream classifier on decolorized images.

Example:
    python run_classification.py --data-dir data/UCM --decolorizer bt601
    python run_classification.py --data-dir data/UCM_vit --decolorizer bt601

Swap --decolorizer for any method (bt601, bt709, average, or vit: / vit2: checkpoints)
to compare downstream accuracy under identical settings.
"""
import argparse
from decolorizers import get_any
from data import make_splits
import classify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="ImageFolder root (one subfolder per class)")
    ap.add_argument("--decolorizer", default="bt601", help="bt601/bt709/average, or vit:CKPT / vit2:CKPT")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dec = get_any(args.decolorizer)
    # Seed the TRAINING randomness too. Without this --seed controls only the
    # train/val split, so head init, loader shuffling and augmentation are
    # uncontrolled and rerunning the same seed gives a different answer.
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_ds, val_ds, classes = make_splits(
        args.data_dir, dec, val_split=args.val_split, seed=args.seed)
    print(f"decolorizer={args.decolorizer}  classes={len(classes)}  "
          f"train={len(train_ds)}  val={len(val_ds)}")
    _, best = classify.run(train_ds, val_ds, len(classes),
                           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    print(f"\nBEST val_acc ({args.decolorizer}): {best:.4f}")


if __name__ == "__main__":
    main()
