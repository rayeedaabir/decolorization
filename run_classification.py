"""Train + evaluate the downstream classifier on decolorized images.

Example:
    python run_classification.py --data-dir data/UCM --decolorizer bt601
    python run_classification.py --data-dir data/AID --decolorizer chroma_demo --epochs 15

Swap --decolorizer for any registered method (bt601, bt709, average, chroma_demo,
and vit: / vit2: checkpoints) to compare downstream accuracy on identical settings.
"""
import argparse
from decolorizers import get, REGISTRY
from data import make_splits
import classify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="ImageFolder root (one subfolder per class)")
    ap.add_argument("--decolorizer", default="bt601", choices=sorted(REGISTRY))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dec = get(args.decolorizer)
    train_ds, val_ds, classes = make_splits(
        args.data_dir, dec, val_split=args.val_split, seed=args.seed)
    print(f"decolorizer={args.decolorizer}  classes={len(classes)}  "
          f"train={len(train_ds)}  val={len(val_ds)}")
    _, best = classify.run(train_ds, val_ds, len(classes),
                           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    print(f"\nBEST val_acc ({args.decolorizer}): {best:.4f}")


if __name__ == "__main__":
    main()
