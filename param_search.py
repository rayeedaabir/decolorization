"""Oracle experiment: does per-image adaptive fusion beat FIXED parameters?

For each image, grid-search Paper 3's fusion weights (alpha = chaos/Retinex
balance, beta = chroma weight) to maximise E-score, vs the fixed (0.9, 0.2). This
is the CEILING for learned adaptive fusion and writes per-image best (alpha,beta)
as ViT training targets (with the full image path, so CSVs can be pooled).

    # flat folder (Cadik / Color250)
    python param_search.py --images data/Cadik_rgb --out targets_cadik.csv
    # nested folder (UCM / AID) - stratified K images per class:
    python param_search.py --images data/UCM/images --per-class 20 --out targets_ucm.csv
    python param_search.py --images data/AID/data    --per-class 15 --out targets_aid.csv
"""
import argparse, glob, os, csv
from pathlib import Path
import numpy as np
from PIL import Image
from decolorizers import branch1_chaos, branch2_retinex, branch3_chroma, fuse
from metrics import contrast_metrics

ALPHAS = [0.6, 0.7, 0.8, 0.9, 1.0]
BETAS  = [0.0, 0.1, 0.2, 0.3, 0.4]
EXTS   = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.ppm", "*.tif")


def load(p, size=256):
    return np.asarray(Image.open(p).convert("RGB").resize((size, size)), np.float64) / 255


def collect(images_dir, recursive, per_class, seed=0):
    """Flat (default), fully recursive (--recursive), or K-per-class (--per-class)."""
    rng = np.random.default_rng(seed)
    if per_class:
        files = []
        for sub in sorted(p for p in Path(images_dir).iterdir() if p.is_dir()):
            fs = sorted(f for e in EXTS for f in glob.glob(str(sub / e)))
            if fs:
                files += [fs[i] for i in sorted(rng.permutation(len(fs))[:per_class])]
        return files
    if recursive:
        return sorted(f for e in EXTS for f in glob.glob(os.path.join(images_dir, "**", e), recursive=True))
    return sorted(f for e in EXTS for f in glob.glob(os.path.join(images_dir, e)))


def search(rgb):
    Lc = branch1_chaos(rgb); Lr = branch2_retinex(Lc); Cp = branch3_chroma(rgb)
    E = lambda a, b: contrast_metrics(rgb, fuse(Lc, Lr, Cp, alpha=a, beta=b))["E"]
    return E(0.9, 0.2), max(((E(a, b), a, b) for a in ALPHAS for b in BETAS), key=lambda t: t[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--out", default="targets.csv")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--per-class", type=int, default=0, help="sample K images per class subfolder")
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    files = collect(a.images, a.recursive, a.per_class)
    if a.n: files = files[:a.n]
    if not files:
        raise SystemExit(f"no images under {a.images} (nested dataset? try --per-class K or --recursive)")
    rows, fx, ox = [], [], []
    for f in files:
        rgb = load(f); fixed, (be, ba, bb) = search(rgb)
        rows.append([f, os.path.basename(f), ba, bb, round(fixed, 4), round(be, 4)])
        fx.append(fixed); ox.append(be)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path","image","best_alpha","best_beta","E_fixed","E_oracle"])
        w.writerows(rows)
    fx, ox = np.array(fx), np.array(ox)
    print(f"images={len(files)}")
    print(f"  E  fixed (alpha=0.9,beta=0.2): {fx.mean():.4f}")
    print(f"  E  per-image oracle          : {ox.mean():.4f}")
    print(f"  mean improvement             : {(ox-fx).mean():+.4f}  ({(ox-fx).mean()/fx.mean()*100:+.2f}%)")
    print(f"  best-alpha spread: {min(r[2] for r in rows)}..{max(r[2] for r in rows)}  "
          f"best-beta spread: {min(r[3] for r in rows)}..{max(r[3] for r in rows)}")
    print(f"  wrote targets -> {a.out}")


if __name__ == "__main__":
    main()
