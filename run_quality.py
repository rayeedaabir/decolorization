"""Compute decolorization quality metrics for a folder of RGB images.

Example:
    python run_quality.py --images data/Cadik --decolorizer bt601
    python run_quality.py --images data/Color250 --decolorizer chroma_demo

Reports mean CCPR / CCFR / E-score over the images, plus the isoluminant
collapse count (a fixed synthetic test, image-independent). numpy only.
"""
import argparse
import glob
import os
import numpy as np
from PIL import Image
from decolorizers import get, REGISTRY
from metrics import contrast_metrics, isoluminant_collapse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of RGB images")
    ap.add_argument("--decolorizer", default="bt601", help="registry name or 'vit:CKPT.pt'")
    args = ap.parse_args()

    if args.decolorizer.startswith("vit2:"):
        from stage2 import load_decolorizer as load2
        dec = load2(args.decolorizer.split(":", 1)[1])
    elif args.decolorizer.startswith("vit:"):
        from fusion_vit import load_decolorizer
        dec = load_decolorizer(args.decolorizer.split(":", 1)[1])
    else:
        dec = get(args.decolorizer)
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.ppm")
    files = sorted(f for e in exts for f in glob.glob(os.path.join(args.images, e)))
    if not files:
        raise SystemExit(f"no images found in {args.images}")

    agg = {"CCPR": [], "CCFR": [], "E": []}
    for f in files:
        rgb = np.asarray(Image.open(f).convert("RGB"), np.float64) / 255.0
        m = contrast_metrics(rgb, dec(rgb))
        for k in agg:
            agg[k].append(m[k])

    print(f"decolorizer={args.decolorizer}  images={len(files)}")
    for k in agg:
        print(f"  {k:5s}: {np.mean(agg[k]):.4f}")
    print(f"  isoluminant_collapse: {isoluminant_collapse(dec)}  (lower=better)")


if __name__ == "__main__":
    main()
