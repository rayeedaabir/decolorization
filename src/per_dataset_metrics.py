#!/usr/bin/env python3
"""Per-dataset contrast metrics for the appendix "full-metrics" table.

Runs one or more decolorizers over every image in a dataset folder and reports the
SAME contrast metrics used in Table I -- CCPR, CCFR, E, and CCPR-iso -- as a per-method
aggregate (mean + 95% bootstrap CI), plus the relative E-gain over BT.709 that Table II
uses. Optionally dumps a per-image CSV so the CIs are reproducible.

It reuses the project's own metric and decolorizer code (the same imports make_teaser.py
uses), so the numbers are directly comparable to Table I / Table II:

    from decolorizers import get_any
    from metrics       import contrast_metrics, isoluminant_ccpr

Run it once per dataset, e.g. (adjust paths/checkpoint/codebook to your tree):

    python per_dataset_metrics.py --dataset Cadik     --data-dir data/Cadik_rgb \
        --ckpt fusion_vit_rgbchroma.pt --codebook codebook_rgbchroma.npz \
        --methods proposed,bt709 --per-image per_image_cadik.csv --out perds_cadik.csv
    python per_dataset_metrics.py --dataset Color250  --data-dir data/Color250_rgb ...
    python per_dataset_metrics.py --dataset "UC Merced" --data-dir data/UCMerced ...
    python per_dataset_metrics.py --dataset AID       --data-dir data/AID ...

Then send me the perds_*.csv files and I'll assemble the appendix table.

Notes
-----
* --data-dir is searched recursively, so class-subfolder datasets (UC Merced, AID) work
  as-is. Use --limit N for a quick smoke test before the full run (AID is 10k images).
* Metrics are computed on the image resized to --size (256 by default), matching how the
  cues are computed in the paper.
* If your metrics.contrast_metrics returns different key names, set them with
  --ccpr-key / --ccfr-key / --e-key (the script prints the available keys on first image).
* Add cv2decolor to --methods for a third row, but it is slow on AID; proposed,bt709 is
  the comprehensive + reproducible default (BT.709 is the reference for the E-gain column).
"""
import argparse, os, glob, csv, sys
import numpy as np
from PIL import Image

try:
    from decolorizers import get_any
    from metrics import contrast_metrics, isoluminant_ccpr
except Exception as e:
    sys.exit(f"[import error] run this from the harness dir that has decolorizers.py + "
             f"metrics.py (the ones make_teaser.py uses): {e}")

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
METHOD_SPECS = {"proposed": None, "bt709": "bt709", "cv2decolor": "cv2decolor"}


def load_rgb(path, size):
    im = Image.open(path).convert("RGB")
    if size and size > 0:
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, np.float64) / 255.0


def list_images(d, limit=0):
    files = sorted(os.path.join(r, f)
                   for r, _, fs in os.walk(d) for f in fs
                   if f.lower().endswith(IMG_EXTS))
    return files[:limit] if limit else files


def bootstrap_ci(x, n=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="label for the output rows, e.g. Cadik")
    ap.add_argument("--data-dir", required=True, help="folder of RGB images (searched recursively)")
    ap.add_argument("--ckpt", default="fusion_vit_rgbchroma.pt")
    ap.add_argument("--codebook", default="codebook_rgbchroma.npz")
    ap.add_argument("--methods", default="proposed,bt709",
                    help="comma list from: proposed, bt709, cv2decolor")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N images (smoke test); 0 = all")
    ap.add_argument("--ccpr-key", default="CCPR")
    ap.add_argument("--ccfr-key", default="CCFR")
    ap.add_argument("--e-key", default="E")
    ap.add_argument("--per-image", default="", help="optional per-image CSV path")
    ap.add_argument("--out", default="per_dataset_metrics.csv")
    a = ap.parse_args()

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHOD_SPECS:
            sys.exit(f"unknown method {m!r}; choose from {list(METHOD_SPECS)}")
    specs = dict(METHOD_SPECS, proposed=f"vit:{a.ckpt}:{a.codebook}")
    decs = {m: get_any(specs[m]) for m in methods}

    imgs = list_images(a.data_dir, a.limit)
    if not imgs:
        sys.exit(f"no images found under {a.data_dir}")
    print(f"[{a.dataset}] {len(imgs)} images x {len(methods)} methods @ {a.size}px")

    per_method = {m: {"CCPR": [], "CCFR": [], "E": [], "CCPR_iso": []} for m in methods}
    rows_per_image = []
    for i, p in enumerate(imgs):
        rgb = load_rgb(p, a.size)
        for m in methods:
            gray = np.asarray(decs[m](rgb), np.float64)
            cm = contrast_metrics(rgb, gray)
            if i == 0 and m == methods[0]:
                print("  contrast_metrics keys:", sorted(cm.keys()))
            vals = {
                "CCPR": cm.get(a.ccpr_key),
                "CCFR": cm.get(a.ccfr_key),
                "E": cm.get(a.e_key),
                "CCPR_iso": isoluminant_ccpr(rgb, gray).get("CCPR_iso"),
            }
            for k, v in vals.items():
                if v is not None:
                    per_method[m][k].append(float(v))
            rows_per_image.append({"dataset": a.dataset, "method": m,
                                   "image": os.path.relpath(p, a.data_dir), **vals})
        if (i + 1) % 250 == 0:
            print(f"  ...{i + 1}/{len(imgs)}")

    e_bt709 = np.mean(per_method["bt709"]["E"]) if "bt709" in per_method and per_method["bt709"]["E"] else None

    summary = []
    for m in methods:
        row = {"dataset": a.dataset, "method": m, "n": len(per_method[m]["E"])}
        for k in ("CCPR", "CCFR", "E", "CCPR_iso"):
            mean, lo, hi = bootstrap_ci(per_method[m][k])
            row[k] = round(mean, 4)
            row[f"{k}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        if e_bt709:
            row["dE_vs_BT709_%"] = round(100.0 * (row["E"] - e_bt709) / e_bt709, 2)
        summary.append(row)

    cols = ["dataset", "method", "n", "CCPR", "CCPR_ci95", "CCFR", "CCFR_ci95",
            "E", "E_ci95", "CCPR_iso", "CCPR_iso_ci95", "dE_vs_BT709_%"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print("\nwrote", a.out)
    for r in summary:
        print(" ", {k: r.get(k) for k in ("method", "n", "E", "CCPR_iso", "dE_vs_BT709_%")})

    if a.per_image:
        with open(a.per_image, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "method", "image",
                                              "CCPR", "CCFR", "E", "CCPR_iso"])
            w.writeheader()
            w.writerows(rows_per_image)
        print("wrote", a.per_image, f"({len(rows_per_image)} rows)")


if __name__ == "__main__":
    main()
