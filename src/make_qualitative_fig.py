#!/usr/bin/env python3
"""Qualitative comparison grid (Fig 3, wide / double-column float).
Rows = sample images; columns = the input RGB then each method's grayscale, proposed last.

Methods are GENERATED on the fly from the RGB inputs with decolorizers.get_any (the same code
path make_teaser.py uses), so you only need the input folder plus the 4-cue checkpoint and
codebook -- no pre-exported per-method folders. Released perceptual outputs you already have as
images can be added as extra columns from a folder.

Each column is given as  --col "Label=KIND:VALUE"  where KIND is:
    gen:<spec>   generate with decolorizers.get_any   (bt709 | cv2decolor | vit:ckpt:codebook)
    dir:<folder> load a pre-exported grayscale image, matched to the input by filename stem

Example (runs with just data/Color250_rgb + the 4-cue artifacts):

    python make_qualitative_fig.py \
        --input data/Color250_rgb \
        --images 007.png 041.png 119.png 203.png \
        --col "BT.709=gen:bt709" \
        --col "cv2decolor=gen:cv2decolor" \
        --col "Proposed (4-cue)=gen:vit:fusion_vit_rgbchroma.pt:codebook_rgbchroma.npz" \
        --out fig_qualitative.png --dpi 600

Add a released perceptual column you have on disk, e.g.:
        --col "Grundland=dir:data/Color250/grundland"

Run from the harness folder (needs decolorizers.py / cues.py / fusion_vit.py on the path).
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ORANGE = "#E69F00"


def load_rgb(path, size):
    if not path or not os.path.exists(path):
        return None
    return np.asarray(Image.open(path).convert("RGB").resize((size, size)), np.float64) / 255.0


def find_in_dir(folder, name):
    """Match the input filename in a folder, tolerating extension/case changes."""
    if not folder or not os.path.isdir(folder):
        return None
    stem = os.path.splitext(name)[0].lower()
    for f in os.listdir(folder):
        if os.path.splitext(f)[0].lower() == stem:
            return os.path.join(folder, f)
    cand = os.path.join(folder, name)
    return cand if os.path.exists(cand) else None


def to_gray(arr):
    """Coerce a decolorizer/loaded output to a 2-D grayscale array in [0,1]."""
    a = np.asarray(arr, np.float64)
    if a.ndim == 3:
        a = a[..., 0]
    if a.max() > 1.5:
        a = a / 255.0
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of RGB input images")
    ap.add_argument("--images", nargs="+", required=True, help="input filenames -> rows")
    ap.add_argument("--col", action="append", default=[], help='"Label=gen:<spec>" or "Label=dir:<folder>"')
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="fig_qualitative.png")
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()

    # parse columns
    cols = [("Input", None, None)]   # (label, kind, value)
    decs = {}
    for c in a.col:
        if "=" not in c:
            raise SystemExit(f'--col must be "Label=gen:<spec>" or "Label=dir:<folder>", got {c!r}')
        lab, rhs = c.split("=", 1)
        if rhs.startswith("gen:"):
            kind, val = "gen", rhs[4:]
        elif rhs.startswith("dir:"):
            kind, val = "dir", rhs[4:]
        else:
            raise SystemExit(f'column value must start with gen: or dir:, got {rhs!r}')
        cols.append((lab.strip(), kind, val))

    # build generators lazily (import only if a gen: column is present)
    if any(k == "gen" for _, k, _ in cols):
        from decolorizers import get_any
        for lab, kind, val in cols:
            if kind == "gen":
                decs[lab] = get_any(val)

    nrow, ncol = len(a.images), len(cols)
    plt.rcParams.update({"font.family": "serif"})
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.55 * ncol, 1.55 * nrow))
    axes = np.atleast_2d(axes).reshape(nrow, ncol)

    for r, name in enumerate(a.images):
        rgb = load_rgb(os.path.join(a.input, name), a.size)
        for c, (lab, kind, val) in enumerate(cols):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            img, is_rgb = None, False
            if kind is None:                          # input
                img, is_rgb = rgb, True
            elif kind == "gen" and rgb is not None:
                img = to_gray(decs[lab](rgb))
            elif kind == "dir":
                p = find_in_dir(val, name)
                if p is None:
                    print(f"[miss] {lab}: no file for {name} in {val}")
                else:
                    img = to_gray(load_rgb(p, a.size))
            if img is not None:
                ax.imshow(img, cmap=None if is_rgb else "gray",
                          vmin=None if is_rgb else 0, vmax=None if is_rgb else 1)
            if r == 0:
                ax.set_title(lab, fontsize=8.5,
                             color=ORANGE if lab.startswith("Proposed") else "black",
                             fontweight="bold" if lab.startswith("Proposed") else "normal")
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.005, right=0.995, top=0.94, bottom=0.005)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    # also emit a vector PDF (ideal for IEEE line-art/figures)
    if a.out.lower().endswith(".png"):
        fig.savefig(a.out[:-4] + ".pdf", bbox_inches="tight")
    print("wrote", a.out, f"({nrow}x{ncol}, dpi={a.dpi}) + .pdf")


if __name__ == "__main__":
    main()
