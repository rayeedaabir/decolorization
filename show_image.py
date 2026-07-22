"""Visualise decolorization on a SINGLE image (great for the professor).

Method comparison (default) - original + each method's grayscale, labelled with
its E-score on this image:
    python show_image.py --image data/Cadik_rgb/56.png --out figs/compare.png

Cue-bank internals - the cues the model combines:
    python show_image.py --image data/Cadik_rgb/56.png --internals --out figs/internals.png

--image accepts ANY RGB image (a dataset image or your own photo).
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from decolorizers import get
from cues import cue_bank, fuse_cues, BT709, CUE_NAMES
from metrics import contrast_metrics


def load(path, size=256):
    return np.asarray(Image.open(path).convert("RGB").resize((size, size)), np.float64) / 255


def comparison(rgb, methods, out, title):
    n = len(methods) + 1
    fig, ax = plt.subplots(1, n, figsize=(2.9 * n, 3.3))
    ax[0].imshow(rgb); ax[0].set_title("Original RGB", fontsize=11); ax[0].axis("off")
    for i, m in enumerate(methods, 1):
        g = get_dec(m)(rgb)
        e = contrast_metrics(rgb, g)["E"]
        ax[i].imshow(g, cmap="gray", vmin=0, vmax=1)
        ax[i].set_title(f"{m}\nE={e:.3f}", fontsize=11); ax[i].axis("off")
    fig.suptitle(title, fontweight="bold"); fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130); print("wrote", out)


def get_dec(name):
    if name.startswith("vit2:"):
        from stage2 import load_decolorizer as l2
        return l2(name.split(":", 1)[1])
    if name.startswith("vit:"):
        from fusion_vit import load_decolorizer as l1
        return l1(name.split(":", 1)[1])
    return get(name)


def internals(rgb, out, title):
    """Show the cue bank the model combines, plus the fixed-standard fusion."""
    c = cue_bank(rgb)
    panels = [("Original RGB", rgb, None)] + \
             [(CUE_NAMES[i], c[i], "gray") for i in range(len(CUE_NAMES))] + \
             [("fused (BT.709)", fuse_cues(c, BT709), "gray")]
    fig, ax = plt.subplots(1, len(panels), figsize=(2.5 * len(panels), 3.1))
    for a, (t, img, cm) in zip(ax, panels):
        if cm: a.imshow(img, cmap=cm, vmin=0, vmax=1)
        else:  a.imshow(img)
        a.set_title(t, fontsize=10); a.axis("off")
    fig.suptitle(title, fontweight="bold"); fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130); print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--methods", default="bt601,bt709", help="registry names, or vit:CKPT / vit2:CKPT")
    ap.add_argument("--internals", action="store_true")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rgb = load(a.image, a.size); name = os.path.basename(a.image)
    if a.internals:
        internals(rgb, a.out or "figs/internals.png", f"Cue bank - {name}")
    else:
        comparison(rgb, a.methods.split(","), a.out or "figs/compare.png",
                   f"Decolorization comparison - {name}")


if __name__ == "__main__":
    main()
