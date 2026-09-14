"""Figure 1 -- the teaser: a benchmark colour-vision plate rendered four ways.

Replaces the photographed Raspberry-Pi teaser (Comment 2.1) with a high-resolution,
publication-quality qualitative comparison drawn DIRECTLY from the image arrays -- no
display photography, no fingers, no background. A 2x2 grid:

    (a) input RGB           (b) BT.709 baseline
    (c) classical baseline  (d) proposed

Every grayscale panel is annotated with its E-score, computed on the same array by
metrics.contrast_metrics -- so the number under each panel is the real contrast-
preservation score for THIS image, not a global average.

    python make_teaser.py --image data/Color250_rgb/<ishihara>.png --ckpt fusion_vit.pt
    python make_teaser.py --image <img> --proposed vit:fusion_vit.pt:codebook.npz \
                          --classical cv2decolor --out figs/teaser

The Raspberry-Pi live-deployment photo moves to Section 3.6 (Hardware/Throughput); this
figure is the benchmark comparison only.

Notes on the classical baseline: OpenCV's cv2.decolor implements Lu et al.'s real-time
contrast-preserving method (SIGGRAPH Asia 2012 Technical Briefs), cited as [8]. Label it
"(c) Lu et al. [8]" (default) or "(c) cv2.decolor (Lu 2012)" -- both are accurate; pick
whichever matches how [8] is described in the running text.
"""
import argparse, os
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from decolorizers import get_any
try:
    from metrics import contrast_metrics, isoluminant_ccpr
except Exception:
    contrast_metrics = isoluminant_ccpr = None

ORANGE, INK, MUTE = "#eb6834", "#1a1a1a", "#6a6a6a"


def load(path, size):
    im = Image.open(path).convert("RGB")
    if size and size > 0:
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, np.float64) / 255.0


def metric_label(rgb, gray, which):
    """Return the annotation string for one grayscale panel.

    which='ccpr_iso' (default) reports the iso-luminant metric, which is what this
    figure is actually about -- aggregate E barely moves on a plate whose collapse is
    confined to a few iso-luminant pairs. 'e' reports aggregate E; 'both' reports both.
    """
    if contrast_metrics is None:
        return None
    parts = []
    try:
        if which in ("e", "both"):
            parts.append(f"E = {contrast_metrics(rgb, gray)['E']:.3f}")
        if which in ("ccpr_iso", "both"):
            iso = isoluminant_ccpr(rgb, gray)["CCPR_iso"]
            parts.append(("CCPR-iso" if which == "ccpr_iso" else "iso") + f" = {iso:.3f}")
    except Exception as e:
        print(f"[metric unavailable: {type(e).__name__}]")
        return None
    return "   ".join(parts) if parts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="a benchmark colour plate (Color250 / Ishihara)")
    ap.add_argument("--baseline", default="bt709", help="fixed-standard baseline decolorizer")
    ap.add_argument("--classical", default="cv2decolor",
                    help="strong classical baseline (default cv2decolor = Lu 2012 [8])")
    ap.add_argument("--proposed", default="", help="proposed decolorizer spec, e.g. vit:fusion_vit.pt")
    ap.add_argument("--ckpt", default="fusion_vit.pt", help="used to build --proposed if not given")
    ap.add_argument("--codebook", default="codebook.npz")
    ap.add_argument("--classical-label", default="Lu et al. [8]")
    ap.add_argument("--metric", default="ccpr_iso", choices=["ccpr_iso", "e", "both"],
                    help="annotation under each grayscale panel (default ccpr_iso: the "
                         "iso-luminant metric this figure is about)")
    ap.add_argument("--out", default="figs/teaser")
    ap.add_argument("--size", type=int, default=0, help="0 = native resolution (crispest)")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--wide", action="store_true", help="two-column width (7.0in) instead of single (3.45in)")
    ap.add_argument("--row", action="store_true", help="1x4 horizontal strip (full-width, short) — best as a top-of-page teaser")
    a = ap.parse_args()

    proposed = a.proposed or f"vit:{a.ckpt}:{a.codebook}"
    rgb = load(a.image, a.size)

    # (title, array, annotation) for each of the four panels
    panels = [("(a) input", rgb, None)]
    for tag, spec in [("(b) BT.709", a.baseline),
                      (f"(c) {a.classical_label}", a.classical),
                      ("(d) proposed", proposed)]:
        dec = get_any(spec)
        gray = np.asarray(dec(rgb), np.float64)
        panels.append((tag, gray, metric_label(rgb, gray, a.metric)))

    if a.row:
        fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.15))
    else:
        W = 7.0 if a.wide else 3.45
        fig, axes = plt.subplots(2, 2, figsize=(W, W * 1.06))
    for ax, (tag, img, lab) in zip(axes.ravel(), panels):
        proposed_panel = tag.startswith("(d)")
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        edge = ORANGE if proposed_panel else "#bbbbbb"
        for s in ax.spines.values():
            s.set_edgecolor(edge); s.set_linewidth(1.8 if proposed_panel else 0.8)
        ax.set_title(tag, fontsize=8.5, color=ORANGE if proposed_panel else INK,
                     fontweight="bold" if proposed_panel else "normal", pad=3)
        if lab is not None:
            ax.text(0.5, -0.055, lab, transform=ax.transAxes,
                    ha="center", va="top", fontsize=8,
                    color=ORANGE if proposed_panel else MUTE,
                    fontweight="bold" if proposed_panel else "normal")

    if a.row:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.15, wspace=0.06)
    else:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.06, wspace=0.06, hspace=0.22)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=a.dpi, bbox_inches="tight", facecolor="white")
    es = "  |  ".join(f"{t.split()[0]} {lab}" for t, _, lab in panels if lab is not None)
    print(f"wrote {a.out}.png and .pdf   ({a.dpi} dpi)   {es}")


if __name__ == "__main__":
    main()
