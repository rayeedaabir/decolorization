"""The top-level system diagram: cue bank through fusion and the selector, in one figure.

A clean, self-contained schematic (Comment 2.2/2.3): standard notation matching the
main text, no floating commentary, explanations moved to the caption.

    python make_system_figure.py --image data/Color250_rgb/17.png --ckpt fusion_vit.pt

Every panel is a real array: the cue maps come from cues.cue_bank, the weights from the
trained codebook, the output from the actual fusion. With --ckpt the selected prototype
is the one the trained network really picks for that image.

Writes figs/system_overview.{png,pdf} at two-column width, 400 dpi.

Notation used (define these once in the caption, not on the canvas):
    I  in R^{HxW x 3}     input image
    c_1..c_5              cue bank  (c1-c3 sRGB channels, c4 multi-scale Retinex,
                                     c5 dominant CIELAB opponent-chroma axis)
    128 x 128 x 3         downsampled path into the selector
    ViT selector          3 blocks, 155k parameters
    p in R^16             logits over the K=16 codebook;  k = argmax p
    w_k in Delta^4        the selected convex weighting (5 non-negative weights, sum 1)
    g = sum_i w_i c_i     grayscale output
"""
import argparse, os
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cues import cue_bank, fuse_cues, BT709

BLUE, ORANGE, INK, MUTE = "#2a78d6", "#eb6834", "#1a1a1a", "#8a8a8a"


def arrow(fig, x0, y0, x1, y1, color=MUTE, lw=1.1, style="-|>"):
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                                   arrowstyle=style, mutation_scale=9,
                                   color=color, lw=lw, shrinkA=0, shrinkB=0,
                                   joinstyle="miter"))


def box(fig, x, y, w, h, label, sub="", color=BLUE, fs=6.5):
    fig.add_artist(FancyBboxPatch((x, y), w, h, transform=fig.transFigure,
                                  boxstyle="round,pad=0.004,rounding_size=0.008",
                                  fc="white", ec=color, lw=1.1, zorder=2))
    fig.text(x + w / 2, y + (h * 0.63 if sub else h / 2), label, ha="center",
             va="center", fontsize=fs, color=INK, fontweight="bold", zorder=3)
    if sub:
        fig.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                 fontsize=fs - 1.5, color=MUTE, zorder=3)


def panel(fig, x, y, w, h, img, title, cmap=None, edge=MUTE, lw=0.8, tfs=6):
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(edge); s.set_linewidth(lw)
    if title:
        ax.set_title(title, fontsize=tfs, color=INK, pad=2.5)
    return ax


def render(fig, rgb, cues, gray, thumb, logits, protos, j, w):
    """Draw the whole schematic from already-computed real arrays."""
    K = len(protos)

    # ---- row 1: the inference path ----------------------------------------
    yt, ph = 0.545, 0.30
    panel(fig, 0.030, yt, 0.115, ph, rgb, "input  $I$")
    fig.text(0.0875, yt - 0.052, r"$I \in \mathbb{R}^{H\times W\times 3}$",
             ha="center", va="top", fontsize=6.0, color=MUTE)

    cx, cw, gap = 0.205, 0.086, 0.011
    for i in range(5):
        panel(fig, cx + i * (cw + gap), yt, cw, ph, cues[i], f"$c_{{{i+1}}}$",
              cmap="gray")
    fig.text(cx + (5 * cw + 4 * gap) / 2, yt + ph + 0.070,
             r"cue bank  $c_1,\dots,c_5$", ha="center",
             fontsize=6.8, color=INK, fontweight="bold")

    fx = cx + 5 * cw + 4 * gap + 0.032
    box(fig, fx, yt + ph / 2 - 0.050, 0.088, 0.10,
        r"$g=\sum_{i=1}^{5} w_i c_i$", "", color=ORANGE, fs=7.5)
    panel(fig, fx + 0.118, yt, 0.115, ph, gray, "output  $g$", cmap="gray",
          edge=ORANGE, lw=1.3)

    box(fig, fx + 0.258, yt + ph / 2 - 0.045, 0.098, 0.09, "downstream",
        "classify / segment", color=BLUE)

    arrow(fig, 0.147, yt + ph / 2, 0.203, yt + ph / 2)
    arrow(fig, cx + 5 * cw + 4 * gap + 0.004, yt + ph / 2, fx - 0.003, yt + ph / 2)
    arrow(fig, fx + 0.090, yt + ph / 2, fx + 0.116, yt + ph / 2, color=ORANGE)
    arrow(fig, fx + 0.235, yt + ph / 2, fx + 0.256, yt + ph / 2)

    # ---- row 2: the selector ----------------------------------------------
    yb = 0.135
    panel(fig, 0.030, yb, 0.075, 0.195, thumb, "", edge=BLUE)
    fig.text(0.0675, yb - 0.030, r"$128\times128\times3$",
             ha="center", va="top", fontsize=5.8, color=MUTE)

    box(fig, 0.140, yb + 0.045, 0.130, 0.105, "ViT selector",
        "3 blocks · 155k params", color=BLUE)

    # logits strip: p in R^16 over the codebook, argmax highlighted
    sx, sw = 0.310, 0.185
    ax = fig.add_axes([sx, yb + 0.045, sw, 0.105])
    vals = (logits - logits.min()) / (np.ptp(logits) + 1e-9)
    ax.bar(range(K), vals, color=[ORANGE if i == j else "#c9d6e8" for i in range(K)],
           width=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(r"$p \in \mathbb{R}^{%d}$   (scores over the codebook)" % K,
                 fontsize=6, color=INK, pad=2)

    wx = sx + sw + 0.045
    wtxt = "  ".join(f"{v:.2f}" for v in w)
    box(fig, wx, yb + 0.045, 0.155, 0.105, r"$w_k \in \Delta^4$",
        f"$k=\\arg\\max\\,p$   [{wtxt}]", color=ORANGE, fs=6.5)

    arrow(fig, 0.107, yb + 0.0975, 0.138, yb + 0.0975, color=BLUE)
    arrow(fig, 0.272, yb + 0.0975, 0.308, yb + 0.0975, color=BLUE)
    arrow(fig, sx + sw + 0.004, yb + 0.0975, wx - 0.003, yb + 0.0975, color=BLUE)
    # selected weighting feeds the fusion block
    arrow(fig, wx + 0.078, yb + 0.152, fx + 0.044, yt + ph / 2 - 0.052,
          color=ORANGE, lw=1.3)
    # input feeds the thumbnail (downsample)
    arrow(fig, 0.0675, yt - 0.075, 0.0675, yb + 0.198, color=BLUE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default="", required=False,
                    help="fusion_vit.pt -- REQUIRED; the figure is fabricated without it")
    ap.add_argument("--codebook", default="codebook.npz")
    ap.add_argument("--out", default="figs/system_overview")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()

    rgb = np.asarray(Image.open(a.image).convert("RGB").resize((a.size, a.size)),
                     np.float64) / 255.0
    cues = cue_bank(rgb)                       # (5, H, W)

    if not a.ckpt:
        raise SystemExit(
            "--ckpt is required.\n"
            "Without it this script cannot know which prototype the network selects, so the\n"
            "score panel would show an invented ramp and the weights would default to\n"
            "BT.709 -- a figure that looks real and is not. Re-run as:\n"
            "    python make_system_figure.py --image <img> --ckpt fusion_vit.pt")

    protos = np.load(a.codebook)["protos"]
    from fusion_vit import FusionSelector, load_state_dict
    import torch
    m = FusionSelector(len(protos))
    m.load_state_dict(load_state_dict(a.ckpt, "cpu")); m.eval()
    small = np.asarray(Image.open(a.image).convert("RGB").resize((128, 128)),
                       np.float32) / 255
    with torch.no_grad():
        logits = m(torch.from_numpy(small).permute(2, 0, 1)[None]).numpy()[0]
    j = int(np.argmax(logits))
    print(f"selector chose prototype P{j}")
    w = protos[j]
    gray = fuse_cues(cues, w)
    thumb = small.astype(np.float64)

    W = 7.0
    fig = plt.figure(figsize=(W, W * 0.42))
    render(fig, rgb, cues, gray, thumb, logits, protos, j, w)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=a.dpi, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}.png and .pdf   ({a.dpi} dpi)")


if __name__ == "__main__":
    main()
