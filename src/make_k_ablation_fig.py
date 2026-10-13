#!/usr/bin/env python3
"""K-ablation figure: contrast preservation vs codebook size K.
Two single-axis panels (never a dual axis): (a) E-score with the codebook ceiling,
(b) CCPR-iso. K=16 marked. Reads k_ablation_4cue.csv (4-cue; the old k_ablation_table.csv is the 5-cue run).
The ceiling curve is taken from k_ceiling_full.csv (written by k_ceiling_full.py: full metric,
same held-out images as realized E). Pass --ceiling "" to fall back to the screening-metric
pool ceiling stored in the K-ablation CSV (not comparable with realized E).

    python make_k_ablation_fig.py --csv k_ablation_table.csv --out fig_k_ablation.png
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

BLUE, ORANGE, MARK = "#0072B2", "#E69F00", "#E69F00"   # Okabe-Ito, colourblind-safe


def read(path):
    K, ce, em, es, im, isd = [], [], [], [], [], []
    for r in csv.DictReader(open(path)):
        K.append(int(r["K"])); ce.append(float(r["codebook_ceiling_E"]))
        em.append(float(r["heldout_E_mean"])); es.append(float(r["heldout_E_std"]))
        im.append(float(r["CCPRiso_mean"])); isd.append(float(r["CCPRiso_std"]))
    return list(map(np.array, (K, ce, em, es, im, isd)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="k_ablation_4cue.csv")
    ap.add_argument("--ceiling", default="k_ceiling_full.csv",
                    help="full-metric held-out ceiling from k_ceiling_full.py ('' = screening pool ceiling)")
    ap.add_argument("--out", default="fig_k_ablation.png")
    a = ap.parse_args()
    K, ce, em, es, im, isd = read(a.csv)
    cs = np.zeros_like(ce)
    if a.ceiling:
        full = {int(r["K"]): (float(r["ceiling_full_heldout_mean"]), float(r["ceiling_full_heldout_std"]))
                for r in csv.DictReader(open(a.ceiling))}
        ce = np.array([full[k][0] for k in K]); cs = np.array([full[k][1] for k in K])
    x = np.arange(len(K))
    mark = int(np.where(K == 16)[0][0]) if (K == 16).any() else None

    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.grid": True, "grid.color": "#d9d9d9",
                         "grid.linewidth": 0.5, "axes.axisbelow": True})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax1.errorbar(x, ce, yerr=cs, fmt="--o", color=ORANGE, ms=4, lw=1.5, capsize=2,
                 label="codebook ceiling (held-out)")
    ax1.errorbar(x, em, yerr=es, fmt="-o", color=BLUE, ms=4, lw=1.8, capsize=2,
                 label="ViT selection (held-out)")
    ax1.set_ylabel("E-score"); ax1.set_title("(a) contrast preservation", fontsize=8)
    ax1.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=6.5, loc="lower right")

    ax2.errorbar(x, im, yerr=isd, fmt="-o", color=BLUE, ms=4, lw=1.8, capsize=2)
    ax2.set_ylabel("CCPR-iso"); ax2.set_title("(b) iso-luminant preservation", fontsize=8)

    for ax in (ax1, ax2):
        ax.set_xticks(x); ax.set_xticklabels(K)
        ax.set_xlabel("codebook size  $K$")
        if mark is not None:
            ax.axvline(mark, color=MARK, lw=1.8, ls="--", zorder=1)   # selected K, clearly visible
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if mark is not None:   # labelled callout on panel (a)
        tr = blended_transform_factory(ax1.transData, ax1.transAxes)
        ax1.text(mark, 0.97, "$K=16$", transform=tr, ha="center", va="top",
                 fontsize=7.5, fontweight="bold", color="#9a6a00",
                 bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=MARK, lw=0.6))
    fig.tight_layout()
    fig.savefig(a.out, dpi=600, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()