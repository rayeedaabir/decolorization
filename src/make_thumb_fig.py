#!/usr/bin/env python3
"""Thumbnail-resolution ablation figure. Two single-axis panels:
(a) E-score vs thumbnail size, each point annotated with the selector's 1-thread latency;
(b) CCPR-iso vs thumbnail size. 128 px marked. Reads thumb_ablation_4cue.csv (4-cue; the old thumb_ablation_table.csv is the 5-cue run).

    python make_thumb_fig.py --csv thumb_ablation_table.csv --out fig_thumb.png
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

BLUE, MARK = "#0072B2", "#E69F00"


def read(path):
    t, ms, em, es, im, isd = [], [], [], [], [], []
    for r in csv.DictReader(open(path)):
        t.append(int(r["thumbnail"])); ms.append(float(r["vit_ms_1thread"]))
        em.append(float(r["heldout_E_mean"])); es.append(float(r["heldout_E_std"]))
        im.append(float(r["CCPRiso_mean"])); isd.append(float(r["CCPRiso_std"]))
    return list(map(np.array, (t, ms, em, es, im, isd)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="thumb_ablation_4cue.csv")
    ap.add_argument("--out", default="fig_thumb.png")
    a = ap.parse_args()
    t, ms, em, es, im, isd = read(a.csv)
    x = np.arange(len(t))
    mark = int(np.where(t == 128)[0][0]) if (t == 128).any() else None

    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.grid": True, "grid.color": "#d9d9d9",
                         "grid.linewidth": 0.5, "axes.axisbelow": True})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax1.errorbar(x, em, yerr=es, fmt="-o", color=BLUE, ms=4, lw=1.8, capsize=2)
    for xi, yi, mi in zip(x, em, ms):
        ax1.annotate(f"{mi:.2f} ms", (xi, yi), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=6.5, color="0.35")
    ax1.set_ylabel("E-score"); ax1.set_title("(a) contrast preservation", fontsize=8)

    ax2.errorbar(x, im, yerr=isd, fmt="-o", color=BLUE, ms=4, lw=1.8, capsize=2)
    ax2.set_ylabel("CCPR-iso"); ax2.set_title("(b) iso-luminant preservation", fontsize=8)

    for ax in (ax1, ax2):
        ax.set_xticks(x); ax.set_xticklabels(t)
        ax.set_xlabel("selector thumbnail (px)")
        if mark is not None:
            ax.axvline(mark, color=MARK, lw=1.8, ls="--", zorder=1)   # selected size, clearly visible
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if mark is not None:   # labelled callout on panel (a)
        tr = blended_transform_factory(ax1.transData, ax1.transAxes)
        ax1.text(mark, 0.04, "128 px", transform=tr, ha="center", va="bottom",
                 fontsize=7.5, fontweight="bold", color="#9a6a00",
                 bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=MARK, lw=0.6))
    fig.tight_layout()
    fig.savefig(a.out, dpi=600, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()