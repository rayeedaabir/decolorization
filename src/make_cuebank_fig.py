#!/usr/bin/env python3
"""Cue-bank composition ablation figure (3-seed). Two single-axis bar panels:
(a) aggregate E-score, (b) CCPR-iso, mean ± s.d. over seeds, across cue subsets.
Reads ablation_table.csv from run_cue_ablation.py (columns config,E_mean,E_std,
CCPRiso_mean,CCPRiso_std); BT.709 is prepended as a fixed reference (no seeds).

    python make_cuebank_fig.py --csv ablation_table.csv --out fig_cuebank.png
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#9e9e9e"
BT709 = ("BT.709", 0.783, 0.0, 0.038, 0.0)   # fixed reference, no seed spread
LABEL = {"RGB": "RGB", "RGB+MSR": "RGB+MSR", "RGB+CHROMA": "RGB+chroma", "full": "Full"}


def read(path):
    rows = [BT709]
    for r in csv.DictReader(open(path)):
        rows.append((LABEL.get(r["configuration"], r["configuration"]),
                     float(r["E_mean"]), float(r["E_std"]),
                     float(r["CCPRiso_mean"]), float(r["CCPRiso_std"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="ablation_table.csv")
    ap.add_argument("--out", default="fig_cuebank.png")
    a = ap.parse_args()
    rows = read(a.csv)
    names = [r[0] for r in rows]
    E = [r[1] for r in rows]; Es = [r[2] for r in rows]
    I = [r[3] for r in rows]; Is = [r[4] for r in rows]
    colors = [GRAY if n == "BT.709" else (ORANGE if n == "RGB+chroma" else BLUE) for n in names]

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x = np.arange(len(names))
    for ax, vals, errs, ttl, ylab in ((ax1, E, Es, "(a) aggregate contrast (E)", "E-score"),
                                      (ax2, I, Is, "(b) iso-luminant (CCPR-iso)", "CCPR-iso")):
        ax.bar(x, vals, yerr=errs, width=0.66, color=colors, edgecolor="white",
               linewidth=1.2, capsize=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.3)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=8)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="y", color="#d9d9d9", lw=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=600, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
