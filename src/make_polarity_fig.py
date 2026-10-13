#!/usr/bin/env python3
"""Fig. 7 (corrected): chroma-polarity ablation on the 4-cue base.

Two panels, three bars each (4-cue, 5-cue = +chroma-complement, 6-cue = +complement+MSR),
mean +/- s.d. over seeds, read straight from polarity_results.csv so the figure can never
drift from the numbers. Neither extension reliably improves on the 4-cue bank.

    python make_polarity_fig.py --csv polarity_results.csv --out fig_polarity.png
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORANGE, BLUE, GRAY = "#E69F00", "#0072B2", "#6f6f6f"
LABELS = {"4cue": "4-cue", "5cue_+1-C": "5-cue\n(+1−C)", "6cue_+1-C+MSR": "6-cue\n(+1−C, MSR)"}
COLORS = {"4cue": ORANGE, "5cue_+1-C": BLUE, "6cue_+1-C+MSR": GRAY}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="polarity_results.csv")
    ap.add_argument("--out", default="fig_polarity.png")
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()

    rows = {r["arm"]: r for r in csv.DictReader(open(a.csv))}
    arms = ["4cue", "5cue_+1-C", "6cue_+1-C+MSR"]
    E = [float(rows[k]["heldout_E_mean"]) for k in arms]
    Es = [float(rows[k]["heldout_E_std"]) for k in arms]
    I = [float(rows[k]["CCPRiso_mean"]) for k in arms]
    Is = [float(rows[k]["CCPRiso_std"]) for k in arms]
    cols = [COLORS[k] for k in arms]
    labs = [LABELS[k] for k in arms]
    x = np.arange(3)

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, (axE, axI) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for ax, vals, errs, ylab, title in [
        (axE, E, Es, "held-out E", "(a) E-score"),
        (axI, I, Is, "CCPR-iso", "(b) CCPR-iso"),
    ]:
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=cols, edgecolor="white",
                      linewidth=0.8, width=0.62, error_kw=dict(lw=1.0, ecolor="#444444"))
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=9, pad=4)
        ax.grid(True, axis="y", color="#e8e8e8", lw=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        lo = min(v - e for v, e in zip(vals, errs)); hi = max(v + e for v, e in zip(vals, errs))
        pad = (hi - lo) * 0.35 + 1e-6
        ax.set_ylim(lo - pad, hi + pad)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + e + pad * 0.12, f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("Chroma-polarity ablation on Color250 (mean ± s.d., 3 seeds) — no reliable gain over 4-cue",
                 fontsize=8.5, y=1.02)
    fig.tight_layout()
    for ext in dict.fromkeys([a.out.rsplit(".", 1)[-1], "pdf"]):
        out = a.out if a.out.endswith(ext) else a.out.rsplit(".", 1)[0] + "." + ext
        fig.savefig(out, dpi=a.dpi, bbox_inches="tight", facecolor="white")
        print("wrote", out)


if __name__ == "__main__":
    main()