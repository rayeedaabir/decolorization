#!/usr/bin/env python3
"""Pareto figure (4-cue): contrast preservation (E) vs single-thread inference cost.
Single-axis scatter, log-x for time. The proposed 4-cue model is the star. Offline
methods with no runnable implementation are omitted (they appear in Table I on quality
alone). Points are hardcoded from the locked Table I / Table VI numbers.

    python make_pareto_4cue.py --out fig_pareto.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#6f6f6f"

# label, E, ms, kind   (runnable methods only)
POINTS = [
    # E: Table I (Color250, native resolution). ms: one single-thread session at 256x256 on
    # 11 Oct 2026 -- time_4cue.py (proposed) and compare_stats.py --time (the other three).
    ("BT.709",            0.7828,  0.31, "fixed"),
    ("BT.709+stretch",    0.7956,  0.35, "fixed"),
    ("cv2decolor [9]",    0.8068, 85.60, "published"),
    ("Proposed (4-cue)",  0.8398, 12.72, "ours"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_pareto.png")
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for lab, e, ms, kind in POINTS:
        if kind == "ours":
            ax.scatter([ms], [e], s=130, marker="*", color=ORANGE, zorder=5, edgecolor="white", linewidth=0.8)
            ax.annotate(lab, (ms, e), textcoords="offset points", xytext=(-6, 8), ha="right",
                        fontsize=8, fontweight="bold", color=ORANGE)
        else:
            c = GRAY if kind == "published" else BLUE
            ax.scatter([ms], [e], s=42, color=c, zorder=4, edgecolor="white", linewidth=0.6)
            dx = (6, 4) if lab != "cv2decolor [9]" else (-6, 6)
            ha = "left" if lab != "cv2decolor [9]" else "right"
            ax.annotate(lab, (ms, e), textcoords="offset points", xytext=dx, ha=ha, fontsize=7.5, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("inference time (ms, 1 CPU thread, log)")
    ax.set_ylabel("E-score")
    ax.grid(True, which="both", color="#e2e2e2", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # annotate the speed gap
    ax.annotate("≈7× faster\nthan cv2decolor", (12.72, 0.8398), textcoords="offset points",
                xytext=(30, -28), fontsize=7, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
    fig.tight_layout()
    fig.savefig(a.out, dpi=600, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
