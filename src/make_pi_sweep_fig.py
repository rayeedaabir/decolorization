#!/usr/bin/env python3
"""Appendix figure: on-device latency and energy vs. input resolution (Raspberry Pi 5).

Visualizes Table A1 (the full 8-resolution sweep). Two panels, log-y so the three
methods share a scale despite spanning four orders of magnitude:

    (a) latency  (ms/frame)      vs resolution
    (b) energy   (mJ/frame)      vs resolution

for all three measured methods (Proposed 4-cue, cv2decolor, BT.709). The 256 px
deployment size is marked, and the 33.3 ms / 30 fps real-time line is drawn in (a).
The point of the figure is what Table A1 states in numbers: from 96 px upward the
proposed method is faster and lower-energy than the contrast-preserving baseline
(cv2decolor is faster only at 32 px, where the selector's fixed cost dominates), and the
proposed method's latency falls from 320 px to 416 px because the on-device build's MSR
cue switches to a downsampled path for its largest surround once the shorter side
reaches 400 px.

Numbers are hardcoded from the locked Table A1 (60 s sustained per method, CPU only with
the libraries' default threading, synthetic frames, active cooler, throttle flag 0x0
throughout). Keep them in sync with the table.

    python make_pi_sweep_fig.py --out fig_pi_sweep.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito (matches the other figure scripts)
ORANGE, BLUE, GRAY = "#E69F00", "#0072B2", "#6f6f6f"

SIZES = np.array([32, 96, 128, 160, 224, 256, 320, 416])

# ms/frame and mJ/frame, from Table A1 (Proposed 4-cue, cv2decolor, BT.709)
MS = {
    "Proposed (4-cue)": [12.59, 22.62, 31.53, 37.87, 68.13, 82.22, 120.45, 113.22],
    "cv2decolor":       [6.12, 59.09, 140.80, 173.96, 339.42, 577.48, 736.71, 1056.71],
    "BT.709":           [0.03, 0.14, 0.24, 0.38, 0.78, 1.08, 1.78, 3.28],
}
MJ = {
    "Proposed (4-cue)": [51.2, 93.0, 126.9, 154.2, 280.0, 310.4, 387.8, 367.3],
    "cv2decolor":       [21.2, 202.5, 486.9, 603.6, 1227.9, 2071.2, 2622.8, 3722.8],
    "BT.709":           [0.1, 0.5, 0.9, 1.4, 2.7, 3.7, 6.1, 11.0],
}
STYLE = {  # color, marker, linewidth, z
    "Proposed (4-cue)": (ORANGE, "*", 2.0, 5, 10),
    "cv2decolor":       (GRAY,   "o", 1.4, 3, 5.5),
    "BT.709":           (BLUE,   "s", 1.4, 3, 4.5),
}
DEPLOY = 256  # the size reported in the main-text Table V


X = np.arange(len(SIZES))          # evenly spaced categorical x (no label crowding)
DEPLOY_X = int(np.where(SIZES == DEPLOY)[0][0])


def _plot(ax, data, ylabel, title, realtime_ms=None):
    for name in ("cv2decolor", "BT.709", "Proposed (4-cue)"):  # proposed last = on top
        c, mk, lw, z, ms = STYLE[name]
        y = data[name]
        ax.plot(X, y, color=c, lw=lw, marker=mk, markersize=ms, zorder=z,
                markeredgecolor="white", markeredgewidth=0.6, label=name)
    if realtime_ms is not None:
        ax.axhline(realtime_ms, color="#b0b0b0", lw=0.9, ls=":", zorder=1)
        ax.text(X[4] - 0.3, realtime_ms * 0.80, "30 fps (33.3 ms)", fontsize=6.5, va="top", ha="left",
                color="#8a8a8a")
    # mark the deployment resolution
    ax.axvline(DEPLOY_X, color="#d9d9d9", lw=0.8, zorder=0)
    ax.set_yscale("log")
    ax.set_xticks(X)
    ax.set_xticklabels([str(s) for s in SIZES], fontsize=7)
    ax.set_xlabel("Input resolution (px)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, pad=4)
    ax.grid(True, which="both", color="#e8e8e8", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_pi_sweep.png")
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    _plot(axL, MS, "Latency (ms/frame, log)", "(a) Latency vs resolution",
          realtime_ms=33.3)
    _plot(axR, MJ, "Energy (mJ/frame, log)", "(b) Energy per frame vs resolution")

    # annotate the 320 -> 416 px latency drop (MSR fast path at >= 400 px; idx 7 = 416 px)
    axL.annotate("MSR fast path\n(\u2265400 px)",
                 xy=(7, 113.22), xytext=(5.3, 3.2), fontsize=6.5, color=ORANGE,
                 ha="center", arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))

    handles, labels = axL.get_legend_handles_labels()
    # legend order: Proposed, cv2decolor, BT.709
    order = [labels.index(n) for n in ("Proposed (4-cue)", "cv2decolor", "BT.709")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=3, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in dict.fromkeys([a.out.rsplit(".", 1)[-1], "pdf"]):
        out = a.out if a.out.endswith(ext) else a.out.rsplit(".", 1)[0] + "." + ext
        fig.savefig(out, dpi=a.dpi, bbox_inches="tight", facecolor="white")
        print("wrote", out)


if __name__ == "__main__":
    main()
