"""The quality-vs-cost figure: the paper's central claim in one picture.

Ranking second on E-score reads as a loss only while cost is invisible. Plotted
against latency the same numbers say something else: the published methods that
match our quality solve a per-image optimization, and the published methods as
cheap as we are score well below us. This draws that plane and marks the Pareto
frontier of the same-hardware points.

EDIT THE NUMBERS BELOW, then:
    python make_pareto.py              # single-column (3.45in) -- default
    python make_pareto.py --wide       # two-column (7.0in)
Writes figs/pareto_quality_cost.{png,pdf} at 400 dpi.

MEASUREMENT PROTOCOL (state this in the caption, not on the canvas):
  * Desktop points are measured on one x86 CPU thread at 256x256, cue bank
    included, via  python model_stats.py --threads 1 --size 256 --methods ...
  * The Raspberry Pi 5 point is an on-device measurement on an ARM Cortex-A76.
    It is a SEPARATE hardware series (open star): same method and same E-score
    as the desktop point, shown only to locate the deployed operating point --
    it is NOT plotted for same-hardware comparison and is excluded from the
    Pareto frontier, which is computed over the desktop-hardware points alone.
  * Where a method's own code was not run, set measured=False; those points are
    drawn hollow. Offline optimization methods with no runnable implementation
    (Gooch, Smith, Grundland & Dodgson, Lu 2014, CIE Y) are omitted rather than
    plotted from author-reported times on unknown hardware -- say so in the
    caption (Comment 2.4d).
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------- fill these in ----------------------------------
# name: (E_score, ms_per_image, kind, measured_by_us?, platform)
#   kind     : "ours" | "opt" (per-image optimization) | "fixed" (fixed projection)
#   platform : "desktop" (x86, 1 thread, 256x256) | "pi" (ARM Cortex-A76, on-device)
POINTS = {
    "BT.709":         (0.7828, 0.27,  "fixed", True, "desktop"),
    "BT.709+stretch": (0.7956, 0.31,  "fixed", True, "desktop"),
    "Lu 2012 (OpenCV)": (0.8068, 83.95, "opt",  True, "desktop"),
    "Ours (desktop)": (0.8305, 26.21, "ours",  True, "desktop"),
    "Ours (Pi 5)":    (0.8305, 77.30, "ours",  True, "pi"),
}
# nudge any label that would collide: name -> (dx, dy) in typographic points
LABEL_OFFSETS = {
    "BT.709":           (9,  -10),   # below-right of its point
    "BT.709+stretch":   (9,    8),   # above-right of its point
    "Lu 2012 (OpenCV)": (0,   12),   # straight above (clear of the far-left labels)
    "Ours (desktop)":   (-9,  11),   # above-left  of the filled star
    "Ours (Pi 5)":      (9,   11),   # above-right of the hollow star
}
OURS_DESKTOP = "Ours (desktop)"
TITLE = "Color250: contrast preservation vs. inference cost"
# ----------------------------------------------------------------------------

BLUE, ORANGE, GREY = "#4C78A8", "#F58518", "#7F7F7F"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", action="store_true", help="two-column width")
    ap.add_argument("--out", default="figs/pareto_quality_cost")
    ap.add_argument("--no-title", action="store_true", help="drop the title (let the caption carry it)")
    ap.add_argument("--label-ours", action="store_true",
                    help="also label the two 'ours' points inline (off by default: the legend already names them)")
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()

    pts = {k: v for k, v in POINTS.items() if v[0] is not None and v[1] is not None}
    if OURS_DESKTOP not in pts:
        raise SystemExit(f"'{OURS_DESKTOP}' needs both an E-score and a latency before plotting")
    incomplete = [k for k in POINTS if k not in pts]
    if incomplete:
        print("not plotted (missing E or latency):", ", ".join(incomplete))

    W, FS = (7.0, 9) if a.wide else (3.45, 7)
    plt.rcParams.update({"font.size": FS})
    fig, ax = plt.subplots(figsize=(W, W * 0.72))

    # -- Pareto frontier over the SAME-HARDWARE (desktop) points only ---------
    desk = {k: v for k, v in pts.items() if v[4] == "desktop"}
    front, best = [], -np.inf
    for name, (e, ms, *_ ) in sorted(desk.items(), key=lambda kv: kv[1][1]):
        if e > best:
            front.append((ms, e)); best = e
    if len(front) > 1:
        fx, fy = zip(*front)
        ax.step(fx, fy, where="post", color=GREY, lw=0.9, ls="--", zorder=1,
                label="_nolegend_")

    # -- connect the two "ours" points: same method, two hardware platforms ---
    ours_pts = {k: v for k, v in pts.items() if v[2] == "ours"}
    if len(ours_pts) == 2:
        (x0, y0), (x1, y1) = [(v[1], v[0]) for v in ours_pts.values()]
        ax.plot([x0, x1], [y0, y1], color=ORANGE, lw=0.8, ls=":", zorder=2,
                label="_nolegend_")

    # -- data points ----------------------------------------------------------
    for name, (e, ms, kind, meas, plat) in pts.items():
        ours = kind == "ours"
        col = ORANGE if ours else (BLUE if kind == "opt" else GREY)
        marker = "*" if ours else ("o" if kind == "opt" else "s")
        # Pi 5 = separate hardware series: hollow star; desktop ours = filled star
        pi = plat == "pi"
        face = "none" if (pi or not meas) else col
        size = 150 if (ours and not pi) else (120 if ours else 48)
        ax.scatter(ms, e, s=size, marker=marker, facecolor=face,
                   edgecolor=col, linewidths=1.5, zorder=4)
        # The legend already names the two 'ours' points (filled vs hollow star),
        # so we do not repeat them as inline labels (Comment 2.4b). The published
        # methods are NOT named in the legend, so they keep their labels.
        if ours and not a.label_ours:
            continue
        dx, dy = LABEL_OFFSETS.get(name, (8, -3))
        ax.annotate(name, (ms, e), textcoords="offset points", xytext=(dx, dy),
                    fontsize=FS - 0.5, zorder=5,
                    fontweight="bold" if ours else "normal",
                    color=ORANGE if ours else "#333333",
                    ha="center" if abs(dx) < 3 else ("right" if dx < 0 else "left"))

    ax.set_xscale("log")
    xs = [v[1] for v in pts.values()]; ys = [v[0] for v in pts.values()]
    ax.set_xlim(min(xs) * 0.30, max(xs) * 5.0)          # room for labels
    pad = (max(ys) - min(ys)) * 0.26
    # extra headroom only when the 'ours' points are labelled inline
    ax.set_ylim(min(ys) - pad, max(ys) + pad * (1.9 if a.label_ours else 1.15))

    ax.set_xlabel("inference time per image (ms, log scale)", fontsize=FS)
    ax.set_ylabel("E-score (contrast preservation, higher is better)", fontsize=FS)
    if not a.no_title:
        ax.set_title(TITLE, fontsize=FS + 1)
    ax.tick_params(labelsize=FS - 1)
    ax.grid(alpha=0.22, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # -- a proper boxed legend, inside the axes (Comment 2.4b) ----------------
    handles = [
        Line2D([], [], ls="", marker="*", mfc=ORANGE, mec=ORANGE, ms=12,
               label="ours — desktop CPU (1 thread)"),
        Line2D([], [], ls="", marker="*", mfc="none", mec=ORANGE, ms=12, mew=1.5,
               label="ours — Raspberry Pi 5 (ARM)"),
        Line2D([], [], ls="", marker="o", mfc=BLUE, mec=BLUE, ms=7,
               label="per-image optimization"),
        Line2D([], [], ls="", marker="s", mfc=GREY, mec=GREY, ms=6,
               label="fixed projection"),
    ]
    ax.legend(handles=handles, fontsize=FS - 1.5, loc="lower right",
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              borderpad=0.5, handletextpad=0.5, labelspacing=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=a.dpi, bbox_inches="tight")
    print(f"wrote {a.out}.png and .pdf   ({len(pts)} methods, {a.dpi} dpi)")


if __name__ == "__main__":
    main()
