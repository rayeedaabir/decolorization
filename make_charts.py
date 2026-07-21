"""Regenerate the analytical figures used in the report/plan.
    python make_charts.py
Writes figs/reproduction_summary.png, figs/ours_vs_paper.png, figs/speed_accuracy.png.
Numbers are the logged results — edit here if they change.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figs", exist_ok=True)

m = ["bt601", "B1", "B1+2", "B1+2+3"]; x = np.arange(4); w = 0.38
ucm = [0.9619, 0.9690, 0.9690, 0.9690]; aid = [0.9295, 0.9340, 0.9360, 0.9390]
cad = [0.6742, 0.7115, 0.7064, 0.7201]; col = [0.7801, 0.7682, 0.7519, 0.7860]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.bar(x-w/2, ucm, w, label="UCM", color="#4C78A8"); a1.bar(x+w/2, aid, w, label="AID", color="#F58518")
a1.set_title("Downstream classification accuracy"); a1.set_xticks(x); a1.set_xticklabels(m)
a1.set_ylim(0.90, 0.975); a1.legend(); a1.set_ylabel("best val acc")
a2.bar(x-w/2, cad, w, label="Cadik", color="#54A24B"); a2.bar(x+w/2, col, w, label="Color250", color="#B279A2")
a2.set_title("E-score (contrast preservation)"); a2.set_xticks(x); a2.set_xticklabels(m)
a2.set_ylim(0.60, 0.82); a2.legend(); a2.set_ylabel("E-score")
fig.suptitle("Paper 3 reproduction — branch-by-branch", fontweight="bold")
fig.tight_layout(); fig.savefig("figs/reproduction_summary.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 4.3)); x = np.arange(2); w = 0.36
ours = [0.9762, 0.9310]; paper = [0.9786, 0.9365]
ax.bar(x-w/2, ours, w, label="Ours (reproduction)", color="#4C78A8")
ax.bar(x+w/2, paper, w, label="Paper 3 (reported)", color="#F58518")
ax.set_xticks(x); ax.set_xticklabels(["UC Merced", "AID"]); ax.set_ylim(0.90, 0.99)
ax.set_ylabel("accuracy"); ax.set_title("Reproduction vs Paper 3 — classification"); ax.legend()
for i, (o, p) in enumerate(zip(ours, paper)):
    ax.text(i, 0.905, f"D {(o-p)/p*100:+.2f}%", ha="center", fontsize=9, color="#555")
fig.tight_layout(); fig.savefig("figs/ours_vs_paper.png", dpi=130); plt.close(fig)

mm = ["bt601", "B1", "B1+2", "B1+2+3", "paper3"]; lat = [0.46, 0.80, 23.85, 26.19, 30.02]
ucm = [0.9619, 0.9690, 0.9690, 0.9690, 0.9762]; aid = [0.9295, 0.9340, 0.9360, 0.9390, 0.9310]
cols = ["#888888", "#4C78A8", "#54A24B", "#E45756", "#B279A2"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, acc, tt in ((a1, ucm, "UC Merced"), (a2, aid, "AID")):
    for i, name in enumerate(mm):
        ax.scatter(lat[i], acc[i], s=90, color=cols[i], zorder=3)
        ax.annotate(name, (lat[i], acc[i]), textcoords="offset points", xytext=(6, 5), fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("decolorization latency (ms/img, log)")
    ax.set_ylabel("best val accuracy"); ax.set_title(f"Speed vs accuracy — {tt}"); ax.grid(True, alpha=0.3)
fig.suptitle("Accuracy vs latency (your runs)", fontweight="bold")
fig.tight_layout(); fig.savefig("figs/speed_accuracy.png", dpi=130); plt.close(fig)
print("wrote figs/reproduction_summary.png, figs/ours_vs_paper.png, figs/speed_accuracy.png")
