"""Regenerate the result figures.

EDIT THE NUMBERS BELOW after your pipeline re-run, then:  python make_charts.py
Entries left as None are skipped, so you can fill the table in as results arrive.
Writes figs/quality_comparison.png, figs/downstream_comparison.png, figs/speed_accuracy.png
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- fill these in --------------------------------
E_SCORE = {                        # E-score (higher is better)
    "BT.709":        {"Cadik": None, "Color250": None},
    "CIE Y":         {"Cadik": None, "Color250": None},
    "Grundland 07":  {"Cadik": None, "Color250": None},
    "Smith 08":      {"Cadik": None, "Color250": None},
    "Color2Gray 05": {"Cadik": None, "Color250": None},
    "Lu 14":         {"Cadik": None, "Color250": None},
    "Ours (S1)":     {"Cadik": None, "Color250": None},
    "Ours (S2)":     {"Cadik": None, "Color250": None},
}
ACCURACY = {                       # downstream best val accuracy
    "BT.709":    {"UCM": None, "AID": None},
    "Ours (S1)": {"UCM": None, "AID": None},
    "Ours (S2)": {"UCM": None, "AID": None},
}
LATENCY_MS = {"BT.709": None, "Ours (S1)": None, "Ours (S2)": None}
# ----------------------------------------------------------------------------

os.makedirs("figs", exist_ok=True)
C = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#9D755D", "#72B7B2", "#333333"]


def grouped(dct, cols, title, ylabel, out):
    names = [k for k, v in dct.items() if all(v[c] is not None for c in cols)]
    if not names:
        print(f"skip {out} — no numbers filled in yet"); return
    x = np.arange(len(names)); w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(names)), 4.3))
    for j, c in enumerate(cols):
        ax.bar(x + (j - (len(cols) - 1) / 2) * w, [dct[n][c] for n in names], w, label=c, color=C[j])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


grouped(E_SCORE, ["Cadik", "Color250"], "Contrast preservation (E-score)", "E-score",
        "figs/quality_comparison.png")
grouped(ACCURACY, ["UCM", "AID"], "Downstream classification accuracy", "best val acc",
        "figs/downstream_comparison.png")

pts = [(k, LATENCY_MS[k], ACCURACY.get(k, {}).get("UCM")) for k in LATENCY_MS]
pts = [(k, l, a) for k, l, a in pts if l is not None and a is not None]
if pts:
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for i, (k, l, a) in enumerate(pts):
        ax.scatter(l, a, s=110, color=C[i], zorder=3)
        ax.annotate(k, (l, a), textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("decolorization latency (ms/img, log)")
    ax.set_ylabel("UCM accuracy"); ax.set_title("Accuracy vs latency", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("figs/speed_accuracy.png", dpi=130); plt.close(fig)
    print("wrote figs/speed_accuracy.png")
else:
    print("skip figs/speed_accuracy.png — fill LATENCY_MS and ACCURACY first")
