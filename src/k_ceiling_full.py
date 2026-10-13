#!/usr/bin/env python3
"""Full-metric codebook ceiling for the K-ablation (Table VII / Fig. 5).

The 'codebook_ceiling_E' column written by run_k_ablation.py is the mean best-prototype E
taken from the codebook's stored E matrix, which build_codebook.py scores with a faster
screening metric (tau = 5,10,...,40; 2,500 pairs) over the whole 1,145-image pool. 'Realized E'
uses the full metric (tau = 1..40; 4,000 pairs) on the held-out 25% split. The two columns are
therefore not comparable.

This script recomputes the ceiling the same way realized E is measured: for each K and each
seed it takes the identical held-out split as run_k_ablation.held_out_eval (same permutation
of the codebook's own image list, same 256x256 resize, same full contrast_metrics), scores
every prototype on every held-out image, keeps the per-image best, and averages. It reports
mean +/- std over seeds, matching the realized-E column.

    python k_ceiling_full.py --tag rgbchroma --out k_ceiling_full.csv
"""
import argparse, csv, os
import numpy as np
from PIL import Image
from cues import cue_bank, fuse_cues
from metrics import contrast_metrics


def codebook_path(K, tag):
    return f"codebook_{tag}_k{K}.npz" if tag else f"codebook_k{K}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tag", default="rgbchroma")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--realized", default="k_ablation_4cue.csv",
                    help="optional: copy realized E from this file into the output for comparison")
    ap.add_argument("--out", default="k_ceiling_full.csv")
    a = ap.parse_args()

    realized = {}
    if a.realized and os.path.exists(a.realized):
        for r in csv.DictReader(open(a.realized)):
            realized[int(r["K"])] = (r["heldout_E_mean"], r["heldout_E_std"])

    cue_cache = {}

    def cues_for(path):
        if path not in cue_cache:
            p = path.replace("\\", "/")
            rgb = np.asarray(Image.open(p).convert("RGB").resize((a.size, a.size)), np.float64) / 255
            cue_cache[path] = (rgb, cue_bank(rgb))
        return cue_cache[path]

    rows = []
    for K in a.sizes:
        cb = codebook_path(K, a.tag)
        if not os.path.exists(cb):
            print(f"   [miss] {cb} -- skipping K={K}"); continue
        d = np.load(cb, allow_pickle=True)
        protos, paths = d["protos"], [str(p) for p in d["paths"]]
        best = {}                                   # image index -> full-metric best-prototype E
        per_seed = []
        for s in a.seeds:
            idx = np.random.default_rng(s).permutation(len(paths))
            va = idx[:max(1, int(len(paths) * a.val_frac))]
            vals = []
            for n, i in enumerate(va):
                if i not in best:
                    rgb, cu = cues_for(paths[i])
                    best[i] = max(contrast_metrics(rgb, fuse_cues(cu, w))["E"] for w in protos)
                vals.append(best[i])
                if (n + 1) % 50 == 0:
                    print(f"   K={K} seed {s}: {n + 1}/{len(va)}", flush=True)
            per_seed.append(float(np.mean(vals)))
            print(f"   K={K:>3} seed {s}: full-metric ceiling = {per_seed[-1]:.4f}", flush=True)
        rows.append((K, float(np.mean(per_seed)), float(np.std(per_seed)),
                     float(d["E"].max(1).mean())))

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["K", "ceiling_full_heldout_mean", "ceiling_full_heldout_std",
                    "ceiling_screening_pool", "realized_E_mean", "realized_E_std"])
        for K, m, sd, scr in rows:
            re_m, re_s = realized.get(K, ("", ""))
            w.writerow([K, f"{m:.4f}", f"{sd:.4f}", f"{scr:.4f}", re_m, re_s])
    print(f"wrote {a.out}")
    for K, m, sd, scr in rows:
        print(f"K={K:>3}  full-metric held-out ceiling {m:.4f} ± {sd:.4f}   (screening, pool: {scr:.4f})")


if __name__ == "__main__":
    main()
