"""Oracle: the best per-image weighting over the cue bank.

Random (Dirichlet) search over the 5-cue simplex maximising E-score, compared against
the fixed BT.709 luma weighting. Establishes the ceiling for a learned model and writes
per-image best weights as ViT training targets.

    python cue_search.py --images data/Cadik_rgb    --out targets_cadik.csv
    python cue_search.py --images data/Color250_rgb --out targets_color250.csv
    python cue_search.py --images data/UCM/images --per-class 20 --out targets_ucm.csv
"""
import argparse, csv, os
import numpy as np
from PIL import Image
from cues import cue_bank, fuse_cues, BT709, N_CUES, collect
from metrics import contrast_metrics


SEARCH_TAUS = range(5, 41, 5)             # coarse sweep during search (speed)


def load(p, size=256):
    return np.asarray(Image.open(p).convert("RGB").resize((size, size)), np.float64) / 255


def search(rgb, n_samples, rng):
    cues = cue_bank(rgb)
    quick = lambda w: contrast_metrics(rgb, fuse_cues(cues, w), taus=SEARCH_TAUS, n_pairs=2500)["E"]
    full = lambda w: contrast_metrics(rgb, fuse_cues(cues, w))["E"]
    # coarse random search over the cue simplex
    best_w, best_q = BT709.copy(), quick(BT709)
    for _ in range(n_samples):
        w = rng.dirichlet(np.ones(N_CUES))
        q = quick(w)
        if q > best_q:
            best_q, best_w = q, w
    # refine on the FULL metric (what we report) by local hill-climbing
    best_e = full(best_w)
    for _ in range(50):
        cand = np.abs(best_w + rng.normal(0, 0.07, N_CUES)); cand /= cand.sum() + 1e-9
        e = full(cand)
        if e > best_e:
            best_e, best_w = e, cand
    return full(BT709), best_e, best_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--out", default="targets.csv")
    ap.add_argument("--per-class", type=int, default=0); ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--samples", type=int, default=150)
    a = ap.parse_args()
    files = collect(a.images, a.recursive or bool(a.per_class), a.per_class)
    if not files:
        raise SystemExit(f"no images under {a.images} (nested? use --per-class K)")
    rng = np.random.default_rng(0)
    rows, fx, ox = [], [], []
    for i, f in enumerate(files):
        e_fix, e_best, w = search(load(f), a.samples, rng)
        rows.append([f, os.path.basename(f)] + [round(v, 5) for v in w] +
                    [round(e_fix, 4), round(e_best, 4)])
        fx.append(e_fix); ox.append(e_best)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(files)}")
    with open(a.out, "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["path", "image", "w_R", "w_G", "w_B", "w_MSR", "w_CHROMA", "E_fixed", "E_oracle"])
        w_.writerows(rows)
    fx, ox = np.array(fx), np.array(ox)
    print(f"images={len(files)}")
    print(f"  E  fixed BT.709 weighting : {fx.mean():.4f}")
    print(f"  E  per-image oracle       : {ox.mean():.4f}")
    print(f"  mean improvement          : {(ox-fx).mean():+.4f}  ({(ox-fx).mean()/fx.mean()*100:+.2f}%)")
    print(f"  wrote targets -> {a.out}")


if __name__ == "__main__":
    main()
