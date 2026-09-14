"""Oracle: the best per-image weighting over the cue bank.

Random (Dirichlet) search over the cue simplex maximising E-score, compared against the
fixed BT.709 luma weighting. Establishes the ceiling for a learned model and writes
per-image best weights as ViT training targets.

    python cue_search.py --images data/Cadik_rgb    --out targets_cadik.csv
    python cue_search.py --images data/Color250_rgb --out targets_color250.csv
    python cue_search.py --images data/UCM/images --per-class 20 --out targets_ucm.csv

CUE-ABLATION MODE (Comment 4.1). --cues restricts the search to a subset of the five
cues; the excluded cues get exactly zero weight, so the printed "per-image oracle" is the
achievable ceiling for that cue configuration -- the upper bound on what any selector over
those cues can reach. This isolates each cue's contribution independent of selector
quality. E_fixed stays the TRUE BT.709 reference in every run, so the runs are comparable.

    python cue_search.py --images data/Color250_rgb --cues R,G,B          --out abl_rgb.csv
    python cue_search.py --images data/Color250_rgb --cues R,G,B,MSR      --out abl_rgb_msr.csv
    python cue_search.py --images data/Color250_rgb --cues R,G,B,CHROMA   --out abl_rgb_chroma.csv
    python cue_search.py --images data/Color250_rgb --cues all            --out abl_full.csv
"""
import argparse, csv, os
import numpy as np
from PIL import Image
from cues import cue_bank, fuse_cues, BT709, N_CUES, CUE_NAMES, collect
from metrics import contrast_metrics


SEARCH_TAUS = range(5, 41, 5)             # coarse sweep during search (speed)


def load(p, size=256):
    return np.asarray(Image.open(p).convert("RGB").resize((size, size)), np.float64) / 255


def parse_cues(spec):
    """'all' | comma list of indices ('0,1,2') | comma list of names ('R,G,B,CHROMA')."""
    if spec is None or spec.strip().lower() in ("all", ""):
        return list(range(N_CUES))
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        if tok.isdigit():
            out.append(int(tok))
        else:
            up = tok.upper()
            if up not in CUE_NAMES:
                raise SystemExit(f"unknown cue '{tok}'. Use indices 0..{N_CUES-1} or names {CUE_NAMES}")
            out.append(CUE_NAMES.index(up))
    out = sorted(set(out))
    if not out or any(i < 0 or i >= N_CUES for i in out):
        raise SystemExit(f"--cues must select from 0..{N_CUES-1} ({CUE_NAMES})")
    return out


def search(rgb, n_samples, rng, active=None):
    cues = cue_bank(rgb)
    if active is None:
        active = list(range(N_CUES))
    active = np.asarray(active, int)
    quick = lambda w: contrast_metrics(rgb, fuse_cues(cues, w), taus=SEARCH_TAUS, n_pairs=2500)["E"]
    full = lambda w: contrast_metrics(rgb, fuse_cues(cues, w))["E"]

    def sample():
        w = np.zeros(N_CUES)
        w[active] = rng.dirichlet(np.ones(len(active)))
        return w

    # E_fixed is ALWAYS the true BT.709 reference, whatever the active subset,
    # so ablation runs share a common baseline.
    e_fixed = full(BT709)

    # oracle search restricted to the active cues; start from BT.709 projected
    # onto the subset (renormalised) -- for any subset containing R,G,B this is
    # exactly BT.709.
    w0 = np.zeros(N_CUES); w0[active] = BT709[active]
    s = w0.sum()
    best_w = (w0 / s) if s > 1e-9 else sample()
    best_q = quick(best_w)
    for _ in range(n_samples):
        w = sample()
        q = quick(w)
        if q > best_q:
            best_q, best_w = q, w
    # refine on the FULL metric by local hill-climbing, staying on the subset
    best_e = full(best_w)
    for _ in range(50):
        cand = best_w.copy()
        cand[active] = np.abs(best_w[active] + rng.normal(0, 0.07, len(active)))
        ssum = cand[active].sum()
        if ssum > 1e-9:
            cand[active] /= ssum
        e = full(cand)
        if e > best_e:
            best_e, best_w = e, cand
    return e_fixed, best_e, best_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--out", default="targets.csv")
    ap.add_argument("--per-class", type=int, default=0); ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--cues", default="all",
                    help="cue subset for ablation: 'all', indices '0,1,2', or names 'R,G,B,CHROMA'")
    a = ap.parse_args()
    active = parse_cues(a.cues)
    files = collect(a.images, a.recursive or bool(a.per_class), a.per_class)
    if not files:
        raise SystemExit(f"no images under {a.images} (nested? use --per-class K)")
    print(f"active cues: {[CUE_NAMES[i] for i in active]}  (indices {active})")
    rng = np.random.default_rng(0)
    rows, fx, ox = [], [], []
    for i, f in enumerate(files):
        e_fix, e_best, w = search(load(f), a.samples, rng, active)
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
    print(f"images={len(files)}   cues={[CUE_NAMES[i] for i in active]}")
    print(f"  E  fixed BT.709 weighting : {fx.mean():.4f}")
    print(f"  E  per-image oracle       : {ox.mean():.4f}")
    print(f"  mean improvement          : {(ox-fx).mean():+.4f}  ({(ox-fx).mean()/fx.mean()*100:+.2f}%)")
    print(f"  wrote targets -> {a.out}")


if __name__ == "__main__":
    main()
