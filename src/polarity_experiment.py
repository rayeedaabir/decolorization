#!/usr/bin/env python3
"""Chroma-polarity ablation on the 4-cue base — the corrected §VI-E experiment.

Compares, under one matched, held-out, multi-seed protocol:
    4-cue : [R, G, B, C]                     (the deployed bank)
    5-cue : [R, G, B, C, 1-C]                (4-cue + chroma complement = the polarity test)
    6-cue : [R, G, B, C, 1-C, MSR]           (5-cue + multi-scale Retinex)

C is the projection onto the dominant CIELAB opponent axis, normalised to [0,1]. An
eigenvector has no canonical sign, so with non-negative simplex weights the selector gets
only one polarity; adding 1-C (the projection onto the negated axis) lets it choose either
polarity per image without a negative weight. This answers the real §VI-E question for the
*4-cue* method: does admitting both polarities (5-cue), or both polarities plus MSR (6-cue),
reliably improve on the 4-cue bank?

SEEDED ORACLE (fair ceilings). Each larger bank's reachable set is a superset of the smaller
(set the new weight(s) to zero), so the oracle ceiling must be monotone 4 <= 5 <= 6. Naive
Dirichlet search in higher dimensions under-explores and can violate that. So the 5-cue
search is SEEDED from the matched 4-cue solution (embedded as C on c4 with 0 on the
complement, and the same magnitude moved to the complement c5), and the 6-cue search is
seeded from the 5-cue solution (MSR = 0), plus sparse Dirichlet draws. This guarantees
E_oracle(4) <= E_oracle(5) <= E_oracle(6) per image and finds a real polarity flip if one
exists. The experiment then asks whether the learned SELECTOR realises any of the ceiling.

Self-contained: computes its own 4-cue oracle (cached to --targets-out), builds matched
codebooks, trains matched selectors per seed, writes its own artifacts; touches nothing
behind the published numbers.

    python polarity_experiment.py --images data/Color250_rgb --seeds 0 1 2 \
        --k 15 --epochs 60 --samples 150 --out polarity_results.csv

Delete polarity_targets.csv to force a fresh oracle. Run from the harness folder.
"""
import argparse, csv, os
import numpy as np
from PIL import Image

from cues import multiscale_retinex, opponent_chroma, _to_float, fuse_cues
from metrics import contrast_metrics, isoluminant_ccpr, wilcoxon_signed_rank
from fusion_vit import FusionSelector, IMG

BT = {4: np.array([0.2126, 0.7152, 0.0722, 0.0]),
      5: np.array([0.2126, 0.7152, 0.0722, 0.0, 0.0]),
      6: np.array([0.2126, 0.7152, 0.0722, 0.0, 0.0, 0.0])}
FAST = dict(taus=range(5, 41, 5), n_pairs=2500)     # matches build_codebook.py / sixcue
SEARCH_TAUS = range(5, 41, 5)


def banks(rgb):
    """Return {4:cue_bank4, 5:cue_bank5, 6:cue_bank6}, all channels in [0,1]."""
    rgb = _to_float(rgb)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    C = opponent_chroma(rgb)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    b4 = np.stack([R, G, B, C])
    b5 = np.stack([R, G, B, C, 1.0 - C])
    b6 = np.stack([R, G, B, C, 1.0 - C, multiscale_retinex(luma)])
    return {4: b4, 5: b5, 6: b6}


def load(p, size=256):
    return np.asarray(Image.open(p).convert("RGB").resize((size, size)), np.float64) / 255


def search(rgb, cu, ndim, n_samples, rng, seeds=()):
    """Seeded Dirichlet search + local refine over the (ndim-1)-simplex, maximising E."""
    quick = lambda w: contrast_metrics(rgb, fuse_cues(cu, w), taus=SEARCH_TAUS, n_pairs=2500)["E"]
    full = lambda w: contrast_metrics(rgb, fuse_cues(cu, w))["E"]
    cands = [BT[ndim].copy()]
    for s in seeds:
        s = np.asarray(s, float)
        if s.sum() > 1e-9:
            cands.append(s / s.sum())
    best_w = max(cands, key=quick); best_q = quick(best_w)
    for _ in range(n_samples):
        alpha = 1.0 if rng.random() < 0.5 else 0.3
        w = rng.dirichlet(np.full(ndim, alpha))
        q = quick(w)
        if q > best_q:
            best_q, best_w = q, w
    best_e = full(best_w)
    for _ in range(50):
        cand = np.abs(best_w + rng.normal(0, 0.07, ndim)); cand /= cand.sum()
        e = full(cand)
        if e > best_e:
            best_e, best_w = e, cand
    return best_w, best_e


def kmeans(X, k, iters=40, seed=0):
    rng = np.random.default_rng(seed); C = X[rng.permutation(len(X))[:k]].copy()
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return C


def make_codebook(W, bt, k):
    protos = np.vstack([bt, kmeans(W, k)]); protos = np.clip(protos, 0, None)
    return protos / (protos.sum(1, keepdims=True) + 1e-9)


def score_matrix(paths, protos, ndim, size):
    E = np.zeros((len(paths), len(protos)), np.float32)
    for i, p in enumerate(paths):
        cu = banks(load(p, size))[ndim]
        for j in range(len(protos)):
            E[i, j] = contrast_metrics(load(p, size), fuse_cues(cu, protos[j]), **FAST)["E"]
    return E


def train_selector(paths, E, n_proto, seed, epochs, bs, lr, temp, device):
    import torch, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    torch.manual_seed(seed); np.random.seed(seed)
    idx = np.random.default_rng(seed).permutation(len(paths))
    nval = max(1, int(len(paths) * 0.25)); va_i, tr_i = idx[:nval], idx[nval:]

    class DS(Dataset):
        def __len__(self): return len(tr_i)
        def __getitem__(self, t):
            i = tr_i[t]
            a = np.asarray(Image.open(paths[i]).convert("RGB").resize((IMG, IMG)), np.float32) / 255
            return torch.from_numpy(a).permute(2, 0, 1), torch.from_numpy(E[i])

    model = FusionSelector(n_proto).to(device)
    opt = torch.optim.Adam(model.parameters(), lr)
    dl = DataLoader(DS(), bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for x, e in dl:
            x, e = x.to(device), e.to(device)
            loss = F.kl_div(F.log_softmax(model(x), 1), F.softmax(e / temp, 1), reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
    return model, va_i


def eval_selector(model, paths, protos, ndim, va_i, size, device):
    import torch
    model.eval(); Es, Is = [], []
    with torch.no_grad():
        for i in va_i:
            rgb = load(paths[i], size); cu = banks(rgb)[ndim]
            small = np.asarray(Image.open(paths[i]).convert("RGB").resize((IMG, IMG)), np.float32) / 255
            j = int(model(torch.from_numpy(small).permute(2, 0, 1)[None].to(device))[0].argmax().item())
            g = fuse_cues(cu, protos[j])
            Es.append(contrast_metrics(rgb, g)["E"]); Is.append(isoluminant_ccpr(rgb, g)["CCPR_iso"])
    return np.array(Es), np.array(Is)


def main():
    ap = argparse.ArgumentParser(description="4-cue chroma-polarity ablation (4 vs 5 vs 6)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--targets-out", default="polarity_targets.csv")
    ap.add_argument("--out", default="polarity_results.csv")
    a = ap.parse_args()
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    paths = sorted(os.path.join(a.images, f) for f in os.listdir(a.images)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    print(f"{len(paths)} images; device={device}")

    # ---- oracle (cached): 4-cue fresh, 5-cue seeded from 4, 6-cue seeded from 5 ----
    if os.path.exists(a.targets_out):
        print(f"[skip] reusing {a.targets_out}")
        by = {r["image"]: r for r in csv.DictReader(open(a.targets_out))}
        W = {n: np.array([[float(by[os.path.basename(p)][f"w{n}_{k}"]) for k in range(n)] for p in paths]) for n in (4, 5, 6)}
        O = {n: np.array([float(by[os.path.basename(p)][f"E{n}"]) for p in paths]) for n in (4, 5, 6)}
    else:
        rng = np.random.default_rng(0)
        W = {4: np.zeros((len(paths), 4)), 5: np.zeros((len(paths), 5)), 6: np.zeros((len(paths), 6))}
        O = {4: np.zeros(len(paths)), 5: np.zeros(len(paths)), 6: np.zeros(len(paths))}
        for i, p in enumerate(paths):
            rgb = load(p, a.size); cb = banks(rgb)
            w4, e4 = search(rgb, cb[4], 4, a.samples, rng)
            emb5 = np.r_[w4, 0.0]; flip5 = np.r_[w4[:3], 0.0, w4[3]]
            w5, e5 = search(rgb, cb[5], 5, a.samples, rng, seeds=(emb5, flip5))
            w6, e6 = search(rgb, cb[6], 6, a.samples, rng, seeds=(np.r_[w5, 0.0],))
            W[4][i], W[5][i], W[6][i] = w4, w5, w6
            O[4][i], O[5][i], O[6][i] = e4, e5, e6
            if (i + 1) % 25 == 0: print(f"   oracle {i+1}/{len(paths)}")
        with open(a.targets_out, "w", newline="") as fh:
            cols = ["image"] + [f"w{n}_{k}" for n in (4, 5, 6) for k in range(n)] + [f"E{n}" for n in (4, 5, 6)]
            w_ = csv.writer(fh); w_.writerow(cols)
            for idx, p in enumerate(paths):
                row = [os.path.basename(p)]
                for n in (4, 5, 6): row += [round(v, 5) for v in W[n][idx]]
                for n in (4, 5, 6): row += [round(float(O[n][idx]), 5)]
                w_.writerow(row)
        print(f"   wrote {a.targets_out}")
    print(f"oracle mean E: 4-cue={O[4].mean():.4f}  5-cue={O[5].mean():.4f}  6-cue={O[6].mean():.4f}  (must be non-decreasing)")
    print(f"oracle complement usage: mean w(1-C) 5-cue={W[5][:,4].mean():.4f}, 6-cue={W[6][:,4].mean():.4f}")

    # ---- codebooks + score matrices (once) ----
    protos = {n: make_codebook(W[n], BT[n], a.k) for n in (4, 5, 6)}
    Emat = {n: score_matrix(paths, protos[n], n, a.size) for n in (4, 5, 6)}
    for n in (4, 5, 6):
        print(f"   codebook{n} ceiling E={Emat[n].max(1).mean():.4f}")

    # ---- per-seed matched train + held-out eval ----
    per = {n: {"E": [], "I": []} for n in (4, 5, 6)}
    p54, p64 = [], []   # per-seed paired Wilcoxon p (5 vs 4, 6 vs 4)
    for s in a.seeds:
        models, vas = {}, {}
        for n in (4, 5, 6):
            models[n], vas[n] = train_selector(paths, Emat[n], len(protos[n]), s, a.epochs, a.bs, a.lr, a.temp, device)
        assert np.array_equal(vas[4], vas[5]) and np.array_equal(vas[5], vas[6])
        e, i = {}, {}
        for n in (4, 5, 6):
            e[n], i[n] = eval_selector(models[n], paths, protos[n], n, vas[n], a.size, device)
            per[n]["E"].append(e[n].mean()); per[n]["I"].append(np.nanmean(i[n]))
        d54, d64 = e[5] - e[4], e[6] - e[4]
        p54.append(wilcoxon_signed_rank(d54[np.isfinite(d54)])[0])
        p64.append(wilcoxon_signed_rank(d64[np.isfinite(d64)])[0])
        print(f"  seed {s}: E 4={e[4].mean():.4f} 5={e[5].mean():.4f} 6={e[6].mean():.4f} | "
              f"p(5vs4)={p54[-1]:.3f} p(6vs4)={p64[-1]:.3f}")

    def ms(xs): return np.mean(xs), np.std(xs)
    print("\n" + "=" * 70)
    print(f"4-CUE CHROMA-POLARITY ABLATION (matched, held-out, seeds {a.seeds})")
    print(f"{'arm':14}{'held-out E':>18}{'CCPR-iso':>18}")
    for n, lab in [(4, "4-cue"), (5, "5-cue (+1-C)"), (6, "6-cue (+1-C,MSR)")]:
        em, es = ms(per[n]["E"]); im, isd = ms(per[n]["I"])
        print(f"{lab:14}{em:>12.4f}±{es:.4f}{im:>12.4f}±{isd:.4f}")
    print(f"paired Wilcoxon per seed  5 vs 4: {[round(x,3) for x in p54]}   6 vs 4: {[round(x,3) for x in p64]}")
    print("=" * 70)

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "heldout_E_mean", "heldout_E_std", "CCPRiso_mean", "CCPRiso_std", "ceiling_E"])
        for n, lab in [(4, "4cue"), (5, "5cue_+1-C"), (6, "6cue_+1-C+MSR")]:
            em, es = ms(per[n]["E"]); im, isd = ms(per[n]["I"])
            w.writerow([lab, f"{em:.4f}", f"{es:.4f}", f"{im:.4f}", f"{isd:.4f}", f"{Emat[n].max(1).mean():.4f}"])
        w.writerow(["p_wilcoxon_5vs4_per_seed", *[f"{x:.4f}" for x in p54]])
        w.writerow(["p_wilcoxon_6vs4_per_seed", *[f"{x:.4f}" for x in p64]])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
