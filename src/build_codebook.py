"""Build a CODEBOOK of cue weightings + score every image against every entry.

Why: near-optimal weightings are non-unique (many very different vectors tie), so
regressing to them with MSE averages between good solutions and scores worse than a
fixed weighting. Instead we discretise: cluster the oracle weights into K prototypes,
measure each prototype's E-score on each image, and let the model SELECT one.

BT.709 is always included as a prototype, so the learned selector has a safe fallback.

    python build_codebook.py --targets targets_*.csv --k 15 --out codebook.npz
"""
import argparse, csv, glob, shutil
import numpy as np
from PIL import Image
from cues import cue_bank, fuse_cues, BT709
from metrics import contrast_metrics
import provenance

FAST = dict(taus=range(5, 41, 5), n_pairs=2500)      # consistent across prototypes


def kmeans(X, k, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.permutation(len(X))[:k]].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=15, help="clusters (BT.709 is added on top)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="codebook.npz")
    ap.add_argument("--no-stamp", action="store_true",
                    help="skip the dated archive copy (not recommended)")
    a = ap.parse_args()

    files = sorted({f for pat in a.targets for f in glob.glob(pat)})
    paths, W = [], []
    for c in files:
        for r in csv.DictReader(open(c)):
            paths.append(r.get("path") or r["image"])
            W.append([float(r[k]) for k in ("w_R", "w_G", "w_B", "w_MSR", "w_CHROMA")])
    W = np.array(W)
    print(f"{len(paths)} images from {len(files)} target files")
    print(f"oracle weight std per cue: [{' '.join(f'{v:.2f}' for v in W.std(0))}]  "
          f"(large std = why MSE regression fails)")

    protos = np.vstack([BT709, kmeans(W, a.k)])       # (K+1, 5); index 0 = BT.709
    protos = np.clip(protos, 0, None)
    protos /= protos.sum(1, keepdims=True) + 1e-9
    K = len(protos)
    print(f"codebook: {K} prototypes (index 0 = BT.709 fallback)")

    E = np.zeros((len(paths), K), np.float32)
    for i, p in enumerate(paths):
        rgb = np.asarray(Image.open(p).convert("RGB").resize((a.size, a.size)), np.float64) / 255
        cu = cue_bank(rgb)
        for j in range(K):
            E[i, j] = contrast_metrics(rgb, fuse_cues(cu, protos[j]), **FAST)["E"]
        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(paths)}")

    prov = provenance.as_json(k=a.k, n_protos=int(K), n_images=len(paths),
                              size=a.size, targets=files)
    np.savez_compressed(a.out, protos=protos, E=E, paths=np.array(paths),
                        provenance=np.array(prov))
    best = E.max(1).mean(); fixed = E[:, 0].mean()
    per_img = E.argmax(1)
    print(f"\nmean E — BT.709 fallback     : {fixed:.4f}")
    print(f"mean E — best SINGLE prototype: {E.mean(0).max():.4f}  (best fixed choice)")
    print(f"mean E — per-image best (ceiling of selection): {best:.4f}  "
          f"({(best-fixed)/fixed*100:+.2f}% over BT.709)")
    print(f"prototypes actually used as best: {len(set(per_img.tolist()))}/{K}")
    print(f"wrote -> {a.out}")
    if not a.no_stamp:
        archive = provenance.stamped_name(a.out)
        shutil.copyfile(a.out, archive)
        print(f"archived  -> {archive}   (dated copy; {a.out} is the live one)")
    print("run  python check_artifacts.py  to confirm the selector matches this codebook")


if __name__ == "__main__":
    main()
