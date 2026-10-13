#!/usr/bin/env python3
"""Regression baseline for Table VIII. Trains the SAME ViT backbone to REGRESS the
per-image oracle cue weights under MSE, instead of SELECTING from a codebook, and reports
absolute held-out E vs BT.709. --constraint sets how the 5 outputs are turned into weights:

    softmax : project onto the simplex (non-negative, sum 1) -- the fair, deployable baseline
    sum1    : normalise raw outputs by their sum (may go negative) -- semi-constrained
    none    : raw outputs used directly (no simplex constraint) -- the naive regressor that
              the original -1.1% failure most likely came from

Reuses the exact backbone (FusionSelector, 5-way head) and the same seed-split protocol as
fusion_vit.py; reads the same oracle-target CSVs that built the codebook.

    python regress_baseline.py --targets targets_cadik.csv targets_color250.csv targets_ucm.csv targets_aid.csv --seeds 0 1 2 --epochs 60 --constraint softmax
    python regress_baseline.py --targets ... --constraint none      # test the unconstrained variant
    python regress_baseline.py --targets ... --constraint sum1      # test the sum-normalised variant
"""
import argparse, csv, glob
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from cues import cue_bank, fuse_cues, BT709
from metrics import contrast_metrics
from fusion_vit import FusionSelector, IMG


def constrain(z, mode):
    if mode == "softmax":
        return F.softmax(z, dim=1)
    if mode == "sum1":
        return z / (z.sum(1, keepdim=True) + 1e-9)
    return z                                   # none: raw outputs


def load_targets(patterns):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    paths, W = [], []
    for c in files:
        for r in csv.DictReader(open(c)):
            paths.append(r.get("path") or r["image"])
            W.append([float(r[k]) for k in ("w_R", "w_G", "w_B", "w_MSR", "w_CHROMA")])
    if not paths:
        raise SystemExit("no rows in --targets")
    return paths, np.array(W, np.float32)


class DS(Dataset):
    def __init__(self, paths, W):
        self.paths, self.W = paths, W

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        a = np.asarray(Image.open(self.paths[i]).convert("RGB").resize((IMG, IMG)), np.float32) / 255
        return torch.from_numpy(a).permute(2, 0, 1), torch.from_numpy(self.W[i])


def run_seed(paths, W, seed, epochs, bs, lr, mode, device):
    torch.manual_seed(seed); np.random.seed(seed)
    idx = np.random.default_rng(seed).permutation(len(paths))
    nval = max(1, int(len(paths) * 0.25))
    va, tr = idx[:nval], idx[nval:]
    dl = DataLoader(DS([paths[i] for i in tr], W[tr]), bs, shuffle=True)
    model = FusionSelector(5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr)
    for _ in range(epochs):
        model.train()
        for x, w in dl:
            x, w = x.to(device), w.to(device)
            loss = F.mse_loss(constrain(model(x), mode), w)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval(); e_reg, e_fix = [], []
    with torch.no_grad():
        for i in va:
            rgb = np.asarray(Image.open(paths[i]).convert("RGB").resize((256, 256)), np.float64) / 255
            cu = cue_bank(rgb)
            small = np.asarray(Image.open(paths[i]).convert("RGB").resize((IMG, IMG)), np.float32) / 255
            x = torch.from_numpy(small).permute(2, 0, 1)[None].to(device)
            w = constrain(model(x), mode)[0].cpu().numpy().astype(np.float64)
            e_reg.append(contrast_metrics(rgb, fuse_cues(cu, w))["E"])
            e_fix.append(contrast_metrics(rgb, fuse_cues(cu, BT709))["E"])
    return float(np.mean(e_reg)), float(np.mean(e_fix))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--constraint", choices=["softmax", "sum1", "none"], default="softmax")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths, W = load_targets(a.targets)
    print(f"{len(paths)} images; regressing oracle weights (MSE, constraint={a.constraint}); device={device}")
    R, Fx = [], []
    for s in a.seeds:
        er, ef = run_seed(paths, W, s, a.epochs, a.bs, a.lr, a.constraint, device)
        R.append(er); Fx.append(ef)
        print(f"  seed {s}: regression E={er:.4f}   BT.709 E={ef:.4f}   ({(er-ef)/ef*100:+.2f}% vs BT.709)")
    R, Fx = np.array(R), np.array(Fx)
    sd = R.std(ddof=1) if len(R) > 1 else 0.0
    print(f"\nconstraint={a.constraint}: regression E {R.mean():.4f} ± {sd:.4f} | BT.709 {Fx.mean():.4f} "
          f"| {(R.mean()-Fx.mean())/Fx.mean()*100:+.2f}% vs BT.709")
    out = a.out or f"regress_baseline_{a.constraint}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "regression_E", "bt709_E", "pct_vs_bt709", "constraint"])
        for s, er, ef in zip(a.seeds, R, Fx):
            w.writerow([s, f"{er:.4f}", f"{ef:.4f}", f"{(er-ef)/ef*100:+.2f}", a.constraint])
        w.writerow(["mean", f"{R.mean():.4f}", f"{Fx.mean():.4f}",
                    f"{(R.mean()-Fx.mean())/Fx.mean()*100:+.2f}", a.constraint])
    print("wrote", out)


if __name__ == "__main__":
    main()
