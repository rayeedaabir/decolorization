#!/usr/bin/env python3
"""Selector thumbnail-resolution ablation: realised accuracy vs selector input size.

The selector reads only a small thumbnail, so its cost is independent of the image's
native resolution -- the thumbnail size is a design knob. This sweeps it (64/96/128/160;
each divisible by the 16 px patch) and reports, per size, the realised held-out E and
CCPR-iso plus the selector's parameter count and single-thread CPU latency, to justify
128 as the accuracy/cost knee.

It reuses your existing codebook.npz (the prototypes and per-image E targets are
independent of the thumbnail size); only the ViT selector is retrained per size. It
imports your FusionSelector and copies fusion_vit.py's exact training objective and
held-out evaluation, so nothing in the harness is modified.

    python run_thumb_ablation.py --codebook codebook.npz \
        --sizes 64 96 128 160 --seeds 0 1 2 --epochs 60

Run from inside the harness folder. Writes thumb_ablation_table.csv and, per size/seed,
thumb_i<size>_s<seed>.pt.
"""
import argparse, csv, os, time
import numpy as np
from PIL import Image


def build_dataset(paths, E, img):
    import torch
    from torch.utils.data import Dataset

    class DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            a = np.asarray(Image.open(paths[i]).convert("RGB").resize((img, img)),
                           np.float32) / 255
            return torch.from_numpy(a).permute(2, 0, 1), torch.from_numpy(E[i])

    return DS()


def train_one(paths, E, protos, img, seed, epochs, bs, lr, temp, device):
    """Copy of fusion_vit.main's training: soft KL targets, Adam, same split by seed."""
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from fusion_vit import FusionSelector

    torch.manual_seed(seed); np.random.seed(seed)
    idx = np.random.default_rng(seed).permutation(len(paths))
    nval = max(1, int(len(paths) * 0.25))
    va_i, tr_i = idx[:nval], idx[nval:]

    ds = build_dataset([paths[i] for i in tr_i], E[tr_i], img)
    model = FusionSelector(len(protos), img=img).to(device)
    opt = torch.optim.Adam(model.parameters(), lr)
    dl = DataLoader(ds, bs, shuffle=True)
    for ep in range(epochs):
        model.train()
        for x, e in dl:
            x, e = x.to(device), e.to(device)
            target = F.softmax(e / temp, dim=1)
            loss = F.kl_div(F.log_softmax(model(x), 1), target, reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
    return model, va_i


def eval_one(model, paths, protos, va_i, img, device):
    """Reproduce fusion_vit.main's held-out eval at 256x256, adding CCPR-iso."""
    import torch
    from cues import cue_bank, fuse_cues
    from metrics import contrast_metrics, isoluminant_ccpr
    model.eval()
    Es, Is = [], []
    with torch.no_grad():
        for i in va_i:
            rgb = np.asarray(Image.open(paths[i]).convert("RGB").resize((256, 256)),
                             np.float64) / 255
            cu = cue_bank(rgb)
            small = np.asarray(Image.open(paths[i]).convert("RGB").resize((img, img)),
                               np.float32) / 255
            x = torch.from_numpy(small).permute(2, 0, 1)[None].to(device)
            j = int(model(x)[0].argmax().item())
            g = fuse_cues(cu, protos[j])
            Es.append(contrast_metrics(rgb, g)["E"])
            Is.append(isoluminant_ccpr(rgb, g)["CCPR_iso"])
    return np.array(Es), np.array(Is)


def selector_latency_ms(K, img, reps=50):
    """Single-thread CPU latency of the ViT forward pass at this thumbnail size."""
    import torch
    from fusion_vit import FusionSelector
    torch.set_num_threads(1)
    m = FusionSelector(K, img=img).eval()
    x = torch.rand(1, 3, img, img)
    with torch.no_grad():
        m(x)  # warm-up
        t = time.perf_counter()
        for _ in range(reps):
            m(x)
        dt = (time.perf_counter() - t) / reps
    return dt * 1e3


def main():
    ap = argparse.ArgumentParser(description="selector thumbnail-resolution ablation")
    ap.add_argument("--codebook", default="codebook.npz")
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 96, 128, 160])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--out", default="thumb_ablation_table.csv")
    a = ap.parse_args()

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    d = np.load(a.codebook, allow_pickle=True)
    protos, E, paths = d["protos"], d["E"], [str(p) for p in d["paths"]]
    K = len(protos)
    print(f"codebook K={K}, {len(paths)} images, device={device}")

    for s in a.sizes:
        if s % 16 != 0:
            raise SystemExit(f"--sizes must be divisible by the 16px patch; {s} is not")

    rows = []
    for img in a.sizes:
        Es, Is = [], []
        for seed in a.seeds:
            ckpt = f"thumb_i{img}_s{seed}.pt"
            model, va_i = train_one(paths, E, protos, img, seed, a.epochs,
                                    a.bs, a.lr, a.temp, device)
            torch.save({"state_dict": model.state_dict(), "img": img}, ckpt)
            e, i = eval_one(model, paths, protos, va_i, img, device)
            Es.append(e.mean()); Is.append(np.nanmean(i))
            print(f"   img={img:>3} seed {seed}:  E={e.mean():.4f}  CCPR-iso={np.nanmean(i):.4f}")
        params = sum(p.numel() for p in __import__("fusion_vit").FusionSelector(K, img=img).parameters())
        lat = selector_latency_ms(K, img)
        rows.append(dict(img=img, params=params, lat=lat,
                         E_mean=float(np.mean(Es)), E_std=float(np.std(Es)),
                         iso_mean=float(np.mean(Is)), iso_std=float(np.std(Is))))

    print("\n" + "=" * 84)
    print("THUMBNAIL-RESOLUTION ABLATION  (held-out, mean +/- std over seeds)")
    print("-" * 84)
    print(f"{'thumb':>6}{'params':>9}{'ViT ms':>9}{'held-out E':>18}{'CCPR-iso':>18}")
    for r in rows:
        print(f"{r['img']:>6}{r['params']/1e3:>8.1f}k{r['lat']:>9.2f}"
              f"{r['E_mean']:>11.4f} ± {r['E_std']:.4f}"
              f"{r['iso_mean']:>11.4f} ± {r['iso_std']:.4f}")
    print("=" * 84)
    print("Note: 'ViT ms' is the selector forward pass alone on one CPU thread; the full "
          "pipeline\ncost is dominated by the cue bank and is essentially constant across "
          "these sizes.")

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["thumbnail", "selector_params", "vit_ms_1thread",
                    "heldout_E_mean", "heldout_E_std", "CCPRiso_mean", "CCPRiso_std"])
        for r in rows:
            w.writerow([r["img"], r["params"], f"{r['lat']:.2f}",
                        f"{r['E_mean']:.4f}", f"{r['E_std']:.4f}",
                        f"{r['iso_mean']:.4f}", f"{r['iso_std']:.4f}"])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
