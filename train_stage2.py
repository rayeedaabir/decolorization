"""Stage 2, step 2 — train the per-pixel fusion ViT on the DOWNSTREAM loss.

The ViT predicts per-pixel weights over the cached cue bank; the fused gray
goes into MobileNetV3 (backbone frozen, head trainable) and the classification loss
backpropagates into the ViT. An anchor term keeps the gray close to the fixed BT.709 weighting so it stays a sensible image rather than a degenerate one.

    python train_stage2.py --cache cache/UCM --epochs 20 --out stage2_vit.pt

For the FAIR downstream number afterwards, export grays and re-run the standard
pipeline:  export_gray.py --decolorizer vit2:stage2_vit.pt  ->  run_classification.py
"""
import argparse, glob, os
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from stage2 import DenseFusionViT, DiffFuse, build_classifier, _MEAN, _STD


class CacheDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
        if not self.files:
            raise SystemExit(f"no .npz under {root} — run cache_cues.py first")
        self.classes = sorted({os.path.basename(os.path.dirname(f)) for f in self.files})
        self.cls = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        rgb = torch.from_numpy(d["rgb"].astype(np.float32))
        br = torch.from_numpy(d["cues"].astype(np.float32))
        y = self.cls[os.path.basename(os.path.dirname(self.files[i]))]
        return rgb, br, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anchor", type=float, default=0.1, help="weight on the 'stay near BT.709' anchor")
    ap.add_argument("--val-frac", type=float, default=0.2); ap.add_argument("--out", default="stage2_vit.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CacheDataset(a.cache)
    nval = max(1, int(len(ds) * a.val_frac))
    tr, va = torch.utils.data.random_split(ds, [len(ds) - nval, nval],
                                           generator=torch.Generator().manual_seed(0))
    dl = DataLoader(tr, a.bs, shuffle=True, num_workers=2)
    vl = DataLoader(va, a.bs, shuffle=False, num_workers=2)
    vit = DenseFusionViT().to(dev); fuse = DiffFuse().to(dev)
    clf = build_classifier(len(ds.classes), freeze=True).to(dev)
    opt = torch.optim.Adam(list(vit.parameters()) +
                           [p for p in clf.parameters() if p.requires_grad], a.lr)
    mean, std = _MEAN.to(dev), _STD.to(dev)
    print(f"ViT {sum(p.numel() for p in vit.parameters())/1e3:.0f}k params · "
          f"{len(ds)} imgs · {len(ds.classes)} classes · {dev}")

    def forward(rgb, br):
        w = F.interpolate(vit(rgb), br.shape[-2:], mode="bilinear", align_corners=False)
        return fuse(br, w).clamp(0, 1)

    best = 0.0
    for ep in range(a.epochs):
        vit.train(); clf.train()
        for rgb, br, y in dl:
            rgb, br, y = rgb.to(dev), br.to(dev), y.to(dev)
            gray = forward(rgb, br)
            loss = F.cross_entropy(clf((gray.repeat(1, 3, 1, 1) - mean) / std), y)
            if a.anchor > 0:                       # fixed BT.709 weighting as the anchor
                ref = (0.2126 * br[:, 0:1] + 0.7152 * br[:, 1:2] + 0.0722 * br[:, 2:3]).clamp(0, 1)
                loss = loss + a.anchor * F.l1_loss(gray, ref)
            opt.zero_grad(); loss.backward(); opt.step()
        vit.eval(); clf.eval(); correct = tot = 0
        with torch.no_grad():
            for rgb, br, y in vl:
                rgb, br, y = rgb.to(dev), br.to(dev), y.to(dev)
                gray = forward(rgb, br)
                correct += (clf((gray.repeat(1, 3, 1, 1) - mean) / std).argmax(1) == y).sum().item()
                tot += y.numel()
        acc = correct / max(tot, 1); best = max(best, acc)
        print(f"epoch {ep+1}/{a.epochs}  val_acc={acc:.4f}  best={best:.4f}")
    torch.save(vit.state_dict(), a.out)
    print(f"saved -> {a.out}   (best val_acc {best:.4f})")


if __name__ == "__main__":
    main()
