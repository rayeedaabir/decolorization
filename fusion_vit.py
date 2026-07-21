"""Stage 1 — a lightweight ViT that predicts GLOBAL cue-bank weights per image.

Workflow:
  1) python cue_search.py --images data/Cadik_rgb --out targets_cadik.csv     (oracle targets)
  2) python fusion_vit.py --targets targets_cadik.csv targets_color250.csv --epochs 80

The ViT reads the image and predicts a weight over each of the 5 cues
(R, G, B, MSR, CHROMA); the grayscale is their weighted sum. Held-out E-score is
compared against the fixed BT.709 weighting. Tiny by design (edge-deployable).
"""
import argparse, csv, os
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from cues import cue_bank, fuse_cues, BT709, N_CUES
from metrics import contrast_metrics

IMG = 128


class FusionViT(nn.Module):
    """image -> weights over the cue bank (softmax, so non-negative and summing to 1)."""
    def __init__(self, img=IMG, patch=16, dim=64, depth=3, heads=4, n_cues=N_CUES):
        super().__init__()
        n = (img // patch) ** 2
        self.patch = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True,
                                           activation="gelu", dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_cues))

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], 1) + self.pos
        return F.softmax(self.head(self.enc(x)[:, 0]), dim=-1)      # (B, 5)


class FusionDataset(Dataset):
    def __init__(self, csvs):
        self.rows = []
        for c in csvs:
            with open(c) as fh:
                for r in csv.DictReader(fh):
                    w = [float(r[k]) for k in ("w_R", "w_G", "w_B", "w_MSR", "w_CHROMA")]
                    self.rows.append((r.get("path") or r["image"], w))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, w = self.rows[i]
        img = np.asarray(Image.open(p).convert("RGB").resize((IMG, IMG)), np.float32) / 255
        return torch.from_numpy(img).permute(2, 0, 1), torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def eval_escore(model, rows, device):
    model.eval(); e_vit, e_fix = [], []
    for p, _ in rows:
        rgb = np.asarray(Image.open(p).convert("RGB").resize((256, 256)), np.float64) / 255
        cues = cue_bank(rgb)
        small = np.asarray(Image.open(p).convert("RGB").resize((IMG, IMG)), np.float32) / 255
        x = torch.from_numpy(small).permute(2, 0, 1)[None].to(device)
        w = model(x)[0].cpu().numpy()
        e_vit.append(contrast_metrics(rgb, fuse_cues(cues, w))["E"])
        e_fix.append(contrast_metrics(rgb, fuse_cues(cues, BT709))["E"])
    return float(np.mean(e_vit)), float(np.mean(e_fix))


def load_decolorizer(checkpoint, device=None):
    """Wrap a trained Stage-1 ViT as an rgb->gray callable: --decolorizer vit:CKPT.pt"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionViT().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)); model.eval()

    @torch.no_grad()
    def dec(rgb):
        rgb = np.asarray(rgb, np.float64)
        if rgb.max() > 1: rgb = rgb / 255.0
        small = np.asarray(Image.fromarray((rgb * 255).astype("uint8")).resize((IMG, IMG)),
                           np.float32) / 255
        x = torch.from_numpy(small).permute(2, 0, 1)[None].to(device)
        return fuse_cues(cue_bank(rgb), model(x)[0].cpu().numpy())
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True, help="one or more target CSVs to pool")
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.3); ap.add_argument("--out", default="fusion_vit.pt")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = FusionDataset(a.targets)
    n_val = max(1, int(len(ds) * a.val_frac))
    tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val],
                                           generator=torch.Generator().manual_seed(0))
    model = FusionViT().to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.1f}k   "
          f"{len(ds)} images   device={device}")
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    dl = DataLoader(tr, a.bs, shuffle=True)
    for _ in range(a.epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.mse_loss(model(x), y).backward(); opt.step()
    torch.save(model.state_dict(), a.out)
    e_vit, e_fix = eval_escore(model, [ds.rows[i] for i in va.indices], device)
    print(f"held-out E:  ViT={e_vit:.4f}   fixed BT.709={e_fix:.4f}   "
          f"improvement={(e_vit-e_fix):+.4f} ({(e_vit-e_fix)/e_fix*100:+.2f}%)")
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
