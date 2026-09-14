"""Stage 1 — a lightweight ViT that SELECTS a cue weighting per image.

Why selection instead of regression: near-optimal weightings are non-unique (many very
different vectors score alike), so regressing them with MSE averages between good
solutions and can score worse than a fixed weighting. Here the ViT instead picks one
entry from a codebook of weightings (built by build_codebook.py), trained with soft
targets from each entry's measured E-score. Prototype 0 is BT.709, so the model always
has a safe fallback.

    python build_codebook.py --targets targets_*.csv --k 15 --out codebook.npz
    python fusion_vit.py --codebook codebook.npz --epochs 60
"""
import argparse, shutil
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from cues import cue_bank, fuse_cues, BT709
from metrics import contrast_metrics
import provenance

IMG = 128


class FusionSelector(nn.Module):
    """image -> logits over the codebook of cue weightings."""
    def __init__(self, n_proto, img=IMG, patch=16, dim=64, depth=3, heads=4):
        super().__init__()
        n = (img // patch) ** 2
        self.patch = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True,
                                           activation="gelu", dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_proto))

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], 1) + self.pos
        return self.head(self.enc(x)[:, 0])          # (B, K) logits


class CodebookDataset(Dataset):
    def __init__(self, paths, E):
        self.paths, self.E = paths, E

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.asarray(Image.open(self.paths[i]).convert("RGB").resize((IMG, IMG)),
                         np.float32) / 255
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(self.E[i])


def load_state_dict(checkpoint, device):
    """Accept both the old bare state_dict and the new {state_dict, meta} format."""
    try:
        obj = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:                       # torch < 1.13 has no weights_only kwarg
        obj = torch.load(checkpoint, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def load_decolorizer(checkpoint, codebook="codebook.npz", device=None):
    """rgb -> gray using the ViT-selected prototype:  --decolorizer vit:fusion_vit.pt"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    protos = np.load(codebook)["protos"]
    model = FusionSelector(len(protos)).to(device)
    model.load_state_dict(load_state_dict(checkpoint, device)); model.eval()

    @torch.no_grad()
    def dec(rgb):
        rgb = np.asarray(rgb, np.float64)
        if rgb.max() > 1:
            rgb = rgb / 255.0
        small = np.asarray(Image.fromarray((rgb * 255).astype("uint8")).resize((IMG, IMG)),
                           np.float32) / 255
        x = torch.from_numpy(small).permute(2, 0, 1)[None].to(device)
        j = int(model(x)[0].argmax().item())
        return fuse_cues(cue_bank(rgb), protos[j])
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook", default="codebook.npz")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--val-frac", type=float, default=0.25); ap.add_argument("--out", default="fusion_vit.pt")
    ap.add_argument("--no-stamp", action="store_true",
                    help="skip the dated archive copy (not recommended)")
    ap.add_argument("--seed", type=int, default=0,
                    help="controls the train/val split and torch init; vary for seed sweeps")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    d = np.load(a.codebook, allow_pickle=True)
    protos, E, paths = d["protos"], d["E"], [str(p) for p in d["paths"]]
    K = len(protos)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    idx = np.random.default_rng(a.seed).permutation(len(paths))
    nval = max(1, int(len(paths) * a.val_frac))
    va_i, tr_i = idx[:nval], idx[nval:]
    tr = CodebookDataset([paths[i] for i in tr_i], E[tr_i])
    model = FusionSelector(K).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.1f}k   "
          f"{len(paths)} images   K={K}   device={device}")
    opt = torch.optim.Adam(model.parameters(), a.lr)
    dl = DataLoader(tr, a.bs, shuffle=True)
    for ep in range(a.epochs):
        model.train()
        for x, e in dl:
            x, e = x.to(device), e.to(device)
            target = F.softmax(e / a.temp, dim=1)          # soft: favour high-E prototypes
            loss = F.kl_div(F.log_softmax(model(x), 1), target, reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1}/{a.epochs}  loss={loss.item():.4f}")
    meta = provenance.stamp(epochs=a.epochs, lr=a.lr, temp=a.temp, bs=a.bs,
                            n_images=len(paths), k=int(K), codebook=a.codebook,
                            codebook_sha=provenance.file_sha(a.codebook))
    torch.save({"state_dict": model.state_dict(), "meta": meta}, a.out)
    if not a.no_stamp:
        archive = provenance.stamped_name(a.out)
        shutil.copyfile(a.out, archive)
        print(f"archived -> {archive}")

    # held-out evaluation on the FULL metric
    model.eval(); e_vit, e_fix = [], []
    best_single = int(E[tr_i].mean(0).argmax())            # best fixed choice from train
    e_single = []
    with torch.no_grad():
        for i in va_i:
            rgb = np.asarray(Image.open(paths[i]).convert("RGB").resize((256, 256)), np.float64) / 255
            cu = cue_bank(rgb)
            small = np.asarray(Image.open(paths[i]).convert("RGB").resize((IMG, IMG)), np.float32) / 255
            x = torch.from_numpy(small).permute(2, 0, 1)[None].to(device)
            j = int(model(x)[0].argmax().item())
            e_vit.append(contrast_metrics(rgb, fuse_cues(cu, protos[j]))["E"])
            e_fix.append(contrast_metrics(rgb, fuse_cues(cu, BT709))["E"])
            e_single.append(contrast_metrics(rgb, fuse_cues(cu, protos[best_single]))["E"])
    v, f, s = np.mean(e_vit), np.mean(e_fix), np.mean(e_single)
    print(f"\nheld-out E   fixed BT.709        : {f:.4f}")
    print(f"held-out E   best single prototype: {s:.4f}  ({(s-f)/f*100:+.2f}% vs BT.709)")
    print(f"held-out E   ViT selection        : {v:.4f}  ({(v-f)/f*100:+.2f}% vs BT.709)")
    print(f"saved -> {a.out}  (use with --decolorizer vit:{a.out})")


if __name__ == "__main__":
    main()
