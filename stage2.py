"""Stage 2 — per-pixel adaptive fusion of the cue bank, trained on a DOWNSTREAM loss.

Stage 1 predicts one global weight per cue; Stage 2 predicts a weight per cue *per
pixel*, trained end-to-end so the grayscale maximises downstream accuracy directly
rather than a quality proxy.

What keeps it tractable: the cue bank is fixed and precomputable (cache_cues.py), so the
ViT only predicts the weights that combine the cues — and that combination is fully
differentiable, so gradients flow from the classifier loss back into the ViT.

    cues (cached, frozen) --> ViT weight maps --> differentiable fuse --> gray
                                                        |
                                      MobileNetV3 --> CE loss --> backprop to ViT
Needs torch / torchvision.
"""
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models


class DiffFuse(nn.Module):
    """cues:(B,5,H,W) ; w:(B,5,H,W) logits -> gray (B,1,H,W). Fully differentiable."""
    def forward(self, cues, w):
        return (cues * F.softmax(w, dim=1)).sum(1, keepdim=True)


class DenseFusionViT(nn.Module):
    """Lightweight dense ViT: RGB -> per-pixel weight logits over the cue bank."""
    def __init__(self, img=128, patch=16, dim=96, depth=4, heads=4, n_cues=5):
        super().__init__()
        self.p, self.g, self.n = patch, img // patch, n_cues
        self.embed = nn.Conv2d(3, dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, self.g * self.g, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True,
                                           activation="gelu", dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.to_logits = nn.Linear(dim, n_cues * patch * patch)

    def forward(self, x):
        B = x.size(0)
        t = self.enc(self.embed(x).flatten(2).transpose(1, 2) + self.pos)
        y = self.to_logits(t).transpose(1, 2).reshape(B, self.n * self.p * self.p, self.g, self.g)
        return F.pixel_shuffle(y, self.p)                      # (B, n_cues, img, img)


def build_classifier(num_classes, freeze=True):
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    if freeze:
        for p in m.features.parameters():
            p.requires_grad = False
    return m


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_decolorizer(checkpoint, device=None):
    """Wrap a trained Stage-2 ViT as an rgb->gray callable:  --decolorizer vit2:CKPT.pt"""
    from cues import cue_bank
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vit = DenseFusionViT().to(device)
    vit.load_state_dict(torch.load(checkpoint, map_location=device)); vit.eval()
    fuse = DiffFuse()

    @torch.no_grad()
    def dec(rgb):
        rgb = np.asarray(rgb, np.float64)
        if rgb.max() > 1:
            rgb = rgb / 255.0
        cu = torch.from_numpy(cue_bank(rgb).astype(np.float32))[None].to(device)
        small = np.asarray(Image.fromarray((rgb * 255).astype("uint8")).resize((128, 128)),
                           np.float32) / 255
        x = torch.from_numpy(small).permute(2, 0, 1)[None].to(device)
        w = F.interpolate(vit(x), cu.shape[-2:], mode="bilinear", align_corners=False)
        return fuse(cu, w).clamp(0, 1)[0, 0].cpu().numpy()
    return dec


if __name__ == "__main__":
    v = DenseFusionViT()
    print(f"DenseFusionViT params: {sum(p.numel() for p in v.parameters())/1e3:.0f}k")
