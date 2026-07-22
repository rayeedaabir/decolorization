"""Downstream classification pipeline (MobileNetV3-Large).

Protocol: 224x224 input, grayscale replicated to 3 channels,
Adam @ lr 5e-4, cross-entropy, 15 epochs. This is the 'does the gray help the
real task?' measurement. Requires torch + torchvision.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models


def build_model(num_classes, pretrained=True):
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v3_large(weights=weights)
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, num_classes)
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def run(train_ds, val_ds, num_classes, epochs=15, batch_size=16, lr=5e-4,
        num_workers=4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    tl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    vl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers)

    best = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        acc = evaluate(model, vl, device)
        best = max(best, acc)
        print(f"epoch {ep + 1:2d}/{epochs}  val_acc={acc:.4f}  best={best:.4f}")
    return model, best
