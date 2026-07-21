"""Decolorize a whole dataset once and save the grays, mirroring the class folders.
Much faster for downstream training (decolorize once, not every epoch), and the way
to classify with the ViT method.

    python export_gray.py --data-dir data/UCM/images --decolorizer vit:fusion_vit.pt --out data/UCM_vit
    python export_gray.py --data-dir data/UCM/images --decolorizer paper3            --out data/UCM_p3
    python run_classification.py --data-dir data/UCM_vit --decolorizer bt601   # bt601 = identity on gray
    python run_classification.py --data-dir data/UCM_p3  --decolorizer bt601
"""
import argparse, os
import numpy as np
from PIL import Image
from decolorizers import get

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".ppm")

def get_dec(name):
    if name.startswith("vit2:"):
        from stage2 import load_decolorizer as load2
        return load2(name.split(":", 1)[1])
    if name.startswith("vit:"):
        from fusion_vit import load_decolorizer
        return load_decolorizer(name.split(":", 1)[1])
    return get(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--decolorizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dec = get_dec(a.decolorizer); n = 0
    for root, _, files in os.walk(a.data_dir):
        rel = os.path.relpath(root, a.data_dir)
        outdir = os.path.join(a.out, rel); os.makedirs(outdir, exist_ok=True)
        for fn in files:
            if fn.lower().endswith(EXTS):
                rgb = np.asarray(Image.open(os.path.join(root, fn)).convert("RGB"), np.float64) / 255
                gray = np.clip(dec(rgb), 0, 1)
                Image.fromarray((gray * 255).astype("uint8"), mode="L").save(
                    os.path.join(outdir, os.path.splitext(fn)[0] + ".png"))
                n += 1
    print(f"exported {n} grayscale images -> {a.out}")

if __name__ == "__main__":
    main()
