"""Visual comparison montage of decolorization methods.

    python make_figures.py --demo --out figs/demo_montage.png
    python make_figures.py --images data/Cadik_rgb --n 5 --out figs/cadik_montage.png

Grid: rows = images, columns = [Original | each method]. Grayscale methods are
shown in gray so you can eyeball contrast / isoluminant-boundary preservation.
"""
import argparse, glob, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from decolorizers import get

CELL, PAD, HEADER = 170, 6, 24
LABELS = {"bt601": "bt601", "bt709": "bt709", "average": "average"}


def get_dec(name):
    if name.startswith("vit2:"):
        from stage2 import load_decolorizer as l2
        return l2(name.split(":", 1)[1])
    if name.startswith("vit:"):
        from fusion_vit import load_decolorizer as l1
        return l1(name.split(":", 1)[1])
    return get(name)

def _font():
    try: return ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    except Exception: return ImageFont.load_default()

def _load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((CELL, CELL)), np.float64) / 255

def _demo_images():
    from color import lab_to_srgb
    out = []
    grid = np.zeros((CELL, CELL, 3))
    for i, (r0, c0) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        th = np.deg2rad([20, 110, 200, 290][i])
        grid[r0*CELL//2:(r0+1)*CELL//2, c0*CELL//2:(c0+1)*CELL//2] = \
            lab_to_srgb(np.array([[[60, 40*np.cos(th), 40*np.sin(th)]]]))[0, 0]
    out.append(("isoluminant patches", grid))
    yy, xx = np.mgrid[0:CELL, 0:CELL] - CELL/2
    th, rad = np.arctan2(yy, xx), np.clip(np.hypot(xx, yy)/(CELL/2), 0, 1)
    lab = np.stack([np.full_like(th, 60), 45*rad*np.cos(th), 45*rad*np.sin(th)], -1)
    out.append(("hue wheel (L*=60)", lab_to_srgb(lab)))
    return out

def _cell(arr):
    if arr.ndim == 2: arr = np.stack([arr]*3, -1)
    return Image.fromarray((np.clip(arr, 0, 1)*255).astype(np.uint8)).resize((CELL, CELL))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images"); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--methods", default="bt601,bt709")
    ap.add_argument("--n", type=int, default=5); ap.add_argument("--out", default="figs/montage.png")
    a = ap.parse_args()
    methods = a.methods.split(","); decs = {m: get_dec(m) for m in methods}
    if a.demo:
        samples = _demo_images()
    else:
        files = sorted(sum([glob.glob(os.path.join(a.images, e)) for e in
                            ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.ppm")], []))[:a.n]
        samples = [(os.path.basename(f), _load(f)) for f in files]
    cols = ["Original"] + [LABELS.get(m, m.split(":")[0]) for m in methods]
    ncol, nrow = len(cols), len(samples)
    W = ncol*CELL + (ncol+1)*PAD; H = HEADER + nrow*CELL + (nrow+1)*PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(canvas); f = _font()
    for j, t in enumerate(cols):
        d.text((PAD + j*(CELL+PAD) + 3, 5), t, fill=(0, 0, 0), font=f)
    for i, (name, rgb) in enumerate(samples):
        y = HEADER + PAD + i*(CELL+PAD)
        canvas.paste(_cell(rgb), (PAD, y))
        for j, m in enumerate(methods):
            canvas.paste(_cell(decs[m](rgb)), (PAD + (j+1)*(CELL+PAD), y))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); canvas.save(a.out)
    print("wrote", a.out, canvas.size)

if __name__ == "__main__":
    main()
