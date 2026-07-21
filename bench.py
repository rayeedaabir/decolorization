"""Time each decolorizer (ms/image) - the speed axis for comparing methods.

    python bench.py --images data/Cadik_rgb
    python bench.py --demo

The downstream classifier is identical across methods, so decolorization latency
is the differentiator (and the number to compare against the future ViT).
"""
import argparse, glob, os, time
import numpy as np
from PIL import Image
from decolorizers import get

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images"); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--methods", default="bt601,bt709,paper3_b1,paper3_b12,paper3_b123,paper3")
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.demo:
        imgs = [np.random.default_rng(k).random((a.size, a.size, 3)) for k in range(8)]
    else:
        files = sorted(sum([glob.glob(os.path.join(a.images, e)) for e in
                            ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.ppm")], []))
        imgs = [np.asarray(Image.open(f).convert("RGB").resize((a.size, a.size)), np.float64)/255
                for f in files[:20]]
    print(f"{len(imgs)} images @ {a.size}x{a.size}, {a.reps} reps\n{'method':14s}{'ms/img':>9s}{'img/s':>9s}")
    for m in a.methods.split(","):
        dec = get(m); dec(imgs[0])
        t = time.perf_counter()
        for _ in range(a.reps):
            for im in imgs: dec(im)
        dt = (time.perf_counter()-t)/(a.reps*len(imgs))
        print(f"{m:14s}{dt*1e3:9.2f}{1/dt:9.1f}")

if __name__ == "__main__":
    main()
