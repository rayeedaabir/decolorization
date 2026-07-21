"""Extract the ORIGINAL colour images from the quality datasets.

Cadik and Color250 both ship with prior methods' grayscale results mixed in with
the originals. Our metrics only need the original colour image (we apply our own
decolorizers to it), so this script copies just the originals into clean, flat
folders that run_quality.py can read directly.

    python prep_datasets.py --cadik data/Cadik --color250 data/Color250/images

Produces:
    data/Cadik_rgb/*.png      (expected 25 originals)
    data/Color250_rgb/*.png   (expected 250 originals)

How the original is identified:
  Cadik    -> in each subfolder, the file named exactly '<subfolder>.ppm'
  Color250 -> files whose name is a plain index (e.g. '17.png'); the results are
              suffixed '<idx>_1'..'<idx>_6' (see the dataset's Readme.txt).
"""
import argparse
import re
from pathlib import Path
from PIL import Image


def _save_rgb(src, dst):
    Image.open(src).convert("RGB").save(dst)


def prep_cadik(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        orig = sub / f"{sub.name}.ppm"
        if not orig.exists():                      # fallback: .ppm whose stem == folder
            cands = [p for p in sub.glob("*.ppm") if p.stem == sub.name]
            orig = cands[0] if cands else None
        if orig and orig.exists():
            _save_rgb(orig, out / f"{sub.name}.png")
            n += 1
        else:
            print(f"  [warn] no original found in {sub.name}")
    print(f"Cadik:    wrote {n} originals -> {out}  (expected 25)")


def prep_color250(images_dir, out):
    images_dir, out = Path(images_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(images_dir.iterdir()):
        if f.is_file() and re.fullmatch(r"\d+", f.stem):   # plain index = original
            _save_rgb(f, out / f"{f.stem}.png")
            n += 1
    print(f"Color250: wrote {n} originals -> {out}  (expected 250)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadik", help="path to the Cadik root (folder of 25 subfolders)")
    ap.add_argument("--color250", help="path to the Color250 'images' folder")
    ap.add_argument("--cadik-out", default="data/Cadik_rgb")
    ap.add_argument("--color250-out", default="data/Color250_rgb")
    a = ap.parse_args()
    if a.cadik:
        prep_cadik(a.cadik, a.cadik_out)
    if a.color250:
        prep_color250(a.color250, a.color250_out)
    if not (a.cadik or a.color250):
        ap.error("give --cadik and/or --color250")
