"""No-reference quality (NIQE, BRISQUE) via pyiqa - compare to Paper 3 Table I.
Lower is better for both.

    pip install pyiqa
    python quality_nr.py --images data/Cadik_rgb    --decolorizer paper3 --ref cadik
    python quality_nr.py --images data/Color250_rgb --decolorizer paper3 --ref color250
    python quality_nr.py --images data/UCM/images --decolorizer paper3 --per-class 20
    python quality_nr.py --images data/AID/data    --decolorizer paper3 --per-class 15

Paper 3 (Proposed): Cadik NIQE 7.34 / BRISQUE 22.11 ; Color250 NIQE 6.23 / BRISQUE 30.38.
NIQE needs images at least ~96 px per side (it works on 96-px blocks); small images are
upscaled to --min-side before scoring so tiny Cadik images don't crash it. pyiqa != the
paper's MATLAB implementation, so treat small gaps as implementation differences.
"""
import argparse
import numpy as np, torch
from PIL import Image
import pyiqa
from decolorizers import get
from cues import collect

PAPER = {"cadik": (7.34, 22.11), "color250": (6.23, 30.38)}


def load_for_iqa(path, min_side):
    """Load RGB in [0,1]; upscale only if the short side is below min_side."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    m = min(w, h)
    if m < min_side:
        s = min_side / m
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return np.asarray(img, np.float64) / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--decolorizer", default="paper3")
    ap.add_argument("--ref", choices=list(PAPER), default=None)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--per-class", type=int, default=0)
    ap.add_argument("--min-side", type=int, default=192, help="upscale images smaller than this")
    a = ap.parse_args()

    files = collect(a.images, a.recursive, a.per_class)
    if not files:
        raise SystemExit(f"no images found under '{a.images}'. Check the path; for nested "
                         f"UCM/AID add --per-class K.")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    niqe = pyiqa.create_metric("niqe", device=dev)
    brisque = pyiqa.create_metric("brisque", device=dev)
    dec = get(a.decolorizer)
    ns, bs, skipped = [], [], 0
    for f in files:
        try:
            rgb = load_for_iqa(f, a.min_side)
            gray = np.clip(dec(rgb), 0, 1).astype(np.float32)
            t = torch.from_numpy(gray)[None, None].repeat(1, 3, 1, 1).to(dev)
            ns.append(niqe(t).item()); bs.append(brisque(t).item())
        except Exception as e:
            skipped += 1
            print(f"  [skip] {f}: {type(e).__name__}")

    if not ns:
        raise SystemExit("all images failed - try a larger --min-side (e.g. 256).")
    N, B = float(np.mean(ns)), float(np.mean(bs))
    print(f"decolorizer={a.decolorizer}  images={len(ns)}" + (f"  (skipped {skipped})" if skipped else ""))
    print(f"  NIQE   : {N:.2f}  (lower better)")
    print(f"  BRISQUE: {B:.2f}  (lower better)")
    if a.ref:
        pn, pb = PAPER[a.ref]
        print(f"  Paper 3 (Proposed, {a.ref}): NIQE {pn}  BRISQUE {pb}")
        print(f"  Delta: NIQE {N-pn:+.2f} ({(N-pn)/pn*100:+.1f}%)  "
              f"BRISQUE {B-pb:+.2f} ({(B-pb)/pb*100:+.1f}%)")


if __name__ == "__main__":
    main()
