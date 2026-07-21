"""Score PRE-COMPUTED grayscale outputs from published methods — the SOTA comparison.

Color250 ships six published methods' results alongside each original (see its
Readme.txt): originals are '<idx>.png'; results are '<idx>_k.png' with
    _1 Lu et al. 2014      _2 Lu et al. 2012        _3 CIE Y
    _4 Grundland & Dodgson 2007   _5 Smith et al. 2008   _6 Gooch et al. Color2Gray 2005

So you get a published-baseline table without reimplementing anything:

    python eval_pregen.py --originals data/Color250_rgb --results data/Color250/images --suffix 3
    for k in 1 2 3 4 5 6; do python eval_pregen.py ... --suffix $k; done
"""
import argparse, glob, os
import numpy as np
from PIL import Image
from metrics import contrast_metrics

NAMES = {"1": "Lu et al. 2014", "2": "Lu et al. 2012", "3": "CIE Y",
         "4": "Grundland & Dodgson 2007", "5": "Smith et al. 2008",
         "6": "Gooch et al. Color2Gray 2005"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--originals", required=True, help="folder of original RGB images")
    ap.add_argument("--results", required=True, help="folder holding the pre-computed grays")
    ap.add_argument("--suffix", required=True, help="method suffix, e.g. 3 for CIE Y")
    a = ap.parse_args()
    origs = sorted(glob.glob(os.path.join(a.originals, "*.png")))
    agg = {"CCPR": [], "CCFR": [], "E": []}
    missing = 0
    for o in origs:
        stem = os.path.splitext(os.path.basename(o))[0]
        hits = [p for e in ("png", "jpg", "jpeg", "bmp", "tif")
                for p in glob.glob(os.path.join(a.results, f"{stem}_{a.suffix}.{e}"))]
        if not hits:
            missing += 1
            continue
        rgb = np.asarray(Image.open(o).convert("RGB"), np.float64) / 255
        gray = np.asarray(Image.open(hits[0]).convert("L"), np.float64) / 255
        if gray.shape != rgb.shape[:2]:
            gray = np.asarray(Image.open(hits[0]).convert("L").resize(
                (rgb.shape[1], rgb.shape[0])), np.float64) / 255
        m = contrast_metrics(rgb, gray)
        for k in agg:
            agg[k].append(m[k])
    if not agg["E"]:
        raise SystemExit("no matching result files — check --results and --suffix")
    print(f"method = {NAMES.get(a.suffix, 'suffix ' + a.suffix)}   images={len(agg['E'])}"
          + (f"   (missing {missing})" if missing else ""))
    for k in agg:
        print(f"  {k:5s}: {np.mean(agg[k]):.4f}")


if __name__ == "__main__":
    main()
