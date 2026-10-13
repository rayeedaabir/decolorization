"""Oxford Flowers-102 as an ImageFolder -- a downstream task where colour actually matters.

Why add a third downstream dataset
----------------------------------
UC Merced and AID are aerial, chromatically muted and near saturation, so BT.709 and
an adaptive projection are indistinguishable on them; the paper says so honestly, but
it leaves the downstream claim looking empty. Flowers-102 is the opposite case: the
discriminative cues are frequently red/green and yellow/green boundaries at similar
lightness -- the canonical iso-luminant confusion, and precisely the failure mode this
method targets. It is therefore the dataset on which a downstream difference is
*predicted* to appear rather than hoped for.

Reporting rule (unchanged from UCM/AID): a single run is noise. Use seed_sweep.py and
report mean +- std. If the difference lands inside seed variance, that is the result --
say so and keep the framing as "accuracy preserved".

    python prep_flowers.py --out data/Flowers102
    # then the ordinary path:
    python export_gray.py --data-dir data/Flowers102 --decolorizer bt709 --out data/Flowers_bt709
    python export_gray.py --data-dir data/Flowers102 --decolorizer vit:fusion_vit.pt --out data/Flowers_vit
    python seed_sweep.py --data-dirs data/Flowers_bt709 data/Flowers_vit --seeds 0 1 2 --out seeds_flowers.csv

Needs torchvision (already required for the classifier). ~350 MB download, once.
"""
import argparse, os, shutil
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/Flowers102")
    ap.add_argument("--root", default="data/_flowers_raw", help="download cache")
    ap.add_argument("--per-class", type=int, default=0,
                    help="cap images per class (0 = all; 40 keeps runtime near UCM's)")
    a = ap.parse_args()

    from torchvision.datasets import Flowers102

    os.makedirs(a.root, exist_ok=True)
    parts = [Flowers102(root=a.root, split=s, download=True)
             for s in ("train", "val", "test")]
    print(f"downloaded Flowers102: {sum(len(p) for p in parts)} images total")

    by_class = defaultdict(list)
    for p in parts:
        for path, label in zip(p._image_files, p._labels):
            by_class[int(label)].append(path)

    n = 0
    for label, paths in sorted(by_class.items()):
        d = os.path.join(a.out, f"class_{label:03d}")
        os.makedirs(d, exist_ok=True)
        sel = sorted(paths)[:a.per_class] if a.per_class else sorted(paths)
        for src in sel:
            dst = os.path.join(d, os.path.basename(str(src)))
            if not os.path.exists(dst):
                shutil.copy2(str(src), dst)
            n += 1
    print(f"wrote {n} images across {len(by_class)} classes -> {a.out}")
    print("layout is ImageFolder, so export_gray.py / run_classification.py work unchanged.")


if __name__ == "__main__":
    main()
