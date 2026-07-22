"""Stage 2, step 1 — precompute and cache the three Paper 3 branch outputs.

The branches are frozen and expensive (~26 ms/image), so we compute them ONCE and
reuse them every epoch. Each cached .npz holds:
    branches (3,S,S)     = [L_chaos, L_retinex, 0.5+0.5*C_proj]   (float16)
    rgb      (3,128,128) = the ViT's input                         (float16)
Class folders are mirrored, so the label is the parent folder name.

    python cache_branches.py --data-dir data/UCM/images --out cache/UCM --per-class 40
    python cache_branches.py --data-dir data/AID/data    --out cache/AID --per-class 30
"""
import argparse, os
import numpy as np
from PIL import Image
from decolorizers import branch1_chaos, branch2_retinex, branch3_chroma
from param_search import collect

VIT = 128

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=0)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--size", type=int, default=224, help="resolution the branches are cached at")
    a = ap.parse_args()
    files = collect(a.data_dir, a.recursive or bool(a.per_class), a.per_class)
    if not files:
        raise SystemExit(f"no images under {a.data_dir} (nested? use --per-class K)")
    done = 0
    for f in files:
        rel = os.path.relpath(f, a.data_dir)
        outp = os.path.join(a.out, os.path.splitext(rel)[0] + ".npz")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        done += 1
        if os.path.exists(outp):
            continue
        im = Image.open(f).convert("RGB")
        big = np.asarray(im.resize((a.size, a.size)), np.float64) / 255
        Lc = branch1_chaos(big); Lr = branch2_retinex(Lc); Cp = branch3_chroma(big)
        branches = np.stack([Lc, Lr, 0.5 + 0.5 * Cp]).astype(np.float16)
        rgb = (np.asarray(im.resize((VIT, VIT)), np.float32) / 255).transpose(2, 0, 1).astype(np.float16)
        np.savez_compressed(outp, branches=branches, rgb=rgb)
        if done % 50 == 0:
            print(f"  cached {done}/{len(files)}")
    print(f"cached {done} images -> {a.out}")

if __name__ == "__main__":
    main()
