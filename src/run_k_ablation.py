#!/usr/bin/env python3
"""Codebook-size (K) ablation -- does K=16 saturate? Justifies the choice.

Reuses your EXACT pipeline: per size it builds a codebook from the given --targets (only
--k changes; BT.709 is always prototype 0, so total size K = k+1) and trains the selector
with fusion_vit.py, then reproduces fusion_vit.py's held-out evaluation (same 0.25 split
from the same seed) and additionally records CCPR-iso, parameter count, and prototypes used.

    # 5-cue (default names; K=16 reuses your main codebook.npz):
    python run_k_ablation.py --train --targets targets_cadik.csv targets_color250.csv targets_ucm.csv targets_aid.csv --sizes 4 8 16 32 64 --seeds 0 1 2 --epochs 60

    # 4-cue (RGB+chroma) -- MUST pass --tag so it builds FRESH codebooks and does NOT reuse
    # the 5-cue codebook_k*.npz / kabl_*.pt files (which is what silently happened before):
    python run_k_ablation.py --train --targets t_rgbchroma_cadik.csv t_rgbchroma_color250.csv t_rgbchroma_ucm.csv t_rgbchroma_aid.csv --tag rgbchroma --sizes 4 8 16 32 64 --seeds 0 1 2 --epochs 60 --out k_ablation_4cue.csv

With --tag <t>, every size is built fresh as codebook_<t>_k<K>.npz and trained as
kabl_<t>_k<K>_s<seed>.pt -- no collision with any earlier run. Run from the harness folder.
"""
import argparse, os, subprocess, sys
import numpy as np
from PIL import Image


def run(cmd):
    print("   $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def codebook_path(K, tag):
    if tag:
        return f"codebook_{tag}_k{K}.npz"
    return "codebook.npz" if K == 16 else f"codebook_k{K}.npz"


def selector_path(K, seed, tag):
    return f"kabl_{tag}_k{K}_s{seed}.pt" if tag else f"kabl_k{K}_s{seed}.pt"


def ensure_codebook(K, targets, size, tag):
    cb = codebook_path(K, tag)
    if os.path.exists(cb):
        print(f"   [skip] {cb}")
        return cb
    if not tag and K == 16:
        sys.exit("[error] untagged K=16 expects your main codebook.npz. Build it first, "
                 "or pass --tag to build a fresh set.")
    if not targets:
        sys.exit(f"[error] {cb} missing and no --targets given.")
    run([sys.executable, "build_codebook.py", "--targets", *targets,
         "--k", str(K - 1), "--size", str(size), "--out", cb])
    return cb


def ensure_selector(K, cb, seed, epochs, tag):
    out = selector_path(K, seed, tag)
    if os.path.exists(out):
        print(f"   [skip] {out}")
        return out
    run([sys.executable, "fusion_vit.py", "--codebook", cb, "--seed", str(seed),
         "--epochs", str(epochs), "--no-stamp", "--out", out])
    return out


def held_out_eval(cb, ckpt, seed, val_frac, device):
    import torch
    from fusion_vit import FusionSelector, load_state_dict, IMG
    from cues import cue_bank, fuse_cues
    from metrics import contrast_metrics, isoluminant_ccpr
    d = np.load(cb, allow_pickle=True)
    protos, paths = d["protos"], [str(p) for p in d["paths"]]
    idx = np.random.default_rng(seed).permutation(len(paths))
    va = idx[:max(1, int(len(paths) * val_frac))]
    model = FusionSelector(len(protos)).to(device)
    model.load_state_dict(load_state_dict(ckpt, device)); model.eval()
    Es, Is, used = [], [], []
    with torch.no_grad():
        for i in va:
            rgb = np.asarray(Image.open(paths[i]).convert("RGB").resize((256, 256)), np.float64) / 255
            cu = cue_bank(rgb)
            small = np.asarray(Image.open(paths[i]).convert("RGB").resize((IMG, IMG)), np.float32) / 255
            x = torch.from_numpy(small).permute(2, 0, 1)[None].to(device)
            j = int(model(x)[0].argmax().item())
            g = fuse_cues(cu, protos[j])
            Es.append(contrast_metrics(rgb, g)["E"]); Is.append(isoluminant_ccpr(rgb, g)["CCPR_iso"])
            used.append(j)
    return np.array(Es), np.array(Is), set(used)


def selector_params(K):
    from fusion_vit import FusionSelector
    return sum(p.numel() for p in FusionSelector(K).parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--targets", nargs="*", default=[])
    ap.add_argument("--tag", default="", help="filename tag; REQUIRED for a non-default cue bank "
                                              "so fresh codebooks/selectors are built (e.g. rgbchroma)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--out", default="k_ablation_table.csv")
    a = ap.parse_args()
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if a.train:
        for K in a.sizes:
            print(f"\n=== prepare K={K} (tag='{a.tag or 'default'}') ===")
            cb = ensure_codebook(K, a.targets, a.size, a.tag)
            for s in a.seeds:
                ensure_selector(K, cb, s, a.epochs, a.tag)

    rows = []
    for K in a.sizes:
        cb = codebook_path(K, a.tag)
        if not os.path.exists(cb):
            print(f"   [miss] {cb} -- skipping K={K}"); continue
        Es, Is, used = [], [], set()
        for s in a.seeds:
            ckpt = selector_path(K, s, a.tag)
            if not os.path.exists(ckpt):
                print(f"   [miss] {ckpt}"); continue
            e, i, u = held_out_eval(cb, ckpt, s, a.val_frac, a.device)
            Es.append(e.mean()); Is.append(np.nanmean(i)); used |= u
            print(f"   K={K:>3} seed {s}:  E={e.mean():.4f}  CCPR-iso={np.nanmean(i):.4f}  prototypes used={len(u)}")
        if Es:
            ceil = float(np.load(cb, allow_pickle=True)["E"].max(1).mean())
            rows.append((K, selector_params(K), ceil, float(np.mean(Es)), float(np.std(Es)),
                         float(np.mean(Is)), float(np.std(Is)), len(used)))

    print("\n" + "=" * 82)
    print(f"CODEBOOK-SIZE (K) ABLATION  tag='{a.tag or 'default'}'  (held-out, mean +/- std; 256x256)")
    print("-" * 82)
    print(f"{'K':>4}{'params':>9}{'ceilE':>9}{'held-out E':>18}{'CCPR-iso':>18}{'#used':>7}")
    for K, p, ce, em, es, im, isd, u in rows:
        print(f"{K:>4}{p/1e3:>8.1f}k{ce:>9.4f}{em:>11.4f} ± {es:.4f}{im:>11.4f} ± {isd:.4f}{u:>7}")
    print("=" * 82)

    import csv
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["K", "selector_params", "codebook_ceiling_E", "heldout_E_mean", "heldout_E_std",
                    "CCPRiso_mean", "CCPRiso_std", "distinct_prototypes_used"])
        for K, p, ce, em, es, im, isd, u in rows:
            w.writerow([K, p, f"{ce:.4f}", f"{em:.4f}", f"{es:.4f}", f"{im:.4f}", f"{isd:.4f}", u])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()