#!/usr/bin/env python3
"""Driver for the cue-bank ablation (Comment 4.1): train + fixed-set eval + Table V.

Every configuration is scored on the SAME test images, so the comparison is paired and
does NOT depend on each codebook's internal train/val split (which differs between the
subset codebooks and your original codebook.npz because the image ordering differs).
Prints Table V as mean +/- std over seeds and writes it to CSV.

    python run_cue_ablation.py                 # evaluate models you already trained (default)
    python run_cue_ablation.py --train         # also train any MISSING config first, then evaluate
    python run_cue_ablation.py --iso           # also report CCPR-iso per config (slower)
    python run_cue_ablation.py --paired        # per-image paired Wilcoxon tests between configs (implies --iso)
    python run_cue_ablation.py --seeds 0 1 2 3 4 --originals data/Color250_rgb

Filename conventions (match the manual commands):
    t_<tag>_<dataset>.csv        oracle targets per dataset
    codebook_<tag>.npz           per-subset codebook   (full bank uses your codebook.npz)
    abl_<tag>_s<seed>.pt         trained selector for one seed

Run it from the harness folder (relative paths, no drive letters -- the vit spec is
parsed on ':' so absolute Windows paths like C:\\... would break it).
"""
import argparse, csv, glob, os, subprocess, sys
import numpy as np

# display name, tag, --cues spec, codebook path
CONFIGS = [
    ("RGB",        "rgb",       "R,G,B",        "codebook_rgb.npz"),
    ("RGB+MSR",    "rgbmsr",    "R,G,B,MSR",    "codebook_rgbmsr.npz"),
    ("RGB+CHROMA", "rgbchroma", "R,G,B,CHROMA", "codebook_rgbchroma.npz"),
    ("full",       "full",      "all",          "codebook.npz"),
]
DATASETS = [
    ("cadik",    "data/Cadik_rgb",    []),
    ("color250", "data/Color250_rgb", []),
    ("ucm",      "data/UCM/images",   ["--per-class", "20"]),
    ("aid",      "data/AID/data",     ["--per-class", "15"]),
]
# the scientifically meaningful comparisons for the paired test (A vs B: does the added cue help?)
COMPARISONS = [("RGB+CHROMA", "RGB"), ("RGB+MSR", "RGB"), ("full", "RGB+CHROMA"), ("full", "RGB")]


def run(cmd):
    print("   $", " ".join(cmd)); subprocess.run(cmd, check=True)


def ensure_trained(tag, cues, codebook, seeds, epochs):
    py = [sys.executable]
    if tag != "full":
        for dshort, ddir, extra in DATASETS:
            out = f"t_{tag}_{dshort}.csv"
            if os.path.exists(out):
                print(f"   [skip] {out}"); continue
            run(py + ["cue_search.py", "--images", ddir, "--cues", cues, "--out", out] + extra)
        if os.path.exists(codebook):
            print(f"   [skip] {codebook}")
        else:
            run(py + ["build_codebook.py", "--targets", *sorted(glob.glob(f"t_{tag}_*.csv")),
                      "--k", "15", "--out", codebook])
    elif not os.path.exists(codebook):
        sys.exit(f"[error] the full bank expects your main {codebook}; build it via your "
                 f"normal pipeline before running this driver with --train.")
    for s in seeds:
        out = f"abl_{tag}_s{s}.pt"
        if os.path.exists(out):
            print(f"   [skip] {out}"); continue
        run(py + ["fusion_vit.py", "--codebook", codebook, "--seed", str(s),
                  "--epochs", str(epochs), "--no-stamp", "--out", out])


def load_images(originals):
    from PIL import Image
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.ppm", "*.tif")
    fs = sorted(f for e in exts for f in glob.glob(os.path.join(originals, e)))
    if not fs:
        sys.exit(f"[error] no images under {originals}")
    return [np.asarray(Image.open(f).convert("RGB"), np.float64) / 255.0 for f in fs], len(fs)


def eval_model(model, codebook, imgs, iso):
    """Return (mean_E, mean_iso_or_None, per_image_E, per_image_iso, per_image_n_iso_pairs)."""
    from decolorizers import get_any
    from metrics import contrast_metrics
    if iso:
        from metrics import isoluminant_ccpr
    dec = get_any(f"vit:{model}:{codebook}")
    Es, Is, Ns = [], [], []
    for rgb in imgs:
        g = dec(rgb)
        Es.append(contrast_metrics(rgb, g)["E"])
        if iso:
            r = isoluminant_ccpr(rgb, g)
            Is.append(r["CCPR_iso"]); Ns.append(r.get("n_iso_pairs", 0))
        else:
            Is.append(np.nan); Ns.append(0)
    Es, Is, Ns = np.array(Es), np.array(Is), np.array(Ns, float)
    return float(np.mean(Es)), (float(np.nanmean(Is)) if iso else None), Es, Is, Ns


def paired_tests(per_img, metric_key, label, iso_min=0):
    """Per-image paired Wilcoxon between configs, seeds already averaged per image.

    iso_min>0 (CCPR-iso only): keep images whose iso-luminant pixel-pair count is at
    least iso_min -- i.e. where the metric is actually meaningful. This is a reliability
    filter defined by image content, not by where the method wins."""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    note = f"  [iso-luminant images only: n_iso_pairs >= {iso_min}]" if (iso_min and metric_key == "iso") else ""
    print(f"\n  metric = {label}  (per-image, paired, seeds averaged){note}")
    for a_name, b_name in COMPARISONS:
        if a_name not in per_img or b_name not in per_img:
            continue
        da, db = per_img[a_name][metric_key], per_img[b_name][metric_key]
        m = np.isfinite(da) & np.isfinite(db)
        if iso_min and metric_key == "iso":
            m &= (per_img[a_name].get("n_iso", np.zeros_like(da)) >= iso_min)
        da, db = da[m], db[m]
        if len(da) < 3:
            print(f"   {a_name:12} vs {b_name:12}: too few paired images ({len(da)})"); continue
        diff = da - db
        md = float(np.mean(diff))
        wins = int(np.sum(diff > 0))
        p = None
        if wilcoxon is not None and np.any(diff != 0):
            try:
                p = float(wilcoxon(da, db).pvalue)
            except Exception:
                p = None
        ps = f"p={p:.4f}" if p is not None else "p=n/a"
        star = "  *" if (p is not None and p < 0.05) else ""
        print(f"   {a_name:12} vs {b_name:12}: Δ={md:+.4f}  {ps}  ({wins}/{len(da)} images higher){star}")


def main():
    ap = argparse.ArgumentParser(description="cue-bank ablation driver (train + fixed-set eval + Table V + paired tests)")
    ap.add_argument("--originals", default="data/Color250_rgb",
                    help="fixed test set: every configuration is scored on THESE images")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--train", action="store_true", help="train any missing config before evaluating")
    ap.add_argument("--iso", action="store_true", help="also report CCPR-iso (slower: iso-luminant pairs)")
    ap.add_argument("--paired", action="store_true",
                    help="per-image paired Wilcoxon tests between configs (implies --iso)")
    ap.add_argument("--iso-min", type=int, default=0, dest="iso_min",
                    help="restrict the paired CCPR-iso test to images with >= this many "
                         "iso-luminant pixel-pairs (where the metric is meaningful)")
    ap.add_argument("--out", default="ablation_table.csv")
    a = ap.parse_args()
    if a.paired:
        a.iso = True

    if a.train:
        for name, tag, cues, cb in CONFIGS:
            print(f"\n=== train {name} ===")
            ensure_trained(tag, cues, cb, a.seeds, a.epochs)

    print(f"\nloading fixed test set: {a.originals}")
    imgs, n = load_images(a.originals)
    print(f"   {n} images -- every configuration is scored on these")
    np.random.seed(0)                       # stabilise the metric's pair sampling across configs

    rows = []
    per_img = {}                            # name -> {"E": per-image array (seeds averaged), "iso": ...}
    for name, tag, cues, cb in CONFIGS:
        Es, Is = [], []
        pe, pi, pn = [], [], []             # per-seed per-image arrays (E, iso, n_iso_pairs)
        for s in a.seeds:
            model = f"abl_{tag}_s{s}.pt"
            if not os.path.exists(model):
                print(f"   [miss] {model}"); continue
            if not os.path.exists(cb):
                print(f"   [miss] {cb} (needed for {name})"); break
            e, i, e_arr, i_arr, n_arr = eval_model(model, cb, imgs, a.iso)
            Es.append(e); pe.append(e_arr)
            if i is not None:
                Is.append(i); pi.append(i_arr); pn.append(n_arr)
            print(f"   {name:12} seed {s}:  E={e:.4f}" + (f"   CCPR-iso={i:.4f}" if i is not None else ""))
        if Es:
            rows.append((name, len(Es), float(np.mean(Es)), float(np.std(Es)),
                         float(np.mean(Is)) if Is else float("nan"),
                         float(np.std(Is)) if Is else float("nan")))
            if a.paired and pe:
                per_img[name] = {"E": np.mean(np.vstack(pe), axis=0),
                                 "iso": np.mean(np.vstack(pi), axis=0) if pi else np.full(n, np.nan),
                                 "n_iso": np.mean(np.vstack(pn), axis=0) if pn else np.zeros(n)}

    tag = os.path.basename(a.originals.rstrip("/\\")) or a.originals
    print("\n" + "=" * 66)
    print(f"TABLE V  --  cue-bank ablation on {tag}  (mean +/- std over seeds)")
    print("-" * 66)
    hdr = f"{'configuration':14}{'seeds':>6}{'E':>18}"
    if a.iso:
        hdr += f"{'CCPR-iso':>20}"
    print(hdr)
    for name, ns, em, es, im, isd in rows:
        line = f"{name:14}{ns:>6}   {em:.4f} ± {es:.4f}"
        if a.iso:
            line += f"    {im:.4f} ± {isd:.4f}"
        print(line)
    print("=" * 66)

    if a.paired and per_img:
        print("\n" + "=" * 66)
        print(f"PAIRED TESTS on {tag}  (Wilcoxon signed-rank, * = p<0.05)")
        print("=" * 66)
        # show the iso-luminant-content distribution so a sensible --iso-min can be picked
        ref = next(iter(per_img.values())).get("n_iso")
        if ref is not None and np.any(ref > 0):
            qs = np.percentile(ref[np.isfinite(ref)], [50, 75, 90, 95])
            print(f"  n_iso_pairs per image  median={qs[0]:.0f}  p75={qs[1]:.0f}  "
                  f"p90={qs[2]:.0f}  p95={qs[3]:.0f}   (use --iso-min to focus the CCPR-iso test)")
        paired_tests(per_img, "iso", "CCPR-iso", iso_min=a.iso_min)
        paired_tests(per_img, "E", "E")
        print("=" * 66)

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["configuration", "n_seeds", "E_mean", "E_std"] + (["CCPRiso_mean", "CCPRiso_std"] if a.iso else [])
        w.writerow(head)
        for name, ns, em, es, im, isd in rows:
            r = [name, ns, f"{em:.4f}", f"{es:.4f}"] + ([f"{im:.4f}", f"{isd:.4f}"] if a.iso else [])
            w.writerow(r)
    print(f"wrote {a.out}")
    if not rows:
        print("\n(no trained models found -- run your seed trainings first, or pass --train to train them here)")


if __name__ == "__main__":
    main()
