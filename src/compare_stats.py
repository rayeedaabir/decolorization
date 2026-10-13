"""Per-image scoring of EVERY method on one benchmark, plus paired significance tests.

Why this exists
---------------
`eval_pregen.py` reports only per-method means, so a 0.0013 gap between rank 1 and
rank 2 cannot be interpreted: it might be decisive or it might be noise. This script
keeps the per-image scores, so the comparison becomes a *paired* one over the same
250 images and the gap gets a p-value and a confidence interval.

It also computes CCPR_iso (metrics.isoluminant_ccpr), which restricts CCPR to
near-iso-luminant pixel pairs -- the failure mode a fixed luma projection cannot
avoid, and the axis on which methods that tie on E can differ sharply.

Usage
-----
    python compare_stats.py --originals data/Color250_rgb --results data/Color250/images \
        --pregen 1,2,3,4,5,6 --methods bt709,vit:fusion_vit.pt \
        --ref vit:fusion_vit.pt --out stats_color250.csv

    # quick smoke test on 10 images first:
    python compare_stats.py ... --limit 10 --out /tmp/smoke.csv

Method specs
------------
  --pregen K[,K...]   published methods read from released outputs '<stem>_K.<ext>'
                      (Color250 ships _1 Lu14, _2 Lu12, _3 CIE Y, _4 Grundland07,
                       _5 Smith08, _6 Color2Gray05)
  --methods NAME[,..] anything decolorizers.get_any understands, applied on the fly
                      (bt601 / bt709 / average / cv2decolor / vit:CKPT / vit2:CKPT)

Outputs
-------
  <out>            per-image CSV: image,method,CCPR,CCFR,E,CCPR_iso,n_iso_pairs
  stdout           ranked summary table + paired tests against --ref
"""
import argparse, csv, glob, os, sys, time
import numpy as np
from PIL import Image
from metrics import (contrast_metrics, isoluminant_ccpr,
                     wilcoxon_signed_rank, bootstrap_ci)
from decolorizers import get_any

PREGEN_NAMES = {"1": "Lu et al. 2014", "2": "Lu et al. 2012", "3": "CIE Y",
                "4": "Grundland & Dodgson 2007", "5": "Smith et al. 2008",
                "6": "Gooch et al. 2005"}
EXTS = ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "ppm")


def find_pregen(results_dir, stem, suffix):
    for e in EXTS:
        hits = glob.glob(os.path.join(results_dir, f"{stem}_{suffix}.{e}"))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--originals", required=True, help="folder of original RGB images")
    ap.add_argument("--results", help="folder holding released grayscale outputs")
    ap.add_argument("--pregen", default="", help="comma list of released-output suffixes")
    ap.add_argument("--methods", default="bt709", help="comma list for get_any()")
    ap.add_argument("--ref", default="", help="method the paired tests compare against")
    ap.add_argument("--out", default="stats.csv")
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    ap.add_argument("--iso-pairs", type=int, default=200_000)
    ap.add_argument("--nr", action="store_true", help="also compute NIQE/BRISQUE (needs pyiqa)")
    ap.add_argument("--time", action="store_true",
                    help="also measure ms/image for the --methods (released outputs cannot be timed)")
    ap.add_argument("--time-size", type=int, default=256)
    ap.add_argument("--time-reps", type=int, default=3)
    ap.add_argument("--time-n", type=int, default=20, help="images used for timing")
    ap.add_argument("--threads", type=int, default=1, help="torch threads; 1 = embedded proxy")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu (default) hides the GPU so latency is a true CPU number")
    ap.add_argument("--min-side", type=int, default=192, help="NIQE upscale floor")
    a = ap.parse_args()

    # MUST happen before get_any() imports torch. The learned decolorizer defaults to
    # CUDA when it is visible, so without this the reported ms/image is a GPU-assisted
    # number wearing a "single CPU thread" label -- which is exactly the claim the
    # edge-device story rests on. Hide the GPU unless it is asked for explicitly.
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    origs = sorted(f for e in EXTS for f in glob.glob(os.path.join(a.originals, "**", f"*.{e}"), recursive=True))
    if not origs:
        raise SystemExit(f"no images in {a.originals}")
    if a.limit:
        origs = origs[:a.limit]

    pregen = [s for s in a.pregen.split(",") if s]
    fly = [s for s in a.methods.split(",") if s]
    if pregen and not a.results:
        raise SystemExit("--pregen needs --results")

    labels, getters = [], {}
    for s in pregen:
        lab = PREGEN_NAMES.get(s, f"pregen_{s}")
        labels.append(lab); getters[lab] = ("pregen", s)
    for m in fly:
        labels.append(m); getters[m] = ("fly", get_any(m))

    niqe = brisque = None
    if a.nr:
        try:
            import torch, pyiqa
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            niqe = pyiqa.create_metric("niqe", device=dev)
            brisque = pyiqa.create_metric("brisque", device=dev)
            print(f"no-reference metrics enabled on {dev}")
        except Exception as e:
            print(f"[warn] --nr requested but unavailable ({type(e).__name__}); skipping")
            a.nr = False

    fields = ["image", "method", "CCPR", "CCFR", "E", "CCPR_iso", "n_iso_pairs"]
    if a.nr:
        fields += ["NIQE", "BRISQUE"]
    fh = open(a.out, "w", newline="", encoding="utf-8")
    w = csv.writer(fh); w.writerow(fields)

    per = {l: {k: [] for k in ("E", "CCPR", "CCFR", "CCPR_iso", "NIQE", "BRISQUE")}
           for l in labels}
    keys = {l: [] for l in labels}
    missing = {l: 0 for l in labels}
    resized = 0
    t0 = time.perf_counter()

    for n, o in enumerate(origs, 1):
        stem = os.path.splitext(os.path.basename(o))[0]
        pil = Image.open(o).convert("RGB")
        rgb = np.asarray(pil, np.float64) / 255.0
        H, W = rgb.shape[:2]
        for lab in labels:
            kind, spec = getters[lab]
            if kind == "pregen":
                p = find_pregen(a.results, stem, spec)
                if p is None:
                    missing[lab] += 1
                    continue
                gi = Image.open(p).convert("L")
                if gi.size != (W, H):
                    gi = gi.resize((W, H)); resized += 1
                gray = np.asarray(gi, np.float64) / 255.0
            else:
                gray = np.clip(np.asarray(spec(rgb), np.float64), 0, 1)

            m = contrast_metrics(rgb, gray)
            iso = isoluminant_ccpr(rgb, gray, n_pairs=a.iso_pairs)
            row = [stem, lab, f"{m['CCPR']:.6f}", f"{m['CCFR']:.6f}", f"{m['E']:.6f}",
                   f"{iso['CCPR_iso']:.6f}", iso["n_iso_pairs"]]
            for k, v in (("CCPR", m["CCPR"]), ("CCFR", m["CCFR"]), ("E", m["E"]),
                         ("CCPR_iso", iso["CCPR_iso"])):
                per[lab][k].append(v)
            if a.nr:
                import torch
                g32 = gray.astype(np.float32)
                if min(g32.shape) < a.min_side:
                    s = a.min_side / min(g32.shape)
                    g32 = np.asarray(Image.fromarray((g32 * 255).astype("uint8")).resize(
                        (max(1, round(g32.shape[1] * s)), max(1, round(g32.shape[0] * s))),
                        Image.LANCZOS), np.float32) / 255
                t = torch.from_numpy(g32)[None, None].repeat(1, 3, 1, 1).to(niqe.device)
                nq, bq = float(niqe(t).item()), float(brisque(t).item())
                per[lab]["NIQE"].append(nq); per[lab]["BRISQUE"].append(bq)
                row += [f"{nq:.4f}", f"{bq:.4f}"]
            keys[lab].append(stem)
            w.writerow(row)
        if n % 25 == 0 or n == len(origs):
            fh.flush()
            el = time.perf_counter() - t0
            print(f"  {n}/{len(origs)} images   {el:6.1f}s   eta {el/n*(len(origs)-n):5.1f}s",
                  flush=True)
    fh.close()

    print(f"\nwrote {a.out}")
    if resized:
        print(f"[note] {resized} released outputs were resized to the original's size")
    for l, c in missing.items():
        if c:
            print(f"[note] {l}: {c} images had no released output and were skipped")

    # ---- restrict every aggregate to the images ALL methods actually scored ----
    # A method whose released output is missing for some images would otherwise be
    # averaged over a different subset than the others, which can invent a gap or
    # flip a rank. The paired tests already intersect; the table must too.
    common = set.intersection(*[set(keys[l]) for l in labels if keys[l]]) if labels else set()
    dropped = {l: len(keys[l]) - len(common) for l in labels}
    if any(dropped.values()):
        print(f"\n[restricting all aggregates to the {len(common)} images every method "
              f"scored; dropped per method: "
              f"{ {l: n for l, n in dropped.items() if n} }]")
        for l in labels:
            sel = [i for i, k in enumerate(keys[l]) if k in common]
            for c in per[l]:
                if per[l][c]:
                    per[l][c] = [per[l][c][i] for i in sel]
            keys[l] = [keys[l][i] for i in sel]

    # ---- cost -----------------------------------------------------------------
    ms = {}
    if a.time:
        try:
            import torch
            torch.set_num_threads(max(1, a.threads))
        except Exception:
            pass
        bench = [np.asarray(Image.open(f).convert("RGB").resize(
                    (a.time_size, a.time_size)), np.float64) / 255
                 for f in origs[:a.time_n]]
        try:
            import torch
            dev_note = (f"{a.device}, torch threads={torch.get_num_threads()}, "
                        f"cuda visible={torch.cuda.is_available()}")
        except Exception:
            dev_note = a.device
        print(f"\ntiming on {len(bench)} images @ {a.time_size}x{a.time_size}, "
              f"{a.time_reps} reps  [{dev_note}]")
        for lab in labels:
            kind, spec = getters[lab]
            if kind != "fly":
                continue                     # released outputs carry no runtime
            spec(bench[0])                   # warm-up
            t = time.perf_counter()
            for _ in range(a.time_reps):
                for im in bench:
                    spec(im)
            ms[lab] = (time.perf_counter() - t) / (a.time_reps * len(bench)) * 1e3
            print(f"  {lab:32s}{ms[lab]:8.2f} ms   {1000/ms[lab]:7.1f} img/s")
        print("  (methods read from released outputs cannot be timed here -- use the")
        print("   runtime reported in their own paper, and mark it as not measured)")

    # ---- summary, ranked by E -------------------------------------------------
    order = sorted(labels, key=lambda l: -np.nanmean(per[l]["E"]) if per[l]["E"] else 1)
    cols = ["CCPR", "CCFR", "E", "CCPR_iso"] + (["NIQE", "BRISQUE"] if a.nr else [])
    width = 34 + 10 * len(cols) + (10 if a.time else 0)
    print("\n" + "=" * width)
    print(f"{'method':32s}" + "".join(f"{c:>10s}" for c in cols)
          + (f"{'ms/img':>10s}" if a.time else ""))
    print("-" * width)
    ref_lab = order[0]
    n_iso = int(np.sum(np.isfinite(per[ref_lab]["CCPR_iso"])))
    n_tot = len(per[ref_lab]["CCPR_iso"])
    if n_iso < n_tot:
        print(f"[CCPR_iso averaged over {n_iso}/{n_tot} images -- the rest have too few")
        print(f" iso-luminant colour pairs for the metric to be defined. That set depends")
        print(f" only on the ORIGINAL image, so it is identical for every method and the")
        print(f" comparison stays paired.]")
    for l in order:
        if not per[l]["E"]:
            continue
        row = f"{l:32s}" + "".join(f"{np.nanmean(per[l][c]):10.4f}" for c in cols)
        if a.time:
            row += f"{ms[l]:10.2f}" if l in ms else f"{'-':>10s}"
        print(row)

    # ---- machine-readable summary, so the figures never retype a number -------
    root, ext = os.path.splitext(a.out)
    spath = f"{root}_summary.csv"
    with open(spath, "w", newline="", encoding="utf-8") as sf:
        sw = csv.writer(sf)
        sw.writerow(["method", "n_images"] + cols + ["ms_per_image", "measured_here"])
        for l in order:
            if not per[l]["E"]:
                continue
            sw.writerow([l, len(per[l]["E"])]
                        + [f"{np.nanmean(per[l][c]):.6f}" for c in cols]
                        + [f"{ms[l]:.3f}" if l in ms else "",
                           "yes" if getters[l][0] == "fly" else "no"])
    print(f"\nwrote {spath}  ->  feeds make_pareto.py and make_radar.py directly")

    # ---- paired tests ---------------------------------------------------------
    if not a.ref:
        return
    if a.ref not in per:
        raise SystemExit(f"--ref '{a.ref}' is not one of: {labels}")
    print(f"\nPaired comparison against '{a.ref}'  (positive = {a.ref} is better)")
    print("=" * 92)
    print(f"{'method':30s}{'dE mean':>10s}{'95% CI':>20s}{'p (Wilcoxon)':>14s}"
          f"{'wins':>8s}{'verdict':>10s}")
    print("-" * 92)
    ref_map = dict(zip(keys[a.ref], per[a.ref]["E"]))
    for l in order:
        if l == a.ref or not per[l]["E"]:
            continue
        other = dict(zip(keys[l], per[l]["E"]))
        common = [k for k in ref_map if k in other]
        d = np.array([ref_map[k] - other[k] for k in common])
        p, nn, _ = wilcoxon_signed_rank(d)
        lo, hi = bootstrap_ci(d)
        wins = float((d > 0).mean()) * 100
        if not np.isfinite(p):
            verdict = "n/a"          # too few paired samples to test
        elif p >= 0.05:
            verdict = "tie"
        else:
            verdict = "BETTER" if d.mean() > 0 else "worse"
        print(f"{l:30s}{d.mean():+10.4f}  [{lo:+.4f},{hi:+.4f}]{p:14.2e}"
              f"{wins:7.0f}%{verdict:>10s}")
    print("-" * 92)
    print("A 'tie' at p>=0.05 is the honest reading of a small mean gap: it means the")
    print("benchmark cannot separate the two methods, not that one of them lost.")


if __name__ == "__main__":
    main()
