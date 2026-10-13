"""Answer one question: can I trust the numbers my current artifacts produce?

Three levels of evidence, cheapest first:

  1. PROVENANCE  -- what each artifact says about how it was built (new artifacts only).
  2. CONSISTENCY -- was fusion_vit.pt actually trained against the codebook.npz sitting
                    beside it? A selector trained on a different codebook still loads,
                    still runs, and is silently wrong: prototype j means something else.
                    For stamped artifacts this is an exact hash check; for older ones it
                    falls back to modification order, which still catches the common case.
  3. REPRODUCTION -- score the current model on a dataset and compare against the numbers
                    the paper reports. This is the only conclusive test.

Run from the repository root:

    python src/check_artifacts.py
    python src/check_artifacts.py --verify --images data/Color250_rgb --dataset color250
"""
import argparse, os, time
import numpy as np
import provenance

ARTIFACTS = ["models/codebook_rgbchroma.npz", "models/fusion_vit_rgbchroma.pt", "models/fusion_vit_np.npz"]

# E-scores reported in the paper (Table I: Color250, native resolution, all 250 images).
REFERENCE = {
    "color250": {"bt709": 0.7828, "vit": 0.8398},
}
TOL = 0.002          # anything inside this is the same model; outside it is not


def mtime(p):
    return os.path.getmtime(p) if os.path.exists(p) else None


def human(t):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "-"


def section(t):
    print(f"\n{t}\n" + "-" * len(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="run the reproduction check")
    ap.add_argument("--images", help="image folder for --verify")
    ap.add_argument("--dataset", choices=sorted(REFERENCE), help="which reference to use")
    ap.add_argument("--ckpt", default="models/fusion_vit_rgbchroma.pt")
    ap.add_argument("--codebook", default="models/codebook_rgbchroma.npz")
    a = ap.parse_args()

    section("1 · Provenance")
    present = [p for p in ARTIFACTS if os.path.exists(p)]
    if not present:
        print("no artifacts found in this folder.")
        return
    for p in ARTIFACTS:
        if os.path.exists(p):
            print(provenance.describe(p))
            print(f"    file mtime     {human(mtime(p))}")
        else:
            print(f"{p}: MISSING")

    section("2 · Consistency  (selector vs codebook)")
    verdict = None
    if not (os.path.exists(a.ckpt) and os.path.exists(a.codebook)):
        print("cannot check -- need both the checkpoint and the codebook.")
    else:
        rec = provenance.read_stamp(a.ckpt)
        cb_now = provenance.file_sha(a.codebook)
        if rec and rec.get("codebook_sha"):
            if rec["codebook_sha"] == cb_now:
                verdict = True
                print(f"OK   selector records codebook_sha={cb_now}, which matches "
                      f"{a.codebook} on disk.")
            else:
                verdict = False
                print(f"MISMATCH  selector was trained against codebook_sha="
                      f"{rec['codebook_sha']}, but {a.codebook} is now {cb_now}.")
                print("          Every prototype index means something different. Any number")
                print("          produced with this pair is invalid. Retrain, or restore the")
                print("          dated codebook copy that matches.")
        else:
            tc, tb = mtime(a.ckpt), mtime(a.codebook)
            print(f"no hash recorded (built before provenance was added), so falling back")
            print(f"to modification order:")
            print(f"    {a.codebook:18s} {human(tb)}")
            print(f"    {a.ckpt:18s} {human(tc)}")
            if tc >= tb:
                print("  PLAUSIBLE  the selector is newer than the codebook, which is the")
                print("             expected order. This is weak evidence, not proof --")
                print("             run --verify to settle it.")
            else:
                verdict = False
                print("  SUSPECT    the codebook is NEWER than the selector, so the codebook")
                print("             was rebuilt after training. The selector's prototype")
                print("             indices no longer refer to these prototypes.")

    if not a.verify:
        section("3 · Reproduction")
        print("not run. To settle it conclusively:")
        print("    python src/check_artifacts.py --verify --images data/Color250_rgb --dataset color250")
        return

    section("3 · Reproduction")
    if not (a.images and a.dataset):
        raise SystemExit("--verify needs --images and --dataset")
    import glob
    from PIL import Image
    from decolorizers import get_any
    from metrics import contrast_metrics

    exts = ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "ppm")
    files = sorted(f for e in exts for f in glob.glob(os.path.join(a.images, f"*.{e}")))
    if not files:
        raise SystemExit(f"no images in {a.images}")
    ref = REFERENCE[a.dataset]
    methods = {"bt709": "bt709", "vit": f"vit:{a.ckpt}:{a.codebook}"}
    print(f"{len(files)} images from {a.images}\n")
    print(f"{'method':10s}{'measured E':>12s}{'paper':>10s}{'delta':>10s}   verdict")
    allok = True
    for key, spec in methods.items():
        dec = get_any(spec)
        vals = []
        for f in files:
            rgb = np.asarray(Image.open(f).convert("RGB"), np.float64) / 255.0
            vals.append(contrast_metrics(rgb, dec(rgb))["E"])
        m = float(np.mean(vals))
        d = m - ref[key]
        ok = abs(d) <= TOL
        allok &= ok
        print(f"{key:10s}{m:12.4f}{ref[key]:10.4f}{d:+10.4f}   "
              f"{'MATCH' if ok else 'DIFFERS'}")
    print()
    if allok:
        print("REPRODUCED. These artifacts are the ones behind the paper's numbers --")
        print("reuse them and do not rebuild.")
    else:
        print("NOT REPRODUCED. The artifacts on disk are not the ones that produced the")
        print("paper's numbers. Rebuild cleanly (see README: Reproduce), then re-derive")
        print("EVERY table from the new run -- do not mix old and new figures.")


if __name__ == "__main__":
    main()
