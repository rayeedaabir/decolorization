"""Run downstream classification over several seeds and report mean ± std.

Single-run accuracies are noisy (BT.709 on UCM spans ~3 pp across seeds), so every
downstream claim must be a mean over seeds with a spread. This automates that and writes
a CSV you can paste straight into the paper.

    python seed_sweep.py --data-dirs data/UCM_bt709 data/UCM_vit --seeds 0 1 2 3 4 --out seeds_ucm.csv
    python seed_sweep.py --data-dirs data/AID_bt709 data/AID_vit --seeds 0 1 2 --out seeds_aid.csv

Each (dir, seed) is one full training run, so budget accordingly (UCM ~8 min each).
Results are appended after every run, so an interrupted sweep keeps what it finished.
"""
import argparse, csv, os
import numpy as np
from decolorizers import get_any
from data import make_splits
import classify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", nargs="+", required=True, help="exported grayscale folders")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", default="seed_sweep.csv")
    a = ap.parse_args()

    dec = get_any("bt601")          # identity on already-grayscale exports
    rows, results = [], {d: [] for d in a.data_dirs}
    new = not os.path.exists(a.out)
    fh = open(a.out, "a", newline=""); w = csv.writer(fh)
    if new:
        w.writerow(["data_dir", "seed", "best_val_acc"]); fh.flush()

    for d in a.data_dirs:
        for s in a.seeds:
            # Seed the TRAINING randomness too, not only the split. Without this the
            # head init, loader shuffling and augmentation are uncontrolled, so a seed
            # is not a repeatable experiment -- rerunning one gives a different answer.
            # (Numbers published before 1 Sep 2026 were produced without this.)
            import torch
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
            np.random.seed(s)
            tr, va, classes = make_splits(d, dec, seed=s)
            print(f"\n=== {os.path.basename(d)}  seed={s}  ({len(classes)} classes) ===")
            _, best = classify.run(tr, va, len(classes), epochs=a.epochs,
                                   batch_size=a.batch_size, lr=a.lr)
            results[d].append(best)
            w.writerow([d, s, f"{best:.4f}"]); fh.flush()
            print(f"  -> {best:.4f}")
    fh.close()

    print("\n" + "=" * 62)
    print(f"{'method (folder)':28s}{'mean':>9s}{'std':>8s}{'min':>8s}{'max':>8s}")
    print("-" * 62)
    for d, v in results.items():
        v = np.array(v)
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        print(f"{os.path.basename(d):28s}{v.mean()*100:8.2f}%{sd*100:7.2f}%"
              f"{v.min()*100:7.2f}%{v.max()*100:7.2f}%")
    names = list(results)
    if len(names) == 2 and all(len(results[n]) > 1 for n in names):
        a_, b_ = np.array(results[names[0]]), np.array(results[names[1]])
        diff = (b_.mean() - a_.mean()) * 100
        se = np.sqrt(a_.var(ddof=1) / len(a_) + b_.var(ddof=1) / len(b_)) * 100
        t = diff / se if se else float("nan")
        print(f"\ndifference: {diff:+.2f} pp   (unpaired SE {se:.2f} pp, "
              f"t ≈ {t:.2f}, n={len(a_)} vs {len(b_)})")
        if not np.isfinite(t) or abs(t) < 2:
            print("|t| < 2 => report as 'comparable', not a win.")
        else:
            print(f"|t| >= 2 => the difference is real at this sample size.")
        print("NOTE: these statistics cover ONLY the runs from THIS invocation, and the")
        print("test is unpaired. If the sweep was split across invocations, or you want")
        print("the paired test (same seed = same split, far more powerful), run:")
        print(f"    python analyze_seeds.py --csv {a.out} --ref {a.data_dirs[0]}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
