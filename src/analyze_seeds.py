"""Correct statistics for a seed sweep, across ALL runs in the CSV.

Why this is not what seed_sweep.py prints
-----------------------------------------
`seed_sweep.py` reports only the runs from the invocation that just finished. Split a
sweep across two invocations -- seeds 0-2, then 3-4 -- and its t-test sees n=2 per arm
and produces a meaningless number (a t of -30 was observed this way). This reads the
whole CSV instead.

It also uses a PAIRED test, which is the right one here: a given seed fixes the
train/val split via make_splits, so every method sees the identical split at that seed.
Pairing removes split-to-split variance and is far more powerful than the two-sample
test seed_sweep.py prints.

    python analyze_seeds.py --csv seeds_flowers.csv --ref data/Flowers_bt709
"""
import argparse, csv, math
from collections import defaultdict
import numpy as np


def _t_sf(t, df):
    """Two-sided p for Student's t, via the incomplete beta -- no scipy."""
    t, df = abs(float(t)), float(df)
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    def betacf(a, b, x, itmax=200, eps=3e-12):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        h = d
        for m in range(1, itmax + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1.0 + aa / c; c = c if abs(c) > 1e-30 else 1e-30
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1.0 + aa / c; c = c if abs(c) > 1e-30 else 1e-30
            de = d * c; h *= de
            if abs(de - 1.0) < eps:
                break
        return h

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1.0) / (a + b + 2.0):
        ib = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) * betacf(a, b, x) / a
    else:
        ib = 1.0 - math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) * betacf(b, a, 1 - x) / b
    return max(0.0, min(1.0, ib))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ref", required=True, help="baseline data_dir, exactly as in the CSV")
    a = ap.parse_args()

    by = defaultdict(dict)
    for r in csv.DictReader(open(a.csv, newline="", encoding="utf-8")):
        by[r["data_dir"]][int(r["seed"])] = float(r["best_val_acc"]) * 100
    if a.ref not in by:
        raise SystemExit(f"--ref '{a.ref}' not in the CSV. Have: {sorted(by)}")

    print(f"{'method':26s}{'n':>4s}{'mean':>9s}{'std':>8s}{'min':>8s}{'max':>8s}")
    print("-" * 63)
    for d in sorted(by, key=lambda k: -np.mean(list(by[k].values()))):
        v = np.array(list(by[d].values()))
        print(f"{d:26s}{len(v):4d}{v.mean():8.2f}%{v.std(ddof=1):7.2f}%"
              f"{v.min():7.2f}%{v.max():7.2f}%")

    print(f"\nPaired against {a.ref}  (same seed = same train/val split)")
    print("=" * 78)
    print(f"{'method':26s}{'diff pp':>9s}{'SE':>7s}{'t':>8s}{'p':>10s}{'seeds':>7s}  verdict")
    print("-" * 78)
    for d in sorted(by):
        if d == a.ref:
            continue
        seeds = sorted(set(by[d]) & set(by[a.ref]))
        if len(seeds) < 2:
            print(f"{d:26s}  only {len(seeds)} shared seed(s) -- cannot test")
            continue
        diff = np.array([by[d][s] - by[a.ref][s] for s in seeds])
        n = len(diff)
        se = diff.std(ddof=1) / math.sqrt(n)
        t = diff.mean() / se if se else float("nan")
        p = _t_sf(t, n - 1) if se else float("nan")
        v = ("comparable" if (not np.isfinite(p) or p >= 0.05)
             else ("BETTER" if diff.mean() > 0 else "WORSE"))
        print(f"{d:26s}{diff.mean():+9.2f}{se:7.2f}{t:8.2f}{p:10.4f}{n:7d}  {v}")
        wins = int((diff > 0).sum())
        print(f"{'':26s}per-seed: " + "  ".join(f"{s}:{x:+.2f}" for s, x in zip(seeds, diff))
              + f"   ({wins}/{n} favour this method)")
    print("\nPaired t on n seeds has n-1 df. With 5 seeds, |t| > 2.78 is p < 0.05.")


if __name__ == "__main__":
    main()
