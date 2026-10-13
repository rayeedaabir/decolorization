#!/usr/bin/env python3
"""Table II helper: per-image-mean relative E-gain of the proposed method over BT.709,
computed from a compare_stats.py per-image CSV (native resolution -- same protocol as Table I).

This is the exact quantity Table II reports ("per-image mean of the relative E-gain over
BT.709"). It also prints the ratio-of-aggregate-means for the Table II note's cross-check.

    python table2_gain.py stats_color250_4cue.csv
    python table2_gain.py stats_cadik_4cue.csv stats_ucm_4cue.csv stats_aid_4cue.csv

The proposed method is auto-detected as the row whose method name starts with "vit"; BT.709
is the row named "bt709". Pass --prop / --bt709 to override.
"""
import argparse, csv, collections, math
import numpy as np


def load(path, prop_key, bt_key):
    E = collections.defaultdict(dict)
    methods = set()
    for r in csv.DictReader(open(path)):
        m = r["method"]; methods.add(m)
        try:
            v = float(r["E"])
            if not math.isnan(v):
                E[m][r["image"]] = v
        except (KeyError, ValueError):
            pass
    prop = prop_key or next((m for m in methods if m.lower().startswith("vit")), None)
    bt = bt_key if bt_key in E else ("bt709" if "bt709" in E else None)
    if prop is None or bt is None:
        raise SystemExit(f"could not find proposed/bt709 in {path}; methods = {sorted(methods)}")
    return E[prop], E[bt], prop, bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--prop", default="")
    ap.add_argument("--bt709", default="bt709")
    a = ap.parse_args()
    print(f"{'dataset':28}{'n':>6}{'per-image-mean ΔE':>20}{'ratio-of-means ΔE':>20}")
    print("-" * 74)
    for path in a.csvs:
        Ep, Eb, prop, bt = load(path, a.prop, a.bt709)
        imgs = sorted(set(Ep) & set(Eb))
        p = np.array([Ep[i] for i in imgs]); b = np.array([Eb[i] for i in imgs])
        pim = float(((p - b) / b * 100).mean())          # per-image mean relative gain (Table II)
        rom = float(100 * (p.mean() - b.mean()) / b.mean())   # ratio of aggregate means (note)
        name = path.replace("stats_", "").replace("_4cue.csv", "").replace(".csv", "")
        print(f"{name:28}{len(imgs):>6}{pim:>+19.2f}%{rom:>+19.2f}%")


if __name__ == "__main__":
    main()
