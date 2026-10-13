#!/usr/bin/env python3
"""On-device power / thermal / memory bench for the Raspberry Pi 5 -- software only.

No external meter needed. The Pi 5 PMIC exposes per-rail voltage and current through
`vcgencmd pmic_read_adc`, so board power is estimated in software by summing V*I over the
rails; temperature, throttling and ARM clock come from vcgencmd; resident memory from
/proc; and the per-stage latency split (cue bank vs selector vs fuse) is timed directly.

Because the PMIC figure is an on-board rail-sum estimate, not a lab wall-power reading,
report ABSOLUTE watts with that caveat and lead with the RELATIVE energy-per-frame between
methods measured the same way on the same board (robust to absolute calibration).

    # sanity: confirm your firmware exposes the PMIC ADC (Pi 5 + current Raspberry Pi OS)
    vcgencmd pmic_read_adc | head

    # sustained run of the proposed method at 224x224 for 60 s
    python pi_power_bench.py --weights fusion_vit_np.npz --size 224 --duration 60

    # relative energy vs the fixed standard and the real-time baseline
    python pi_power_bench.py --weights fusion_vit_np.npz --size 224 --duration 45 \
        --methods ours,bt709,cv2decolor

Run on the Pi, on mains power, with the machine otherwise idle. Writes pi_power_<size>.csv.
"""
import argparse, csv, os, re, subprocess, time
import numpy as np


# ------------------------------------------------------------------ telemetry
def _vcgencmd(*args):
    try:
        return subprocess.run(["vcgencmd", *args], text=True, capture_output=True,
                              timeout=5).stdout
    except Exception:
        return ""


# NOTE: vcgencmd right-aligns rail names with leading spaces, so the name does NOT start
# at column 0. Allow leading whitespace after ^ (the old r"^(\S+?)" matched 0 lines and made
# power read n/a even though pmic_read_adc was working).
_RAIL = re.compile(r"^\s*(\S+?)_([AV])\s+\w+\(\d+\)=([\d.]+)", re.M)


def read_power_w():
    """Sum V*I over PMIC rails -> approximate board power (W). None if unavailable."""
    out = _vcgencmd("pmic_read_adc")
    if not out.strip():
        return None
    rails = {}
    for name, kind, val in _RAIL.findall(out):
        rails.setdefault(name, {})[kind] = float(val)
    p = sum(d["A"] * d["V"] for d in rails.values() if "A" in d and "V" in d)
    return p if p > 0 else None


def read_temp_c():
    m = re.search(r"temp=([\d.]+)", _vcgencmd("measure_temp"))
    return float(m.group(1)) if m else None


def read_throttled():
    m = re.search(r"throttled=(0x[0-9a-fA-F]+)", _vcgencmd("get_throttled"))
    return int(m.group(1), 16) if m else None


def read_arm_mhz():
    m = re.search(r"=(\d+)", _vcgencmd("measure_clock", "arm"))
    return int(m.group(1)) / 1e6 if m else None


def read_rss_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def decode_throttle(v):
    if v is None:
        return "n/a"
    if v == 0:
        return "0x0 (clean)"
    bits = {0: "under-volt now", 1: "freq-capped now", 2: "throttled now", 3: "soft-temp now",
            16: "under-volt occurred", 17: "freq-cap occurred", 18: "throttle occurred",
            19: "soft-temp occurred"}
    on = [t for b, t in bits.items() if v & (1 << b)]
    return f"{hex(v)} ({'; '.join(on)})"


# ------------------------------------------------------------------ frames + methods
def make_frames(size, image, n=4):
    if image and os.path.exists(image):
        import cv2
        im = cv2.imread(image)[:, :, ::-1]
        im = cv2.resize(im, (size, size)).astype(np.float32) / 255
        return [im.copy() for _ in range(n)]
    rng = np.random.default_rng(1)
    return [rng.random((size, size, 3)).astype(np.float32) for _ in range(n)]


def get_method(name, weights):
    """Return (callable rgb->gray, per_stage_fn or None). 'ours' also exposes the split."""
    if name == "ours":
        from pi_decolor import Decolorizer
        from cues import cue_bank, fuse_cues
        dec = Decolorizer(weights)

        def call(rgb):
            return dec(rgb)

        def split(rgb):
            import cv2
            r = np.asarray(rgb, np.float32)
            t0 = time.perf_counter()
            small = cv2.resize(r, (dec.img, dec.img), interpolation=cv2.INTER_AREA)
            j = int(np.argmax(dec._logits(small)))
            t1 = time.perf_counter()
            cu = cue_bank(r)
            t2 = time.perf_counter()
            fuse_cues(cu, dec.protos[j])
            t3 = time.perf_counter()
            return dict(selector=(t1 - t0) * 1e3, cue_bank=(t2 - t1) * 1e3, fuse=(t3 - t2) * 1e3)

        return call, split
    # published baselines (numpy / opencv only, no torch)
    from decolorizers import get
    dec = get(name)
    return (lambda rgb: dec(rgb)), None


# ------------------------------------------------------------------ bench loop
def bench(name, weights, frames, duration, warmup=10):
    call, split = get_method(name, weights)
    for _ in range(warmup):
        call(frames[0])

    powers, temps, rss = [], [], []
    thr0 = read_throttled()
    n, t_end = 0, time.perf_counter() + duration
    t0 = time.perf_counter()
    next_sample = 0.0
    while time.perf_counter() < t_end:
        call(frames[n % len(frames)])
        n += 1
        now = time.perf_counter() - t0
        if now >= next_sample:                      # sample telemetry ~4x/sec
            p = read_power_w()
            if p is not None:
                powers.append(p)
            t = read_temp_c()
            if t is not None:
                temps.append(t)
            r = read_rss_mb()
            if r is not None:
                rss.append(r)
            next_sample = now + 0.25
    dt = time.perf_counter() - t0
    ms = dt / n * 1e3
    fps = n / dt
    pw = float(np.mean(powers)) if powers else None
    energy = pw * ms if pw is not None else None    # W * ms = mJ per frame
    stage = split(frames[0]) if split else None
    return dict(method=name, size=frames[0].shape[0], frames=n, ms=ms, fps=fps,
                power_w=pw, energy_mj=energy,
                temp_peak=(max(temps) if temps else None),
                temp_end=(temps[-1] if temps else None),
                rss_mb=(max(rss) if rss else None),
                throttled=read_throttled(), thr_start=thr0, stage=stage)


def fmt(v, spec):
    return "n/a" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser(description="Pi 5 software power/thermal/memory bench")
    ap.add_argument("--weights", default="fusion_vit_np.npz")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--duration", type=float, default=60.0, help="seconds per method")
    ap.add_argument("--methods", default="ours",
                    help="comma list: ours,bt709,cv2decolor (baselines give relative energy)")
    ap.add_argument("--image", default=None, help="optional real frame; random if omitted")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if read_power_w() is None:
        print("[warn] `vcgencmd pmic_read_adc` returned no rails -- power will read n/a. "
              "Check firmware/OS; latency, temperature and memory still work.")

    frames = make_frames(a.size, a.image)
    results = []
    for m in [x.strip() for x in a.methods.split(",") if x.strip()]:
        print(f"\n=== {m} @ {a.size}x{a.size} for {a.duration:.0f}s ===")
        try:
            r = bench(m, a.weights, frames, a.duration)
        except Exception as e:
            print(f"   [skip {m}: {type(e).__name__}: {e}]")
            continue
        results.append(r)
        print(f"   {r['fps']:.1f} fps  ({r['ms']:.1f} ms/frame)   "
              f"power={fmt(r['power_w'],'.2f')} W   energy/frame={fmt(r['energy_mj'],'.1f')} mJ")
        print(f"   temp peak={fmt(r['temp_peak'],'.1f')}C end={fmt(r['temp_end'],'.1f')}C   "
              f"RSS={fmt(r['rss_mb'],'.0f')} MB   throttled={decode_throttle(r['throttled'])}")
        if r["stage"]:
            s = r["stage"]
            print(f"   per-stage: cue bank {s['cue_bank']:.1f} ms | selector {s['selector']:.1f} ms "
                  f"| fuse {s['fuse']:.1f} ms")

    if len(results) > 1 and all(r["energy_mj"] for r in results):
        base = next((r for r in results if r["method"] == "bt709"), results[0])
        print("\nrelative energy per frame (lower is better):")
        for r in results:
            print(f"   {r['method']:12} {r['energy_mj']:.1f} mJ   "
                  f"({r['energy_mj']/base['energy_mj']:.2f}x {base['method']})")

    out = a.out or f"pi_power_{a.size}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "size", "frames", "ms_per_frame", "fps", "power_w",
                    "energy_mj_per_frame", "temp_peak_c", "temp_end_c", "rss_mb", "throttled_hex"])
        for r in results:
            w.writerow([r["method"], r["size"], r["frames"], f"{r['ms']:.2f}", f"{r['fps']:.2f}",
                        fmt(r["power_w"], ".3f"), fmt(r["energy_mj"], ".2f"),
                        fmt(r["temp_peak"], ".1f"), fmt(r["temp_end"], ".1f"),
                        fmt(r["rss_mb"], ".0f"),
                        (hex(r["throttled"]) if r["throttled"] is not None else "n/a")])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
