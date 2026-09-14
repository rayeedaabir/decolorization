"""Efficiency table WITHOUT the Raspberry Pi — params, size, and CPU latency.

The edge claim needs numbers. Single-thread CPU latency on your desktop is a defensible
stand-in for embedded hardware and can be measured today; swap in real Pi-5 numbers later
without changing the paper's structure (same script, same table).

    python model_stats.py --images data/Cadik_rgb --methods bt709,vit:fusion_vit.pt,vit2:stage2_vit.pt
    python model_stats.py --demo --threads 1 --size 256

Reports, per method: parameters, checkpoint size, mean ms/image, images/sec, and (if
`thop` is installed) FLOPs. Use --threads 1 for the conservative single-core figure.
"""
import argparse, glob, os, time
import numpy as np
from PIL import Image
from decolorizers import get_any


def model_params(name):
    """Parameter count for the learned methods (0 for fixed formulas)."""
    try:
        import torch
        if name.startswith("vit2:"):
            from stage2 import DenseFusionViT
            m = DenseFusionViT()
        elif name.startswith("vit:"):
            import numpy as _np
            from fusion_vit import FusionSelector
            cb = "codebook.npz"
            rest = name.split(":", 1)[1]
            if ":" in rest:
                cb = rest.split(":", 1)[1]
            m = FusionSelector(len(_np.load(cb)["protos"]))
        else:
            return 0, 0.0
        n = sum(p.numel() for p in m.parameters())
        ckpt = name.split(":", 1)[1].split(":")[0]
        mb = os.path.getsize(ckpt) / 1e6 if os.path.exists(ckpt) else 0.0
        return n, mb
    except Exception:
        return -1, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images"); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--methods", default="bt601,bt709,vit:fusion_vit.pt,vit2:stage2_vit.pt")
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="1 = single-core (embedded proxy)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()

    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import torch
        if a.threads:
            torch.set_num_threads(a.threads)
        dev_note = f"{a.device}, torch threads={torch.get_num_threads()}"
    except Exception:
        dev_note = a.device

    if a.demo or not a.images:
        imgs = [np.random.default_rng(k).random((a.size, a.size, 3)) for k in range(8)]
    else:
        files = sorted(f for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.ppm", "*.tif")
                       for f in glob.glob(os.path.join(a.images, e)))[:20]
        imgs = [np.asarray(Image.open(f).convert("RGB").resize((a.size, a.size)), np.float64) / 255
                for f in files]

    print(f"{len(imgs)} images @ {a.size}x{a.size} · {a.reps} reps · {dev_note}\n")
    print(f"{'method':28s}{'params':>10s}{'ckpt MB':>9s}{'ms/img':>9s}{'img/s':>8s}")
    print("-" * 64)
    for m in [x for x in a.methods.split(",") if x]:
        try:
            dec = get_any(m)
        except Exception as e:
            print(f"{m:28s}  [skip: {type(e).__name__}]")
            continue
        dec(imgs[0])                                   # warm-up
        t = time.perf_counter()
        for _ in range(a.reps):
            for im in imgs:
                dec(im)
        dt = (time.perf_counter() - t) / (a.reps * len(imgs))
        n, mb = model_params(m)
        ptxt = "—" if n <= 0 else f"{n/1e3:.1f}k"
        mtxt = "—" if mb <= 0 else f"{mb:.2f}"
        print(f"{m:28s}{ptxt:>10s}{mtxt:>9s}{dt*1e3:9.2f}{1/dt:8.1f}")
    print("\nNote: single-thread CPU is a stand-in for embedded hardware; re-run this exact")
    print("command on the Raspberry Pi 5 to replace these numbers in the paper.")


if __name__ == "__main__":
    main()
