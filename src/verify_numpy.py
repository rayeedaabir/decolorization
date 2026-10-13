"""Confirm the numpy port matches PyTorch  (run on the GPU box, needs torch).

Run this BEFORE trusting anything the Pi produces.

    python verify_numpy.py --images data/Cadik_rgb

WHY THIS COMPARES LOGITS
------------------------
The previous version compared the fused grayscale output. That cannot detect a broken
port: both paths end in the SAME numpy call, `fuse_cues(cue_bank(rgb), protos[j])`, so
the transformer's only influence on the compared image is the integer j. The test was
therefore binary -- identical when the arg-max agreed, and 0.05-0.40 apart when it did
not, never anything in between. A port emitting garbage logits that happened to arg-max
to the same prototype (say, one biased toward prototype 0, which is BT.709 and the
designated fallback) would have printed PASS.

This version compares the 16 raw logits, which exercises the patch embedding, all three
attention blocks, both layer norms and the head. It also reports the preprocessing gap:
the torch path resizes with PIL after a uint8 truncation, the numpy path with
cv2.INTER_AREA on float32, so the two see slightly different 128x128 inputs by design.
That gap is expected and small; what matters is that it does not flip the arg-max.
"""
import argparse, glob, os
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/Cadik_rgb")
    ap.add_argument("--ckpt", default="fusion_vit.pt")
    ap.add_argument("--codebook", default="codebook.npz")
    ap.add_argument("--weights", default="fusion_vit_np.npz")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="max allowed logit difference on IDENTICAL input")
    a = ap.parse_args()

    import torch
    from fusion_vit import FusionSelector, load_state_dict
    from pi_decolor import Decolorizer

    protos = np.load(a.codebook)["protos"]
    tm = FusionSelector(len(protos))
    tm.load_state_dict(load_state_dict(a.ckpt, "cpu")); tm.eval()
    nd = Decolorizer(a.weights)

    files = sorted(f for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.ppm")
                   for f in glob.glob(os.path.join(a.images, e)))[:a.n]
    if not files:
        raise SystemExit(f"no images in {a.images}")

    print(f"{'image':22s}{'logit maxdiff':>15s}{'same input':>12s}"
          f"{'argmax t/n':>12s}   verdict")
    print("-" * 72)
    worst_same, worst_pre, agree, argmax_agree = 0.0, 0.0, 0, 0
    for f in files:
        rgb = np.asarray(Image.open(f).convert("RGB"), np.float64) / 255

        # (a) identical input on both sides -- isolates the ported arithmetic
        same = np.asarray(Image.open(f).convert("RGB").resize((nd.img, nd.img)),
                          np.float32) / 255
        with torch.no_grad():
            lt = tm(torch.from_numpy(same).permute(2, 0, 1)[None]).numpy()[0]
        ln = nd._logits(same)
        d_same = float(np.abs(lt - ln).max())

        # (b) each path's own preprocessing -- the real deployment difference
        import cv2
        small_n = cv2.resize(rgb.astype(np.float32), (nd.img, nd.img),
                             interpolation=cv2.INTER_AREA)
        small_t = np.asarray(Image.fromarray((rgb * 255).astype("uint8"))
                             .resize((nd.img, nd.img)), np.float32) / 255
        with torch.no_grad():
            lt2 = tm(torch.from_numpy(small_t).permute(2, 0, 1)[None]).numpy()[0]
        ln2 = nd._logits(small_n)
        d_pre = float(np.abs(lt2 - ln2).max())
        jt, jn = int(lt2.argmax()), int(ln2.argmax())

        worst_same = max(worst_same, d_same); worst_pre = max(worst_pre, d_pre)
        ok = d_same < a.tol
        agree += ok; argmax_agree += (jt == jn)
        print(f"{os.path.basename(f):22s}{d_same:15.2e}{'OK' if ok else 'MISMATCH':>12s}"
              f"{f'{jt}/{jn}':>12s}   {'' if jt == jn else 'ARGMAX DIFFERS'}")

    print("-" * 72)
    print(f"ported arithmetic : {agree}/{len(files)} within {a.tol:g} on identical input "
          f"(worst {worst_same:.2e})")
    print(f"preprocessing gap : worst logit difference {worst_pre:.2e} "
          f"(expected -- PIL/uint8 vs cv2/float32 resize)")
    print(f"selection         : {argmax_agree}/{len(files)} images pick the same prototype")

    if agree == len(files) and argmax_agree == len(files):
        print("\nPASS - the ported forward pass is numerically faithful and the two")
        print("preprocessing paths do not change which prototype is selected.")
    elif agree == len(files):
        print("\nWARN - the port is faithful, but preprocessing differences flip the")
        print("prototype on some images. The Pi will differ from the GPU box on those.")
        print("Not necessarily wrong, but do not claim identical output.")
    else:
        print("\nFAIL - do not deploy; the ported forward pass disagrees with PyTorch.")


if __name__ == "__main__":
    main()
