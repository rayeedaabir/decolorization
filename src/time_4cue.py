#!/usr/bin/env python3
"""Single-thread CPU latency of the TRUE 4-cue (RGB + opponent-chroma) pipeline, for the
cost table after dropping MSR. The RGB+chroma codebook already carries MSR weight = 0 on
every prototype, so the grayscale output is identical whether or not MSR is computed; this
script simply skips the (expensive) multi-scale Retinex cue so the timing reflects the real
4-cue cost. Also prints the per-stage split and a sanity check that the 4-cue output equals
the 5-cue-with-MSR-zeroed output.

    python time_4cue.py --ckpt fusion_vit_rgbchroma.pt --codebook codebook_rgbchroma.npz --threads 1 --size 256
"""
import argparse, os, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="fusion_vit_rgbchroma.pt")
    ap.add_argument("--codebook", default="codebook_rgbchroma.npz")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--images", default=None, help="optional image dir; random frames if omitted")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""     # single-thread CPU only
    import torch
    torch.set_num_threads(a.threads)
    from PIL import Image
    from cues import _to_float, opponent_chroma, fuse_cues, cue_bank
    from fusion_vit import FusionSelector, load_state_dict, IMG

    protos = np.load(a.codebook)["protos"]                 # (K,5), MSR column (idx 3) == 0
    protos4 = protos[:, [0, 1, 2, 4]]                      # drop MSR -> (K,4)
    model = FusionSelector(len(protos)); model.load_state_dict(load_state_dict(a.ckpt, "cpu")); model.eval()

    def cue_bank4(rgb):
        rgb = _to_float(rgb)
        return np.stack([rgb[..., 0], rgb[..., 1], rgb[..., 2], opponent_chroma(rgb)])

    @torch.no_grad()
    def dec4(rgb):
        small = np.asarray(Image.fromarray((np.clip(_to_float(rgb), 0, 1) * 255).astype("uint8"))
                           .resize((IMG, IMG)), np.float32) / 255
        x = torch.from_numpy(small).permute(2, 0, 1)[None]
        j = int(model(x)[0].argmax().item())
        return fuse_cues(cue_bank4(rgb), protos4[j])

    if a.images and os.path.isdir(a.images):
        import glob
        fs = sorted(f for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp") for f in glob.glob(os.path.join(a.images, e)))[:8]
        imgs = [np.asarray(Image.open(f).convert("RGB").resize((a.size, a.size)), np.float64) / 255 for f in fs]
    else:
        imgs = [np.random.default_rng(k).random((a.size, a.size, 3)) for k in range(8)]

    # sanity: 4-cue output == 5-cue output with MSR-zeroed prototype
    r = imgs[0]
    cu5 = cue_bank(r)
    g5 = fuse_cues(cu5, protos[int(model(torch.from_numpy(
        np.asarray(Image.fromarray((r * 255).astype("uint8")).resize((IMG, IMG)), np.float32) / 255
    ).permute(2, 0, 1)[None])[0].argmax().item())])
    g4 = dec4(r)
    print(f"sanity |4cue - 5cue(MSR=0)| max = {np.abs(g4 - g5).max():.2e}  (should be ~0)")

    dec4(imgs[0])                                          # warm-up
    t = time.perf_counter()
    for _ in range(a.reps):
        for im in imgs:
            dec4(im)
    dt = (time.perf_counter() - t) / (a.reps * len(imgs))

    # per-stage split
    r = imgs[0]
    t0 = time.perf_counter()
    small = np.asarray(Image.fromarray((np.clip(_to_float(r), 0, 1) * 255).astype("uint8")).resize((IMG, IMG)), np.float32) / 255
    j = int(model(torch.from_numpy(small).permute(2, 0, 1)[None])[0].argmax().item()); t1 = time.perf_counter()
    cu = cue_bank4(r); t2 = time.perf_counter()
    fuse_cues(cu, protos4[j]); t3 = time.perf_counter()

    n = sum(p.numel() for p in model.parameters())
    print(f"\n4-cue (RGB+chroma) @ {a.size}x{a.size}, {a.threads} thread(s): "
          f"{dt*1e3:.2f} ms/frame  ({1/dt:.1f} fps)   params={n/1e3:.1f}k size={n*4/1e6:.2f}MB")
    print(f"  per-stage: selector {(t1-t0)*1e3:.1f} ms | cue bank {(t2-t1)*1e3:.1f} ms | fuse {(t3-t2)*1e3:.1f} ms")
    print("  -> these ms/fps/params feed the 'Proposed (desktop)' row of Table IV (current 4-cue = 12.07 ms).")


if __name__ == "__main__":
    main()
