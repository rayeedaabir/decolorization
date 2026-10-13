"""Raspberry-Pi inference: cue bank + numpy ViT selector. NO PyTorch required.

Only numpy and opencv are needed. Drop this next to cues.py / color.py on the Pi and
plug `Decolorizer` into whatever camera loop you already have:

    from pi_decolor import Decolorizer
    dec = Decolorizer("fusion_vit_np.npz")     # once, at startup
    gray = dec(frame_rgb)                      # per frame: HxWx3 [0,1] or uint8 -> HxW float

    python pi_decolor.py --self-test           # sanity check, no camera needed
    python pi_decolor.py --bench --size 224    # latency at a given processing size
"""
import argparse
import time
import numpy as np
from cues import cue_bank, fuse_cues


# ---------------------------------------------------------------- numpy layers
def _erf(x):
    """Abramowitz & Stegun 7.1.26 — max error ~1.5e-7, far below anything that could
    flip an arg-max, and avoids a scipy dependency."""
    s = np.sign(x); x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def _layernorm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _mha(x, in_w, in_b, out_w, out_b, heads):
    """Multi-head self-attention, mirroring torch.nn.MultiheadAttention (batch_first)."""
    T, D = x.shape
    qkv = x @ in_w.T + in_b                      # (T, 3D)
    q, k, v = np.split(qkv, 3, axis=-1)
    hd = D // heads
    q = q.reshape(T, heads, hd).transpose(1, 0, 2)
    k = k.reshape(T, heads, hd).transpose(1, 0, 2)
    v = v.reshape(T, heads, hd).transpose(1, 0, 2)
    att = _softmax((q @ k.transpose(0, 2, 1)) / np.sqrt(hd), axis=-1)
    ctx = (att @ v).transpose(1, 0, 2).reshape(T, D)
    return ctx @ out_w.T + out_b


class Decolorizer:
    """Loads the exported selector and applies it frame by frame."""

    def __init__(self, npz_path="fusion_vit_np.npz"):
        z = np.load(npz_path)
        self.W = {k: z[k] for k in z.files}
        self.protos = self.W["protos"]
        K, dim, patch, depth, heads = self.W["_meta"]
        self.dim, self.patch, self.depth, self.heads = int(dim), int(patch), int(depth), int(heads)
        self.img = self.patch * int(np.sqrt(self.W["pos"].shape[1] - 1))   # 128
        # patch-embed conv (out, in, k, k) -> (out, in*k*k)
        self.pw = self.W["patch.weight"].reshape(self.dim, -1)
        self.pb = self.W["patch.bias"]
        self.last_prototype = None

    # -------------------------------------------------------------- network
    def _logits(self, rgb_small):
        """rgb_small: (img, img, 3) float32 in [0,1] -> (K,) logits"""
        p, g, D = self.patch, self.img // self.patch, self.dim
        x = rgb_small.transpose(2, 0, 1)                          # (3, H, W)
        # extract patches in the same (c, ky, kx) order the conv weight expects
        t = x.reshape(3, g, p, g, p).transpose(1, 3, 0, 2, 4).reshape(g * g, -1)
        tok = t @ self.pw.T + self.pb                             # (64, D)
        tok = np.concatenate([self.W["cls"].reshape(1, D), tok], 0) + self.W["pos"][0]
        for i in range(self.depth):
            pre = f"enc.layers.{i}."
            a = _mha(tok, self.W[pre + "self_attn.in_proj_weight"],
                     self.W[pre + "self_attn.in_proj_bias"],
                     self.W[pre + "self_attn.out_proj.weight"],
                     self.W[pre + "self_attn.out_proj.bias"], self.heads)
            tok = _layernorm(tok + a, self.W[pre + "norm1.weight"], self.W[pre + "norm1.bias"])
            f = _gelu(tok @ self.W[pre + "linear1.weight"].T + self.W[pre + "linear1.bias"])
            f = f @ self.W[pre + "linear2.weight"].T + self.W[pre + "linear2.bias"]
            tok = _layernorm(tok + f, self.W[pre + "norm2.weight"], self.W[pre + "norm2.bias"])
        cls = _layernorm(tok[0], self.W["head.0.weight"], self.W["head.0.bias"])
        return cls @ self.W["head.1.weight"].T + self.W["head.1.bias"]

    # -------------------------------------------------------------- public
    def __call__(self, rgb):
        import cv2
        rgb = np.asarray(rgb)
        if rgb.dtype == np.uint8 or rgb.max() > 1.0:
            rgb = rgb.astype(np.float32) / 255.0
        small = cv2.resize(rgb.astype(np.float32), (self.img, self.img),
                           interpolation=cv2.INTER_AREA)
        j = int(np.argmax(self._logits(small)))
        self.last_prototype = j
        return fuse_cues(cue_bank(rgb), self.protos[j])

    def all_prototypes(self, rgb):
        """Every prototype fused from ONE cue bank, plus the index the selector chose.

        The cue bank dominates per-frame cost; each prototype on top of it is just a
        weighted sum of five cached maps. So all 16 outputs cost barely more than one,
        which is what makes the live 4x4 grid in pi_live_decolor.py affordable on a Pi.

        Returns (grays, j) -- a list of K arrays in [0,1], and the selected index.
        """
        import cv2
        rgb = np.asarray(rgb)
        if rgb.dtype == np.uint8 or rgb.max() > 1.0:
            rgb = rgb.astype(np.float32) / 255.0
        small = cv2.resize(rgb.astype(np.float32), (self.img, self.img),
                           interpolation=cv2.INTER_AREA)
        j = int(np.argmax(self._logits(small)))
        self.last_prototype = j
        cu = cue_bank(rgb)
        return [fuse_cues(cu, w) for w in self.protos], j


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="fusion_vit_np.npz")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    dec = Decolorizer(a.weights)
    print(f"loaded: {len(dec.protos)} prototypes, dim={dec.dim}, depth={dec.depth}, "
          f"input={dec.img}x{dec.img}")

    if a.self_test:
        rng = np.random.default_rng(0)
        for s in (128, 224, 256):
            img = rng.random((s, s, 3)).astype(np.float32)
            g = dec(img)
            assert g.shape == (s, s), g.shape
            assert 0.0 <= g.min() and g.max() <= 1.0
            print(f"  {s}x{s}: ok, prototype {dec.last_prototype}, "
                  f"range [{g.min():.3f}, {g.max():.3f}]")
        print("SELF-TEST PASSED")

    if a.bench:
        rng = np.random.default_rng(1)
        imgs = [rng.random((a.size, a.size, 3)).astype(np.float32) for _ in range(4)]
        dec(imgs[0])
        t = time.perf_counter()
        for _ in range(a.reps):
            for im in imgs:
                dec(im)
        dt = (time.perf_counter() - t) / (a.reps * len(imgs))
        print(f"  {a.size}x{a.size}: {dt*1e3:.1f} ms/frame  ({1/dt:.1f} fps)")


if __name__ == "__main__":
    main()
