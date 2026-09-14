"""Decolorizers: RGB -> grayscale, behind one tiny interface.

This module holds only the FIXED, PUBLISHED standards used as baselines. The learned
methods live in fusion_vit.py (Stage 1) and stage2.py (Stage 2) and are selected in the
runners with the 'vit:' / 'vit2:' prefixes.

    Input : rgb  HxWx3, float [0,1] or uint8
    Output: gray HxW,   float [0,1]
"""
import numpy as np

REGISTRY = {}


def register(name):
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


def get(name, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown decolorizer '{name}'. available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def _to_float(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8 or rgb.max() > 1.0:
        rgb = rgb.astype(np.float64) / 255.0
    return rgb.astype(np.float64)


class Decolorizer:
    name = "base"

    def __call__(self, rgb):
        raise NotImplementedError


@register("bt601")
class BT601(Decolorizer):
    """ITU-R BT.601 luma — the default in most image tools."""
    def __call__(self, rgb):
        return np.clip(_to_float(rgb) @ [0.299, 0.587, 0.114], 0, 1)


@register("bt709")
class BT709(Decolorizer):
    """ITU-R BT.709 luma (HDTV)."""
    def __call__(self, rgb):
        return np.clip(_to_float(rgb) @ [0.2126, 0.7152, 0.0722], 0, 1)


@register("average")
class Average(Decolorizer):
    """Unweighted channel mean."""
    def __call__(self, rgb):
        return np.clip(_to_float(rgb).mean(-1), 0, 1)


def get_any(name, **kwargs):
    """Resolve ANY method name used across the scripts:

        'bt601' / 'bt709' / 'average'   -> fixed published standards
        'vit:CKPT.pt'                   -> Stage 1 (codebook selection)
        'vit2:CKPT.pt'                  -> Stage 2 (per-pixel fusion)

    Stage 1 also needs its codebook; pass 'vit:CKPT.pt:codebook.npz' to point at a
    non-default codebook file.
    """
    if name.startswith("vit2:"):
        from stage2 import load_decolorizer as _l2
        return _l2(name.split(":", 1)[1])
    if name.startswith("vit:"):
        from fusion_vit import load_decolorizer as _l1
        rest = name.split(":", 1)[1]
        if ":" in rest:                      # 'ckpt.pt:codebook.npz'
            ckpt, cb = rest.split(":", 1)
            return _l1(ckpt, codebook=cb)
        return _l1(rest)
    return get(name, **kwargs)


@register("cv2decolor")
class CV2Decolor(Decolorizer):
    """OpenCV's cv2.decolor -- the real-time contrast-preserving decolorization of
    Lu et al. (SIGGRAPH Asia 2012 Technical Briefs).

    Added for the ICASSP revision so the cost axis has a *measured* published
    reference point on identical hardware, rather than runtimes quoted from papers
    that used different machines. Note this is the authors' fast variant; the
    quality figures in Table I still come from the released outputs bundled with
    Color250 (suffix _2), not from this implementation.
    """
    def __call__(self, rgb):
        import cv2
        x = (np.clip(_to_float(rgb), 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
        gray, _ = cv2.decolor(np.ascontiguousarray(x))
        return gray.astype(np.float64) / 255.0


@register("bt709stretch")
class BT709Stretched(Decolorizer):
    """BT.709 followed by a per-image min-max stretch to [0,1].

    A CONTROL, not a method. It carries exactly as much chromatic information as
    plain BT.709 -- none beyond the fixed luma projection -- but it has full output
    dynamic range, which is the other thing CCPR rewards. Any contrast-preservation
    claim has to beat this, or the gain might be range rather than chroma.

    Reported alongside the real methods so the confound is visible instead of latent.
    """
    def __call__(self, rgb):
        g = _to_float(rgb) @ np.array([0.2126, 0.7152, 0.0722])
        lo, hi = float(g.min()), float(g.max())
        return np.clip((g - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(g)
