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
