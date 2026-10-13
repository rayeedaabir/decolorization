"""sRGB <-> CIE XYZ <-> CIELAB conversions (pure numpy, D65 white).

Everything in the harness that touches chrominance (opponent-colour baselines,
the CCPR/CCFR metrics, the isoluminant test) goes through here, so there is a
single, dependency-free source of truth for colour math.

Convention: sRGB values are floats in [0, 1]; Lab has L in [0, 100].
"""
import numpy as np

# Linear-sRGB -> XYZ (D65), from the sRGB spec.
_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])  # D65 reference white
_DELTA = 6.0 / 29.0


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055)


def srgb_to_lab(rgb):
    """rgb: (..., 3) in [0, 1] -> Lab (..., 3), L in [0, 100]."""
    lin = _srgb_to_linear(rgb)
    xyz = lin @ _M_RGB2XYZ.T / _WHITE
    f = np.where(xyz > _DELTA ** 3, np.cbrt(xyz), xyz / (3 * _DELTA ** 2) + 4.0 / 29.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_srgb(lab):
    """lab: (..., 3), L in [0, 100] -> sRGB (..., 3) in [0, 1] (gamut-clipped)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def finv(t):
        return np.where(t > _DELTA, t ** 3, 3 * _DELTA ** 2 * (t - 4.0 / 29.0))

    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1) * _WHITE
    lin = xyz @ _M_XYZ2RGB.T
    return np.clip(_linear_to_srgb(lin), 0.0, 1.0)
