"""Decolorization quality metrics.

- contrast_metrics: CCPR / CCFR / E-score  (Lu et al., 2014; Paper 3 eqs 23-25)
- isoluminant_collapse: how many equal-brightness colour pairs merge (Paper 3 eq 26)

NOTE on scale: colour difference is CIELAB deltaE (full L*a*b*); grayscale
difference is taken on a 0-100 scale so the two are comparable under a shared
threshold tau. This matches the spirit of the CCPR/CCFR definitions; if you
need to reproduce Paper 3's exact numbers, align the tau sweep + sampling with
their setup (500 pairs, tau = 1..40).
"""
import numpy as np
from color import srgb_to_lab, lab_to_srgb


def _sample_pairs(n_pixels, n_pairs, rng):
    i = rng.integers(0, n_pixels, size=n_pairs)
    j = rng.integers(0, n_pixels, size=n_pairs)
    keep = i != j
    return i[keep], j[keep]


def contrast_metrics(rgb, gray, taus=range(1, 41), n_pairs=4000, seed=0):
    """Return dict with mean CCPR, CCFR, E-score over the tau sweep.

    CCPR: of colour-distinct pairs, fraction still distinct in gray (higher=better)
    CCFR: of gray-distinct pairs, fraction that were truly colour-distinct
          (higher=better; penalizes invented/false contrast)
    E   : harmonic mean of the two.
    """
    rgb = np.asarray(rgb, np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    lab = srgb_to_lab(rgb).reshape(-1, 3)
    g = np.clip(np.asarray(gray, np.float64), 0, 1).reshape(-1) * 100.0

    rng = np.random.default_rng(seed)
    i, j = _sample_pairs(lab.shape[0], n_pairs, rng)
    d_color = np.linalg.norm(lab[i] - lab[j], axis=1)
    d_gray = np.abs(g[i] - g[j])

    ccpr, ccfr, escore = [], [], []
    for t in taus:
        omega = d_color >= t                       # colour-distinguishable pairs
        theta = d_gray >= t                        # gray-distinguishable pairs
        p = float((d_gray[omega] >= t).mean()) if omega.any() else 0.0
        f = 1.0 - float((d_color[theta] < t).mean()) if theta.any() else 1.0
        e = 0.0 if (p + f) == 0 else 2 * p * f / (p + f)
        ccpr.append(p); ccfr.append(f); escore.append(e)
    return {"CCPR": float(np.mean(ccpr)),
            "CCFR": float(np.mean(ccfr)),
            "E": float(np.mean(escore))}


def isoluminant_collapse(decolorizer, N=64, L=60.0, r=40.0, eps=0.01):
    """Count equal-luminance colour pairs that collapse to the same gray.

    Builds N colours on a hue circle at fixed L* and chroma radius r in CIELAB
    (so they are perceptually distinct but share brightness), decolorizes, and
    counts pairs whose gray difference < eps. Lower = better. Pure luma methods
    score high here; chroma-aware methods score low.
    """
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    lab = np.stack([np.full(N, L), r * np.cos(theta), r * np.sin(theta)], axis=-1)
    rgb = lab_to_srgb(lab.reshape(N, 1, 3))           # N x 1 x 3
    gray = np.asarray(decolorizer(rgb)).reshape(N)
    diff = np.abs(gray[:, None] - gray[None, :])
    iu = np.triu_indices(N, k=1)
    return int((diff[iu] < eps).sum())
