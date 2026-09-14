"""Decolorization quality metrics.

- contrast_metrics: CCPR / CCFR / E-score  (Lu et al., IJCV 2014)
- isoluminant_collapse: how many equal-brightness colour pairs merge to the same gray

NOTE on scale: colour difference is CIELAB deltaE (full L*a*b*); grayscale
difference is taken on a 0-100 scale so the two are comparable under a shared
threshold tau. This matches the spirit of the CCPR/CCFR definitions of
Lu et al. (IJCV 2014); align the tau sweep + sampling if you need to match a
particular paper's protocol.
"""
import math
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


# ---------------------------------------------------------------------------
# Added for the ICASSP revision: paired significance testing and the
# iso-luminant restriction of CCPR. See "RUN SHEET - GPU Experiments.md".
# ---------------------------------------------------------------------------

def _rankdata(x):
    """Average ranks (1-based), ties averaged. numpy-only stand-in for scipy."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _norm_sf(z):
    """Upper-tail probability of the standard normal (no scipy)."""
    return 0.5 * math.erfc(abs(z) / math.sqrt(2.0))


def wilcoxon_signed_rank(diff):
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Normal approximation with tie correction; adequate for n=250 and avoids a
    scipy dependency. Returns (p_value, n_nonzero, z).
    """
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    d = d[d != 0.0]
    n = len(d)
    if n < 10:
        return float("nan"), n, float("nan")
    r = _rankdata(np.abs(d))
    w_plus = float(r[d > 0].sum())
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = float(((counts ** 3) - counts).sum())
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return float("nan"), n, float("nan")
    z = (w_plus - mu) / math.sqrt(var)
    return 2.0 * _norm_sf(z), n, z


def bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of paired differences."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def isoluminant_ccpr(rgb, gray, taus=range(1, 41), n_pairs=200_000, dL=2.0,
                     min_count=50, seed=0):
    """CCPR restricted to near-iso-luminant pixel pairs.

    Standard CCPR averages over *all* colour-distinct pairs, so it is dominated by
    pairs that already differ in lightness -- pairs any luma projection preserves
    for free. That is why methods rank closely on E while differing sharply on the
    failure mode that motivates adaptive decolorization.

    This variant keeps only pairs whose CIELAB lightness differs by at most `dL`,
    i.e. exactly the pairs a fixed luma projection is obliged to collapse, and asks
    what fraction remain distinguishable in the grayscale output. It needs only the
    original RGB and the grayscale result, so it can be computed for published
    methods from their released outputs.

    Pairs are drawn by sampling an anchor pixel and a partner from within its
    lightness window (sort + searchsorted), which yields far more qualifying pairs
    than rejection-sampling uniformly.

    Higher is better. Report alongside CCFR so it is clear the gain is not simply
    invented contrast.
    """
    rgb = np.asarray(rgb, np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    lab = srgb_to_lab(rgb).reshape(-1, 3)
    g = np.clip(np.asarray(gray, np.float64), 0, 1).reshape(-1) * 100.0
    if lab.shape[0] != g.shape[0]:
        raise ValueError(f"pixel count mismatch: rgb {lab.shape[0]}, gray {g.shape[0]}")

    n = lab.shape[0]
    rng = np.random.default_rng(seed)
    L = lab[:, 0]
    order = np.argsort(L, kind="stable")
    Ls = L[order]

    a_pos = rng.integers(0, n, size=n_pairs)
    lo = np.searchsorted(Ls, Ls[a_pos] - dL, side="left")
    hi = np.searchsorted(Ls, Ls[a_pos] + dL, side="right")
    span = np.maximum(hi - lo, 1)
    b_pos = lo + (rng.random(n_pairs) * span).astype(np.int64)
    b_pos = np.clip(b_pos, 0, n - 1)

    i, j = order[a_pos], order[b_pos]
    keep = i != j
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return {"CCPR_iso": float("nan"), "n_iso_pairs": 0}

    d_chroma = np.linalg.norm(lab[i, 1:] - lab[j, 1:], axis=1)
    d_gray = np.abs(g[i] - g[j])

    vals = []
    for t in taus:
        omega = d_chroma >= t
        c = int(omega.sum())
        vals.append(float((d_gray[omega] >= t).mean()) if c >= min_count else np.nan)
    vals = np.asarray(vals, float)
    if np.all(np.isnan(vals)):
        return {"CCPR_iso": float("nan"), "n_iso_pairs": int(len(i))}
    return {"CCPR_iso": float(np.nanmean(vals)), "n_iso_pairs": int(len(i))}
