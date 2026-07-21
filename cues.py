"""The cue bank — a small set of ESTABLISHED, published colour/contrast cues that a
learned model combines into a grayscale image.

    0  R        raw red channel
    1  G        raw green channel
    2  B        raw blue channel
    3  MSR      multi-scale Retinex of the luma  (Jobson et al., IEEE TIP, 1997)
    4  CHROMA   dominant CIELAB opponent-chroma axis, found by PCA of (a*,b*)
                (cf. Grundland & Dodgson, Pattern Recognition, 2007)

Predicting weights over R,G,B generalises the fixed BT.601 / BT.709 luma standards;
MSR supplies local contrast; CHROMA supplies the chromatic distinctions that any pure
luminance projection destroys (isoluminant colours). Every cue is in [0,1] and is
precomputable, so the bank can be cached once per image.
"""
import numpy as np
from color import srgb_to_lab

N_CUES = 5
CUE_NAMES = ["R", "G", "B", "MSR", "CHROMA"]
BT709 = np.array([0.2126, 0.7152, 0.0722, 0.0, 0.0])   # the fixed-standard reference


def _to_float(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8 or rgb.max() > 1.0:
        rgb = rgb.astype(np.float64) / 255.0
    return rgb.astype(np.float64)


def multiscale_retinex(luma, sigmas=(15, 80, 200), eps=1e-6):
    """Multi-scale Retinex (Jobson et al., 1997): log-domain centre/surround at
    several scales, averaged and min-max normalised."""
    import cv2
    L = luma.astype(np.float32)
    logL = np.log1p(255.0 * L)
    acc = np.zeros_like(L, dtype=np.float64)
    for s in sigmas:
        blur = cv2.GaussianBlur(L, (0, 0), sigmaX=float(s), borderType=cv2.BORDER_REFLECT)
        acc += logL - np.log1p(255.0 * blur)
    M = acc / len(sigmas)
    return (M - M.min()) / (np.ptp(M) + eps)


def opponent_chroma(rgb, eps=1e-8):
    """Project the CIELAB (a*,b*) plane onto its dominant axis (first principal
    component) and normalise to [0,1]. Captures the image's main chromatic contrast."""
    lab = srgb_to_lab(rgb)
    ab = np.stack([lab[..., 1], lab[..., 2]], -1).reshape(-1, 2)
    ab = ab - ab.mean(0)
    cov = np.cov(ab.T) + eps * np.eye(2)
    axis = np.linalg.eigh(cov)[1][:, -1]              # top eigenvector
    proj = (ab @ axis).reshape(rgb.shape[:2])
    return (proj - proj.min()) / (np.ptp(proj) + eps)


def cue_bank(rgb):
    """rgb HxWx3 -> cues (5,H,W) in [0,1]."""
    rgb = _to_float(rgb)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    return np.stack([rgb[..., 0], rgb[..., 1], rgb[..., 2],
                     multiscale_retinex(luma), opponent_chroma(rgb)])


def fuse_cues(cues, w):
    """w: (5,) global or (5,H,W) per-pixel, non-negative. Returns gray HxW in [0,1]."""
    w = np.asarray(w, np.float64)
    if w.ndim == 1:
        w = w[:, None, None]
    return np.clip((cues * w).sum(0), 0.0, 1.0)


def collect(images_dir, recursive=False, per_class=0, seed=0):
    """Gather image paths: flat (default), fully recursive, or K images per class folder."""
    import glob, os
    from pathlib import Path
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.ppm", "*.tif")
    rng = np.random.default_rng(seed)
    if per_class:
        files = []
        for sub in sorted(p for p in Path(images_dir).iterdir() if p.is_dir()):
            fs = sorted(f for e in exts for f in glob.glob(str(sub / e)))
            if fs:
                files += [fs[i] for i in sorted(rng.permutation(len(fs))[:per_class])]
        return files
    if recursive:
        return sorted(f for e in exts
                      for f in glob.glob(os.path.join(images_dir, "**", e), recursive=True))
    return sorted(f for e in exts for f in glob.glob(os.path.join(images_dir, e)))
