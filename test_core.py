"""Sanity tests for the numpy core (no torch needed).   Run:  python test_core.py"""
import numpy as np
from color import srgb_to_lab, lab_to_srgb
from decolorizers import get, REGISTRY
from cues import cue_bank, fuse_cues, BT709, CUE_NAMES, N_CUES
from metrics import contrast_metrics, isoluminant_collapse


def test_color_roundtrip():
    rgb = np.random.default_rng(0).random((32, 32, 3))
    err = np.abs(rgb - lab_to_srgb(srgb_to_lab(rgb))).max()
    assert err < 1e-3, f"Lab roundtrip error too high: {err}"
    print(f"  Lab roundtrip max err = {err:.2e}")


def test_cue_bank():
    rgb = np.random.default_rng(1).random((96, 96, 3))
    c = cue_bank(rgb)
    assert c.shape == (N_CUES, 96, 96), c.shape
    assert c.min() >= -1e-9 and c.max() <= 1 + 1e-9, "cues must lie in [0,1]"
    # the bank must contain the fixed standard exactly
    err = np.abs(fuse_cues(c, BT709) - get("bt709")(rgb)).max()
    assert err < 1e-9, f"BT.709 via cue bank differs by {err}"
    print(f"  cues={CUE_NAMES}  BT.709 reproduced to {err:.1e}")


def test_isoluminant_separation():
    """The CHROMA cue must separate equal-luma colours that fixed luma collapses."""
    red = np.array([225, 40, 40.]) / 255; green = np.array([0, 155, 40.]) / 255
    img = np.zeros((64, 64, 3)); img[:, :32] = red; img[:, 32:] = green
    c = cue_bank(img)
    sep = lambda w: abs(fuse_cues(c, w)[:, :32].mean() - fuse_cues(c, w)[:, 32:].mean())
    fixed = sep(BT709)
    chroma = sep(np.array([0.15, 0.35, 0.05, 0.0, 0.45]))
    print(f"  red/green separation: BT.709={fixed:.3f}  chroma-weighted={chroma:.3f}")
    assert chroma > fixed, "chroma cue should improve isoluminant separation"


def test_contrast_metrics():
    rgb = np.random.default_rng(2).random((96, 96, 3))
    for name in ("bt601", "bt709", "average"):
        m = contrast_metrics(rgb, get(name)(rgb))
        assert all(0.0 <= m[k] <= 1.0 for k in m), f"{name}: out of range {m}"
        print(f"  {name:8s} CCPR={m['CCPR']:.3f} CCFR={m['CCFR']:.3f} E={m['E']:.3f}")
    print(f"  iso-collapse bt709 = {isoluminant_collapse(get('bt709'))}")


if __name__ == "__main__":
    print("registry:", sorted(REGISTRY))
    print("test_color_roundtrip");        test_color_roundtrip()
    print("test_cue_bank");               test_cue_bank()
    print("test_isoluminant_separation"); test_isoluminant_separation()
    print("test_contrast_metrics");       test_contrast_metrics()
    print("\nALL CORE TESTS PASSED")
