"""Make every artifact say when and how it was built.

Why this exists
---------------
`codebook.npz` and `fusion_vit.pt` are overwritten in place by default, so after a few
exploratory re-runs there is no way to tell whether the checkpoint on disk is the one
that produced the numbers in the paper. Worse, the two can drift apart: retraining the
selector against a *new* codebook while an *old* codebook.npz sits beside it produces
results that look fine and are silently wrong.

This module records, inside each artifact:
  - when it was built, and with what command line
  - the SHA-256 of the source files that determine its behaviour
  - for the selector, the SHA-256 of the codebook it was trained against

`check_artifacts.py` reads those stamps back and flags a mismatch. Nothing here changes
any numerical result; it only records how the result was produced.
"""
import hashlib, json, os, platform, sys, time

# Source files whose contents change what an artifact means.
TRACKED = ("cues.py", "metrics.py", "color.py", "build_codebook.py", "fusion_vit.py")


def file_sha(path, n=12):
    """Short SHA-256 of a file, or None if it is not there."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def datestamp():
    """UTC-safe local timestamp for filenames: 20260831-1742."""
    return time.strftime("%Y%m%d-%H%M")


def stamped_name(path, when=None):
    """codebook.npz -> codebook_20260831-1742.npz"""
    root, ext = os.path.splitext(path)
    return f"{root}_{when or datestamp()}{ext}"


def stamp(**extra):
    """The provenance record embedded in an artifact."""
    rec = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": platform.python_version(),
        "argv": " ".join(sys.argv),
        "sources": {f: file_sha(f) for f in TRACKED if os.path.exists(f)},
    }
    rec.update(extra)
    return rec


def as_json(**extra):
    """Provenance as a JSON string, for stuffing into an .npz."""
    return json.dumps(stamp(**extra), indent=1)


def read_stamp(path):
    """Pull the provenance record back out of a .npz or a .pt. None if absent."""
    try:
        if path.endswith(".npz"):
            import numpy as np
            d = np.load(path, allow_pickle=True)
            if "provenance" not in d:
                return None
            return json.loads(str(d["provenance"]))
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "meta" in obj:
            return obj["meta"]
        return None
    except Exception:
        return None


def describe(path):
    """One-line-per-field summary for the console."""
    rec = read_stamp(path)
    if rec is None:
        return (f"{path}: no provenance stamp -- built before this was added, "
                f"or by an older script. Age on disk: "
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))}"
                if os.path.exists(path) else f"{path}: MISSING")
    lines = [f"{path}"]
    for k in ("built_at", "host", "n_images", "k", "epochs", "codebook", "codebook_sha"):
        if k in rec:
            lines.append(f"    {k:14s} {rec[k]}")
    if "sources" in rec:
        lines.append("    sources       " +
                     "  ".join(f"{k}={v}" for k, v in sorted(rec["sources"].items())))
    lines.append(f"    argv           {rec.get('argv','?')}")
    return "\n".join(lines)
