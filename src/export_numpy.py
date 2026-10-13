"""Export the trained Stage-1 selector to a plain .npz  (run on the GPU box, needs torch).

The Pi then needs only numpy + opencv — no PyTorch install, which is the most fragile
part of deploying to aarch64.

    python export_numpy.py --ckpt fusion_vit.pt --codebook codebook.npz --out fusion_vit_np.npz
"""
import argparse
import numpy as np
import torch
from fusion_vit import FusionSelector, load_state_dict
import provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="fusion_vit.pt")
    ap.add_argument("--codebook", default="codebook.npz")
    ap.add_argument("--out", default="fusion_vit_np.npz")
    a = ap.parse_args()

    protos = np.load(a.codebook)["protos"].astype(np.float32)
    model = FusionSelector(len(protos))
    model.load_state_dict(load_state_dict(a.ckpt, "cpu"))
    model.eval()

    out = {"protos": protos}
    for k, v in model.state_dict().items():
        out[k] = v.detach().cpu().numpy().astype(np.float32)
    # record architecture so the numpy side needs no assumptions
    out["_meta"] = np.array([len(protos), model.patch.out_channels,
                             model.patch.kernel_size[0], len(model.enc.layers), 4],
                            dtype=np.int32)   # [K, dim, patch, depth, heads]
    # carry provenance across to the Pi so a deployed model is traceable too
    out["provenance"] = np.array(provenance.as_json(
        source_ckpt=a.ckpt, source_ckpt_sha=provenance.file_sha(a.ckpt),
        codebook=a.codebook, codebook_sha=provenance.file_sha(a.codebook)))
    np.savez_compressed(a.out, **out)
    n = sum(v.size for k, v in out.items() if not k.startswith(("_", "protos")))
    print(f"exported {n/1e3:.1f}k parameters + {len(protos)} prototypes -> {a.out}")
    print("keys:", ", ".join(sorted(k for k in out if not k.startswith("_"))[:6]), "...")


if __name__ == "__main__":
    main()
