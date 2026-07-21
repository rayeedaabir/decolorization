# decolor-harness

Evaluation harness for **RGB → grayscale decolorization**. It's the reusable
"spine" for the project: any decolorizer (classical or learned) plugs into one
interface, and the same quality metrics + downstream classifier judge them all
on identical settings. Build this first — then dropping in your ViT (or Paper 3,
or Faisal's Mamba) gives comparable numbers immediately.

## What's here

| File | Role | Needs torch? |
|---|---|---|
| `color.py` | sRGB ↔ CIELAB conversions | no |
| `decolorizers.py` | methods + registry (`bt601`, `bt709`, `average`, `chroma_demo`, stubs: `paper3`, `vit`) | no |
| `metrics.py` | CCPR / CCFR / E-score + isoluminant-collapse | no |
| `data.py` | `DecolorizedDataset` (applies a decolorizer on the fly) | yes |
| `classify.py` | MobileNetV3-Large train/eval | yes |
| `run_quality.py` | CLI: quality metrics on an image folder | no |
| `run_classification.py` | CLI: downstream accuracy | yes |
| `test_core.py` | numpy-only sanity tests | no |
| `DATASETS.md` | where to get UCM / AID / Cadik / Color250 | — |

## Setup

```bash
pip install -r requirements.txt        # torch/torchvision only needed for training
python test_core.py                    # verify the numpy core (no torch needed)
```

## Quickstart

```bash
# 1) Image-quality metrics (no torch)
python run_quality.py --images data/Cadik --decolorizer bt601
python run_quality.py --images data/Cadik --decolorizer chroma_demo

# 2) Downstream classification (needs torch + a GPU ideally)
python run_classification.py --data-dir data/UCM --decolorizer bt601
python run_classification.py --data-dir data/UCM --decolorizer bt709
```

Compare methods by re-running with a different `--decolorizer`; everything else
is held fixed.

## The decolorizer interface

```python
# rgb: HxWx3 float[0,1] or uint8   ->   gray: HxW float[0,1]
class MyMethod(Decolorizer):
    def __call__(self, rgb):
        ...
```

Register it and it's instantly available to both CLIs:

```python
@register("mymethod")
class MyMethod(Decolorizer):
    ...
```

## Next steps (from the project plan)
1. Fill in `paper3` — the strong classical baseline / distillation teacher.
2. Reproduce Paper 3's UCM / AID numbers to validate the harness.
3. Implement `vit` — your lightweight model — and benchmark it here.
4. Add a segmentation task (mIoU) and an on-device latency script.

See `../Project Plan - Lightweight ViT Decolorization.md`.

> Metric note: CCPR/CCFR use CIELAB ΔE for colour and a 0–100 gray scale under a
> shared threshold τ. Good for *relative* comparison out of the box; align the τ
> sweep + sampling with Paper 3 if you need to match their absolute values.
