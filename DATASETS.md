# Datasets

Two kinds:
- **Classification** sets (have class subfolders) → used with `run_classification.py`.
- **Quality** sets (single images, no labels) → used with `run_quality.py`.

Put everything under `decolor-harness/data/`.

## Classification

### UC Merced (UCM) — 21 classes × 100 images, 256×256
Layout as shipped: `data/UCM/images/<class>/<class>NN.tif`
Point the classifier at the folder that holds the class subfolders:
```
python run_classification.py --data-dir data/UCM/images --decolorizer bt601
```

### AID — 30 classes, 10,000 images, 600×600
Layout as shipped: `data/AID/data/<Class>/<class>_N.png`
```
python run_classification.py --data-dir data/AID/data --decolorizer bt601
```

## Quality (originals + bundled results → run prep first)

Both of these ship with prior methods' grayscale outputs mixed in. Run the prep
script once to pull out just the original colour images:

```
python prep_datasets.py --cadik data/Cadik --color250 data/Color250/images
```

### Cadik — 25 images
Each subfolder has the original `<name>.ppm` plus other methods' outputs
(`*_CIE_Y.ppm`, `*_Grundland07.ppm`, `*_Smith08.ppm`, `*Bala04.tif`, …).
Prep copies the 25 originals to `data/Cadik_rgb/`.
```
python run_quality.py --images data/Cadik_rgb --decolorizer bt601
```

### Color250 — 250 images
`images/` holds 250 originals (plain index names, e.g. `17.png`) plus six results
per image (`17_1.png` … `17_6.png`) — hence ~1751 files total. Prep copies the
250 originals to `data/Color250_rgb/`.
```
python run_quality.py --images data/Color250_rgb --decolorizer bt601
```

## Notes
- `bt601` / `bt709` / etc. are **methods the script applies on the fly** — you never
  need a pre-made grayscale folder. `run_quality.py --decolorizer bt601` takes the
  colour originals and produces the bt601 grayscale itself, then scores it.
- Cadik & Color250 have **no classes**, so they are quality-only (not classification).
- The bundled `_k` results are other papers' methods — keep them; they're handy later
  as extra comparison baselines.
- Segmentation dataset (Phase 2) is still TBD — decide scope with the professor.
