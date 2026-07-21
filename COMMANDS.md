# Commands — decolor-harness

Run everything from inside `decolor-harness/`. Activate your env first
(`.venv\Scripts\activate`) and put datasets under `data/` (see `DATASETS.md`).

Registered methods (use as `--decolorizer`): `bt601`, `bt709`, `average`,
`paper3_b1` (chaos), `paper3_b12` (+Retinex), `paper3_b123` (+chroma = base),
`paper3` (full, with post-processing).

---

## 1. Verify the code — no data needed
```
python test_core.py
```
**Output:** each test prints values, ending in `ALL CORE TESTS PASSED`. Confirms the
colour math, metrics, and all three Paper 3 branches work.

## 2. Prepare the quality datasets — one-time
```
python prep_datasets.py --cadik data/Cadik --color250 data/Color250/images
```
**Output:**
```
Cadik:    wrote 25 originals -> data/Cadik_rgb  (expected 25)
Color250: wrote 250 originals -> data/Color250_rgb  (expected 250)
```
Creates `data/Cadik_rgb/` and `data/Color250_rgb/` (originals only, no bundled results).

## 3. Image-quality metrics
```
python run_quality.py --images data/Cadik_rgb --decolorizer paper3_b123
```
**Output:** folder-mean CCPR / CCFR / E-score + the isoluminant-collapse count:
```
decolorizer=paper3_b123  images=25
  CCPR : 0.6173
  CCFR : 0.9918
  E    : 0.7201
  isoluminant_collapse: 62  (lower=better)
```
Swap `--decolorizer` to compare methods. *Note:* for the paper's **quality** config the
full method runs with saliency off — so `paper3_b123` (the base) is the fair "proposed"
row for quality tables; `paper3` (saliency on) is the downstream config.

## 4. Downstream classification  (needs torch + your 4060)
```
python run_classification.py --data-dir data/UCM/images --decolorizer paper3
python run_classification.py --data-dir data/AID/data   --decolorizer paper3
```
**Output:** per-epoch validation accuracy then the best:
```
decolorizer=paper3  classes=21  train=1680  val=420
epoch  1/15  val_acc=0.83..  best=0.83..
...
BEST val_acc (paper3): 0.96..
```

## 5. Visual comparison — SINGLE image  (for the professor)
```
python show_image.py --image data/Cadik_rgb/1.png --out figs/compare.png
python show_image.py --image data/Cadik_rgb/1.png --internals --out figs/internals.png
```
- **default:** Original RGB + each method's grayscale, each labelled with its E-score.
- **`--internals`:** the pipeline — chaos → Retinex → chroma → saliency → full paper3.
- Works on ANY image path (a dataset image *or your own photo*).

**Output:** `wrote figs/compare.png` (or `figs/internals.png`) — open the PNG.

## 6. Visual comparison — GRID of several images
```
python make_figures.py --images data/Cadik_rgb --n 5 --out figs/cadik_montage.png
```
**Output:** `wrote figs/cadik_montage.png (W, H)`. Grid: rows = images, columns =
[Original | bt601 | B1 | B1+2 | B1+2+3 | paper3 (full)].

## 7. Timing / latency
```
python bench.py --images data/Cadik_rgb
```
**Output:** ms/image and images/second per method:
```
method           ms/img    img/s
bt601              0.1x    ....
paper3            ~20-30   ....
```
Re-run on the 4060 box **and the Raspberry Pi 5** — those are your deployment numbers
and the baseline the ViT must beat on the accuracy↔latency trade-off.

---

### Quick "show the professor" sequence
```
python show_image.py --image data/Cadik_rgb/3.png --out figs/compare.png
python show_image.py --image data/Cadik_rgb/3.png --internals --out figs/internals.png
python make_figures.py --images data/Cadik_rgb --n 5 --out figs/cadik_montage.png
python bench.py --images data/Cadik_rgb
```
