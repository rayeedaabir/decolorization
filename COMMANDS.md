# Commands — full pipeline

Run everything from inside the code folder with your `image` env active.
Methods available: `bt601`, `bt709`, `average`, plus `vit:CKPT.pt` (Stage 1) and
`vit2:CKPT.pt` (Stage 2). PowerShell syntax where loops are used.

---

## 0 · Verify the install
```powershell
python test_core.py
```
Expect `ALL CORE TESTS PASSED` and `cues=['R','G','B','MSR','CHROMA']`.

## 1 · Dataset prep — one-time (skip if `data/Cadik_rgb` + `data/Color250_rgb` exist)
```powershell
python prep_datasets.py --cadik data/Cadik --color250 data/Color250/images
```

## 2 · Published-method baselines  ⏱ ~5 min
The six methods bundled with Color250 — your SOTA comparison, no reimplementation.
```powershell
foreach ($k in 1..6) { python eval_pregen.py --originals data/Color250_rgb --results data/Color250/images --suffix $k }
```
`_1` Lu 2014 · `_2` Lu 2012 · `_3` CIE Y · `_4` Grundland 07 · `_5` Smith 08 · `_6` Color2Gray 05

## 3 · Fixed-standard baselines — quality
```powershell
python run_quality.py --images data/Cadik_rgb    --decolorizer bt709
python run_quality.py --images data/Color250_rgb --decolorizer bt709
python quality_nr.py  --images data/Cadik_rgb    --decolorizer bt709
python quality_nr.py  --images data/Color250_rgb --decolorizer bt709
```

## 4 · Fixed-standard baseline — downstream  ⏱ UCM ~10 min, AID ~40 min
Export first so the baseline and the learned methods go through an identical pipeline.
```powershell
python export_gray.py --data-dir data/UCM/images --decolorizer bt709 --out data/UCM_bt709
python export_gray.py --data-dir data/AID/data   --decolorizer bt709 --out data/AID_bt709
python run_classification.py --data-dir data/UCM_bt709 --decolorizer bt601
python run_classification.py --data-dir data/AID_bt709 --decolorizer bt601
```
(`bt601` on an already-gray image is the identity, so you train on the exported grays.)

## 5 · The oracle — ceiling + Stage-1 targets  ⏱ ~30–45 min (slowest step)
```powershell
python cue_search.py --images data/Cadik_rgb    --out targets_cadik.csv
python cue_search.py --images data/Color250_rgb --out targets_color250.csv
python cue_search.py --images data/UCM/images --per-class 20 --out targets_ucm.csv
python cue_search.py --images data/AID/data    --per-class 15 --out targets_aid.csv
```
Add `--samples 80` to roughly halve the time if needed (default 150).

## 6 · Stage 1 — global cue weights
```powershell
python fusion_vit.py --targets targets_cadik.csv targets_color250.csv targets_ucm.csv targets_aid.csv --epochs 80
```
Prints held-out E for the ViT vs fixed BT.709, and saves `fusion_vit.pt`.

## 7 · Evaluate Stage 1
```powershell
python run_quality.py --images data/Cadik_rgb    --decolorizer vit:fusion_vit.pt
python run_quality.py --images data/Color250_rgb --decolorizer vit:fusion_vit.pt
python quality_nr.py  --images data/Cadik_rgb    --decolorizer vit:fusion_vit.pt
python export_gray.py --data-dir data/UCM/images --decolorizer vit:fusion_vit.pt --out data/UCM_vit
python export_gray.py --data-dir data/AID/data   --decolorizer vit:fusion_vit.pt --out data/AID_vit
python run_classification.py --data-dir data/UCM_vit --decolorizer bt601
python run_classification.py --data-dir data/AID_vit --decolorizer bt601
```

## 8 · Stage 2 — per-pixel fusion on the downstream loss
```powershell
python cache_cues.py --data-dir data/UCM/images --out cache/UCM --per-class 40
python train_stage2.py --cache cache/UCM --epochs 20 --out stage2_vit.pt
```

## 9 · Evaluate Stage 2
```powershell
python run_quality.py --images data/Cadik_rgb    --decolorizer vit2:stage2_vit.pt
python run_quality.py --images data/Color250_rgb --decolorizer vit2:stage2_vit.pt
python export_gray.py --data-dir data/UCM/images --decolorizer vit2:stage2_vit.pt --out data/UCM_vit2
python run_classification.py --data-dir data/UCM_vit2 --decolorizer bt601
```

## 10 · Latency
```powershell
python bench.py --images data/Cadik_rgb --methods bt601,bt709,vit:fusion_vit.pt,vit2:stage2_vit.pt
```
Re-run this one on the Raspberry Pi 5 for the deployment numbers.

## 11 · Figures
```powershell
python show_image.py --image data/Cadik_rgb/<file>.png --internals --out figs/internals.png
python show_image.py --image data/Cadik_rgb/<file>.png --methods bt709,vit:fusion_vit.pt,vit2:stage2_vit.pt --out figs/compare.png
python make_figures.py --images data/Cadik_rgb --n 5 --methods bt709,vit:fusion_vit.pt --out figs/montage.png
```
Then paste your numbers into the block at the top of `make_charts.py` and run:
```powershell
python make_charts.py
```

---

### Critical path
Steps **2** and **5** are the ones that decide whether the re-based method holds up —
the published-baseline table and the new oracle ceiling. Run those first.
