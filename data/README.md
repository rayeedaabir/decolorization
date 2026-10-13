# Datasets (not included)

The evaluation datasets are third-party and are **not redistributed** here. Download each
from its original source, follow its licence, and arrange it as below. All commands in the
top-level README are run from the repository root and expect exactly this layout (the
codebooks store their training-image paths relative to the repository root, e.g.
`data\UCM\images\tenniscourt\tenniscourt98.tif`).

## Expected layout

```
data/
├── Cadik/                 raw Čadík download (25 subfolders)         → prep_datasets.py
├── Cadik_rgb/             25 original colour images      (Tables II, IX; Fig. 1)
├── Color250/images/       raw Color250 download, including the released outputs of the
│                          six published methods (<idx>_1 … <idx>_6)  (Table I)
├── Color250_rgb/          250 original colour images     (Tables I, II, IX; Figs. 2, 8)
├── UCM/images/<class>/    UC Merced, 21 classes × 100 images (.tif)  (Tables II, III)
├── AID/data/<class>/      AID, 30 classes, 10,000 images (.jpg)       (Tables II, III)
└── Flowers102/<class>/    Oxford Flowers-102 as an ImageFolder        (Table III)
```

Create the clean folders from the raw downloads with:

```bash
python src/prep_datasets.py --cadik data/Cadik --color250 data/Color250/images   # → data/Cadik_rgb, data/Color250_rgb
python src/prep_flowers.py --out data/Flowers102                                  # torchvision download, ~350 MB
```

The downstream study (Table III) trains on exported grayscale copies, written by
`src/export_gray.py` into `data/UCM_bt709`, `data/UCM_vit4`, `data/AID_bt709`, `data/AID_vit4`,
`data/Flowers_bt709_full`, `data/Flowers_cv2_full` and `data/Flowers_vit4` (commands in the
top-level README).

## Selector training pool (1,145 images)

The oracle targets in `results/t_rgbchroma_*.csv` and the codebook were computed on all 25
Čadík images, all 250 Color250 images, a class-balanced sample of UC Merced (20 per class,
420 images) and of AID (15 per class, 450 images). Flowers-102 is not used for selector
training.

## Sources

- **Čadík:** M. Čadík, “Perceptual evaluation of color-to-grayscale image conversions,” *Comput. Graph. Forum*, vol. 27, no. 7, pp. 1745–1754, 2008. https://doi.org/10.1111/j.1467-8659.2008.01319.x
- **Color250:** C. Lu, L. Xu, and J. Jia, “Contrast preserving decolorization with perception-based quality metrics,” *Int. J. Comput. Vis.*, vol. 110, no. 2, pp. 222–239, 2014. https://doi.org/10.1007/s11263-014-0732-6
- **UC Merced (UCM):** Y. Yang and S. Newsam, “Bag-of-visual-words and spatial extensions for land-use classification,” in *Proc. 18th ACM SIGSPATIAL Int. Conf. Adv. Geogr. Inf. Syst.*, 2010, pp. 270–279. https://doi.org/10.1145/1869790.1869829
- **Aerial Image Dataset (AID):** G.-S. Xia *et al.*, “AID: A benchmark data set for performance evaluation of aerial scene classification,” *IEEE Trans. Geosci. Remote Sens.*, vol. 55, no. 7, pp. 3965–3981, 2017. https://doi.org/10.1109/TGRS.2017.2685945
- **Oxford Flowers-102:** M.-E. Nilsback and A. Zisserman, “Automated flower classification over a large number of classes,” in *Proc. 6th Indian Conf. Comput. Vis., Graph. Image Process. (ICVGIP)*, 2008, pp. 722–729.
