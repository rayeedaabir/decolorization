#!/usr/bin/env python3
"""Figure 4 (clean) — live Raspberry Pi demo, rebuilt from the banner-free snapshot panes.
No overlaid timing/weight text; method names are typeset column headers instead.

Structure (matches the original Fig 4):
  (a),(b)  comparison rows:  Input | BT.709 | Proposed | cv2decolor   (two scenes)
  (c),(d)  codebook rows:    Input | 16-prototype grid (selected boxed) (same two scenes)

Edit SETS below to point at your four snapshot folders, then:
    python make_fig4.py --root "fig4_sets/Sets" --out figure4_clean.png
"""
import argparse, glob, os
from PIL import Image, ImageDraw, ImageFont

# which snapshot folder supplies each row, and the hero (clearest) scene first
SETS = {
    "cmp_a": ("Set 2", "Colour-vision test plate"),   # Ishihara "26" — hero
    "cmp_b": ("Set 1", "Hand-held toys"),             # Iron-Man scene
    "cod_c": ("Set 4", "Colour-vision test plate"),   # Ishihara codebook
    "cod_d": ("Set 3", "Hand-held toys"),             # Iron-Man codebook
}
COLS = [("input", "Input"), ("bt709", "BT.709"), ("ours", "Proposed"), ("cv2decolor", "cv2decolor")]

GAP, BORDER, MARGIN_L, HDR_H = 8, 1, 48, 30
BLOCK_GAP = 26
INK, FRAME, BG = "black", "#8a8a8a", "white"


def font(sz, bold=False):
    names = (["timesbd.ttf", "FreeSerifBold.ttf", "LiberationSerif-Bold.ttf", "DejaVuSerif-Bold.ttf"] if bold else
             ["times.ttf", "FreeSerif.ttf", "LiberationSerif-Regular.ttf", "DejaVuSerif.ttf"])
    for n in names:
        try: return ImageFont.truetype(n, sz)
        except Exception: continue
    return ImageFont.load_default()


SCALE = 1   # set from --scale: panes are upsampled (LANCZOS) and every size multiplied, so the
            # figure reaches print resolution at its column width; it adds no image detail.


def pane(root, setname, key, target_h=None):
    g = glob.glob(os.path.join(root, setname, f"*_{key}.png"))
    if not g:
        raise SystemExit(f"missing '{key}' in {setname}")
    im = Image.open(sorted(g)[0]).convert("RGB")
    if SCALE != 1 and target_h is None:
        im = im.resize((im.width * SCALE, im.height * SCALE), Image.LANCZOS)
    if target_h and im.height != target_h:
        im = im.resize((round(im.width * target_h / im.height), target_h), Image.LANCZOS)
    return im


def framed(im):
    c = Image.new("RGB", (im.width + 2 * BORDER, im.height + 2 * BORDER), FRAME)
    c.paste(im, (BORDER, BORDER)); return c


def center_text(d, cx, y, text, fnt, fill=INK):
    w = d.textbbox((0, 0), text, font=fnt)[2]
    d.text((cx - w / 2, y), text, font=fnt, fill=fill)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="fig4_sets/Sets")
    ap.add_argument("--out", default="figure4_clean.png")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--scale", type=int, default=1, help="2 = double pixel size for >=300 dpi at 6.3 in")
    a = ap.parse_args()
    global SCALE, GAP, BORDER, MARGIN_L, HDR_H, BLOCK_GAP
    SCALE = a.scale
    GAP, BORDER, MARGIN_L, HDR_H, BLOCK_GAP = (v * SCALE for v in (GAP, BORDER, MARGIN_L, HDR_H, BLOCK_GAP))

    f_hdr, f_row, f_sub = font(20 * SCALE), font(22 * SCALE, bold=True), font(17 * SCALE)

    # ---- comparison panes (native 320x240) ----
    cmp_rows = []
    for rk in ("cmp_a", "cmp_b"):
        setname, _ = SETS[rk]
        cmp_rows.append([framed(pane(a.root, setname, k)) for k, _ in COLS])
    pw, ph = cmp_rows[0][0].width, cmp_rows[0][0].height      # framed pane size
    content_w = 4 * pw + 3 * GAP
    block_x = MARGIN_L

    # ---- codebook panes: input + grid, scaled so [input | grid] == content_w ----
    cod_rows = []
    for rk in ("cod_c", "cod_d"):
        setname, _ = SETS[rk]
        gh = 480 * SCALE
        inp = pane(a.root, setname, "input", target_h=gh)
        grid = pane(a.root, setname, "grid", target_h=gh)
        # scale the pair to fill content_w exactly, keep aspect
        scale = content_w / (inp.width + GAP * 3 + grid.width)
        inp = inp.resize((round(inp.width * scale), round(gh * scale)), Image.LANCZOS)
        grid = grid.resize((round(grid.width * scale), round(gh * scale)), Image.LANCZOS)
        cod_rows.append((framed(inp), framed(grid)))
    ch = cod_rows[0][0].height

    # ---- canvas ----
    W = MARGIN_L + content_w + 10 * SCALE
    H = HDR_H + 2 * (ph + GAP) + BLOCK_GAP + 2 * (ch + GAP) + 34 * SCALE
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)

    # column headers over the comparison block
    for i, (_, lab) in enumerate(COLS):
        cx = block_x + i * (pw + GAP) + pw / 2
        center_text(d, cx, 6 * SCALE, lab, f_hdr)

    y = HDR_H
    labels = ["(a)", "(b)"]
    for r, row in enumerate(cmp_rows):
        for i, im in enumerate(row):
            canvas.paste(im, (block_x + i * (pw + GAP), y))
        d.text((10 * SCALE, y + ph / 2 - 12 * SCALE), labels[r], font=f_row, fill=INK)
        y += ph + GAP

    y += BLOCK_GAP - GAP
    # codebook sub-headers
    inp_w = cod_rows[0][0].width
    center_text(d, block_x + inp_w / 2, y - 24 * SCALE, "Input", f_sub)
    center_text(d, block_x + inp_w + GAP * 3 + cod_rows[0][1].width / 2, y - 24 * SCALE,
                "Learned codebook — 16 prototypes (selected boxed)", f_sub)
    clabels = ["(c)", "(d)"]
    for r, (inp, grid) in enumerate(cod_rows):
        canvas.paste(inp, (block_x, y))
        canvas.paste(grid, (block_x + inp.width + GAP * 3, y))
        d.text((10 * SCALE, y + ch / 2 - 12 * SCALE), clabels[r], font=f_row, fill=INK)
        y += ch + GAP

    base = a.out.rsplit(".", 1)[0]
    canvas.save(base + ".png", dpi=(a.dpi, a.dpi))
    canvas.save(base + ".pdf", "PDF", resolution=float(a.dpi))
    print(f"wrote {base}.png  ({canvas.width}x{canvas.height})")
    print(f"wrote {base}.pdf")


if __name__ == "__main__":
    main()