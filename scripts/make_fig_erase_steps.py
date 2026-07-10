"""Build the step-by-step ink-erase/composite figure (Reviewer 3, comment 4.1).

Panels, all cropped to the same window around one math detection:
  (a) input slide crop, detector box (dashed) + tight ink box (solid)
  (b) the binary ink mask actually used for erasure (threshold 195,
      elliptical dilation r=2), computed inside the 6-px-padded box
  (c) the crop after _erase_ink_region ran — mask-only fill
  (d) the deployed pipeline's own composited output for the same window

(b) and (c) are produced by calling the deployed functions in infer.py;
(d) is cropped from outputs/jakob_full_corpus (the real corpus run), so
every panel is pipeline truth, not an illustration.

Usage: .venv/bin/python scripts/make_fig_erase_steps.py
Writes paper/ieee_access/fig_erase_a.png ... fig_erase_d.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer import _erase_ink_region, _ink_bbox  # noqa: E402

LECTURE = "01_Vollständige_Induktion"
PAGE = "01_Vollständige_Induktion_page-0003.jpg"
# region: \sum_{k=1}^{100} k = 1+2+3+4+...+100  (conf 0.83, rendered)
BBOX_INDEX = 1

PAD_VIEW = 45          # context margin around the detector box in the crop
ERASE_PAD = 6          # deployed value (infer.py _erase_ink_region default)
THRESHOLD = 195        # deployed value
DILATE = 2             # deployed value

results = json.load(open(ROOT / "outputs/jakob_full_corpus" / LECTURE / "results.json"))
# NFC/NFD unicode differs between the Linux-written JSON and this source
# file, so resolve the key by its page suffix rather than exact match.
slide_key = next(k for k in results if k.endswith(PAGE.split("page-")[-1]) and "page-0003" in k)
region = results[slide_key][BBOX_INDEX]
assert region["type"] == "math" and region.get("rendered"), region
x1, y1, x2, y2 = (int(v) for v in region["bbox"])
print(f"region: bbox=({x1},{y1},{x2},{y2}) conf={region['confidence']:.2f}")
print(f"latex: {region['latex']}")

src = Image.open(ROOT / slide_key).convert("RGB")
out = Image.open(ROOT / "outputs/jakob_full_corpus" / LECTURE / PAGE).convert("RGB")
iw, ih = src.size
vx1, vy1 = max(0, x1 - PAD_VIEW), max(0, y1 - PAD_VIEW)
vx2, vy2 = min(iw, x2 + PAD_VIEW), min(ih, y2 + PAD_VIEW)

# --- (a) input with detector box (dashed indigo) + ink box (solid green) ---
panel_a = src.crop((vx1, vy1, vx2, vy2)).copy()
draw = ImageDraw.Draw(panel_a)
ink = _ink_bbox(src, x1, y1, x2, y2)
tx1, ty1, tx2, ty2 = ink
LW = 6
INDIGO, GREEN = (67, 56, 202), (21, 128, 61)


def dashed_rect(d, box, color, width, dash=28, gap=18):
    bx1, by1, bx2, by2 = box
    for (sx, sy, ex, ey) in [(bx1, by1, bx2, by1), (bx1, by2, bx2, by2),
                             (bx1, by1, bx1, by2), (bx2, by1, bx2, by2)]:
        length = max(abs(ex - sx), abs(ey - sy))
        n = max(1, length // (dash + gap))
        for i in range(int(n) + 1):
            t0 = i * (dash + gap) / length
            t1 = min(1.0, (i * (dash + gap) + dash) / length)
            if t0 >= 1.0:
                break
            d.line([(sx + (ex - sx) * t0, sy + (ey - sy) * t0),
                    (sx + (ex - sx) * t1, sy + (ey - sy) * t1)],
                   fill=color, width=width)


dashed_rect(draw, (x1 - vx1, y1 - vy1, x2 - vx1, y2 - vy1), INDIGO, LW)
draw.rectangle((tx1 - vx1, ty1 - vy1, tx2 - vx1, ty2 - vy1), outline=GREEN, width=LW)

# --- (b) the erase mask, exactly as _erase_ink_region builds it ---
rx1, ry1 = max(0, x1 - ERASE_PAD), max(0, y1 - ERASE_PAD)
rx2, ry2 = min(iw, x2 + ERASE_PAD), min(ih, y2 + ERASE_PAD)
crop = np.array(src.crop((rx1, ry1, rx2, ry2)))
gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
_, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1))
mask = cv2.dilate(mask, kernel, iterations=1)
canvas = np.full((vy2 - vy1, vx2 - vx1), 255, np.uint8)          # white
canvas[ry1 - vy1:ry2 - vy1, rx1 - vx1:rx2 - vx1] = 255 - mask    # ink pixels black
panel_b = Image.fromarray(canvas).convert("RGB")

# --- (c) after the deployed erase (mutates a copy of the slide) ---
erased = src.copy()
_erase_ink_region(erased, x1, y1, x2, y2, pad=ERASE_PAD)
panel_c = erased.crop((vx1, vy1, vx2, vy2))

# --- (d) deployed pipeline output, same window ---
assert out.size == src.size, (out.size, src.size)
panel_d = out.crop((vx1, vy1, vx2, vy2))

dest = ROOT / "paper/ieee_access"
for name, img in [("fig_erase_a", panel_a), ("fig_erase_b", panel_b),
                  ("fig_erase_c", panel_c), ("fig_erase_d", panel_d)]:
    img.save(dest / f"{name}.png")
    print(f"wrote {dest / name}.png {img.size}")
