"""Build the GT-vs-prediction detection figure (Reviewer 3, comment 5.1).

Runs the deployed YOLOv8x checkpoint (jakob_detector_v3/hires1536-3) on the
v2 validation split at the deployed settings (imgsz=1536, conf 0.35,
IoU 0.45), matches predictions to ground truth at IoU >= 0.5, prints a
per-slide table, and renders annotated slides:
  ground truth  = solid slate-gray boxes
  predictions   = dashed boxes, green=text / magenta=math, with confidence

Usage:
  .venv/bin/python scripts/make_fig_detection_examples.py            # stats + all annotated slides
  .venv/bin/python scripts/make_fig_detection_examples.py <stem>...  # only these slides
Annotated slides go to outputs/fig_detection/; copy chosen ones to the paper.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs/detect/runs/jakob_detector_v3/hires1536-3/weights/best.pt"
VAL_IMAGES = sorted((ROOT / "data/processed/jakob_detection_v2/images/val").glob("*"))
LABELS = ROOT / "data/processed/jakob_detection_v2/labels/val"
OUT = ROOT / "outputs/fig_detection"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["text", "math"]
SLATE = (71, 85, 105)
CLASS_COL = {0: (21, 128, 61), 1: (162, 28, 175)}  # text green, math magenta


def load_gt(stem, w, h):
    boxes = []
    for line in (LABELS / f"{stem}.txt").read_text().strip().splitlines():
        c, cx, cy, bw, bh = (float(v) for v in line.split())
        boxes.append((int(c), (cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def dashed_rect(d, box, color, width, dash=30, gap=20):
    bx1, by1, bx2, by2 = box
    for (sx, sy, ex, ey) in [(bx1, by1, bx2, by1), (bx1, by2, bx2, by2),
                             (bx1, by1, bx1, by2), (bx2, by1, bx2, by2)]:
        length = max(abs(ex - sx), abs(ey - sy), 1)
        i = 0
        while i * (dash + gap) < length:
            t0 = i * (dash + gap) / length
            t1 = min(1.0, (i * (dash + gap) + dash) / length)
            d.line([(sx + (ex - sx) * t0, sy + (ey - sy) * t0),
                    (sx + (ex - sx) * t1, sy + (ey - sy) * t1)],
                   fill=color, width=width)
            i += 1


only = set(sys.argv[1:])
model = YOLO(str(WEIGHTS))
print(f"{'slide':55s} GT  TP  FN  FP")
for img_path in VAL_IMAGES:
    stem = img_path.stem
    if only and stem not in only:
        continue
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    res = model.predict(str(img_path), imgsz=1536, conf=0.35, iou=0.45,
                        verbose=False)[0]
    preds = [(int(c), *xyxy, float(cf)) for c, xyxy, cf in
             zip(res.boxes.cls.tolist(),
                 res.boxes.xyxy.tolist(),
                 res.boxes.conf.tolist())]
    gts = load_gt(stem, w, h)

    matched_gt, matched_pred = set(), set()
    pairs = sorted(((iou(g[1:], p[1:5]), gi, pi)
                    for gi, g in enumerate(gts) for pi, p in enumerate(preds)
                    if g[0] == p[0]), reverse=True)
    for ov, gi, pi in pairs:
        if ov < 0.5 or gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
    fn = len(gts) - len(matched_gt)
    fp = len(preds) - len(matched_pred)
    print(f"{stem:55s} {len(gts):2d}  {len(matched_gt):2d}  {fn:2d}  {fp:2d}")

    LW = max(4, w // 700)
    draw = ImageDraw.Draw(im)
    for c, *box in gts:
        draw.rectangle(box, outline=SLATE, width=LW)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", w // 48)
    except OSError:
        font = ImageFont.load_default()
    for pi, (c, x1, y1, x2, y2, cf) in enumerate(preds):
        col = CLASS_COL[c]
        dashed_rect(draw, (x1, y1, x2, y2), col, LW)
        tag = f"{CLASS_NAMES[c]} {cf:.2f}"
        tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
        ty = y1 - th - 6 if y1 - th - 6 > 0 else y2 + 4
        draw.rectangle((x1, ty - 2, x1 + tw + 8, ty + th + 2), fill=(255, 255, 255))
        draw.text((x1 + 4, ty), tag, fill=col, font=font)
    im.save(OUT / f"{stem}_annotated.jpg", quality=92)
