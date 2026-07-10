"""Draw the fixed-seed correctness-audit sample (Reviewer 2, comment 2).

Samples 100 of the corpus's rendered math regions (rendered==true in each
lecture's results.json) with random.Random(42), then writes:

  outputs/audit_sample/audit_sample.csv       one row per region, verdict
                                              column empty — to be filled by
                                              the judging author with
                                              correct | minor | meaning-changing
  outputs/audit_sample/sheets/sheet_NN.png    contact sheets, 5 regions per
                                              sheet: original ink crop above,
                                              deployed composite output below

Judging protocol (runbook §3d): compare the typeset replacement against the
source ink; (a) fully correct, (b) minor meaning-preserving deviation,
(c) meaning-changing error.

Usage: .venv/bin/python scripts/make_audit_sample.py
"""
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CORPUS_OUT = ROOT / "outputs/jakob_full_corpus"
DEST = ROOT / "outputs/audit_sample"
SHEETS = DEST / "sheets"
SHEETS.mkdir(parents=True, exist_ok=True)

SEED = 42
N = 100
MARGIN = 25          # context around the detector box in each crop
PAIR_W = 1600        # normalised crop width on the sheet
PER_SHEET = 5

population = []
for rj in sorted(CORPUS_OUT.glob("*/results.json")):
    lecture = rj.parent.name
    results = json.load(open(rj))
    for slide_key, regions in sorted(results.items()):
        for idx, reg in enumerate(regions):
            if reg.get("type") == "math" and reg.get("rendered"):
                population.append((lecture, slide_key, idx, reg))
print(f"population: {len(population)} rendered math regions")

sample = random.Random(SEED).sample(population, N)
sample.sort(key=lambda s: (s[0], s[1], s[2]))

try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    FONT_SM = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
except OSError:
    FONT = FONT_SM = ImageFont.load_default()


def crop_pair(lecture, slide_key, reg):
    page = Path(slide_key).name
    src = Image.open(ROOT / slide_key).convert("RGB")
    out = Image.open(CORPUS_OUT / lecture / page).convert("RGB")
    x1, y1, x2, y2 = (int(v) for v in reg["bbox"])
    iw, ih = src.size
    box = (max(0, x1 - MARGIN), max(0, y1 - MARGIN),
           min(iw, x2 + MARGIN), min(ih, y2 + MARGIN))
    crops = []
    for im in (src, out):
        c = im.crop(box)
        s = PAIR_W / c.width
        crops.append(c.resize((PAIR_W, max(1, int(c.height * s))), Image.LANCZOS))
    return crops


rows = []
sheet_imgs = []
for i, (lecture, slide_key, idx, reg) in enumerate(sample, 1):
    rid = f"A{i:03d}"
    page = Path(slide_key).name
    rows.append({
        "id": rid, "lecture": lecture, "page": page, "region_index": idx,
        "bbox": ";".join(str(int(v)) for v in reg["bbox"]),
        "confidence": f"{reg['confidence']:.3f}", "latex": reg["latex"],
        "verdict": "", "notes": "",
    })
    ink, comp = crop_pair(lecture, slide_key, reg)
    label_h, gap = 44, 14
    tile = Image.new("RGB", (PAIR_W, label_h + ink.height + comp.height + gap + 8),
                     (255, 255, 255))
    d = ImageDraw.Draw(tile)
    d.text((4, 4), f"{rid}   {lecture} / {page}  region {idx}", fill=(0, 0, 0),
           font=FONT)
    tile.paste(ink, (0, label_h))
    d.line([(0, label_h + ink.height + gap // 2),
            (PAIR_W, label_h + ink.height + gap // 2)], fill=(160, 160, 160),
           width=2)
    tile.paste(comp, (0, label_h + ink.height + gap))
    sheet_imgs.append(tile)

for s in range(0, len(sheet_imgs), PER_SHEET):
    batch = sheet_imgs[s:s + PER_SHEET]
    sep = 26
    h = sum(t.height for t in batch) + sep * (len(batch) - 1)
    sheet = Image.new("RGB", (PAIR_W, h), (120, 120, 120))
    y = 0
    for t in batch:
        sheet.paste(t, (0, y))
        y += t.height + sep
    name = SHEETS / f"sheet_{s // PER_SHEET + 1:02d}.png"
    sheet.save(name)

with open(DEST / "audit_sample.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

per_lecture = {}
for lecture, *_ in sample:
    per_lecture[lecture] = per_lecture.get(lecture, 0) + 1
print(f"sampled {N} with seed {SEED}; per lecture: {per_lecture}")
print(f"wrote {DEST/'audit_sample.csv'} and {len(range(0, N, PER_SHEET))} sheets")
