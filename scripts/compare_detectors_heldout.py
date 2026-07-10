"""Compare v2 vs v2_hires_w0 detectors on the 4 held-out Extremwertberechnungen pages.

Outputs:
  outputs/heldout_compare/{page}_v2.jpg          annotated v2 prediction
  outputs/heldout_compare/{page}_v2_hires_w0.jpg annotated v2_hires_w0 prediction
  outputs/heldout_compare/summary.json           per-page detection counts at multiple conf thresholds
"""

import json
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SLIDE_DIR = ROOT / "data/Dr_Judith_Jakob_Slides/11 Extremwertberechnungen in mehreren Variablen_kommentiert images"
OUT_DIR = ROOT / "outputs/heldout_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HELDOUT_PAGES = [7, 10, 15, 28]
MODELS = {
    "v2": ROOT / "runs/detect/runs/detect/jakob_finetune_v2/weights/best.pt",
    "v2_hires_w0": ROOT / "runs/detect/runs/detect/jakob_finetune_v2_hires_w0/weights/best.pt",
}
THRESHOLDS = [0.25, 0.4, 0.5]
CLASS_NAMES = {0: "text", 1: "math"}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 165, 255)}  # text=green, math=orange

summary = {}

for model_name, weights in MODELS.items():
    print(f"\n=== {model_name} ({weights.name}) ===")
    model = YOLO(str(weights))
    imgsz = 960 if "hires" in model_name else 640

    for page in HELDOUT_PAGES:
        img_path = SLIDE_DIR / f"11 Extremwertberechnungen in mehreren Variablen_kommentiert_page-{page:04d}.jpg"
        assert img_path.exists(), img_path
        results = model.predict(str(img_path), imgsz=imgsz, conf=0.05, verbose=False)
        r = results[0]
        boxes = r.boxes
        page_key = f"page_{page:04d}"
        if page_key not in summary:
            summary[page_key] = {}
        summary[page_key][model_name] = {}

        for thresh in THRESHOLDS:
            mask = boxes.conf >= thresh
            cls_ids = boxes.cls[mask].cpu().numpy().astype(int)
            confs = boxes.conf[mask].cpu().numpy()
            counts = {CLASS_NAMES[c]: int((cls_ids == c).sum()) for c in [0, 1]}
            mean_conf = float(confs.mean()) if len(confs) else 0.0
            summary[page_key][model_name][f"conf>={thresh}"] = {
                "text": counts["text"],
                "math": counts["math"],
                "total": int(mask.sum()),
                "mean_conf": round(mean_conf, 3),
            }

        # Annotate image at default conf=0.25 for visualization
        img = cv2.imread(str(img_path))
        for box in boxes:
            conf = float(box.conf[0])
            if conf < 0.25:
                continue
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            color = CLASS_COLORS[cls_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        out_path = OUT_DIR / f"{page_key}_{model_name}.jpg"
        cv2.imwrite(str(out_path), img)
        n_at_25 = summary[page_key][model_name]["conf>=0.25"]["total"]
        print(f"  page {page:04d}: {n_at_25} boxes @ conf>=0.25  -> {out_path.name}")

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nWrote summary to {OUT_DIR / 'summary.json'}")
print("\n=== Side-by-side at conf>=0.25 ===")
print(f"{'page':<10} {'v2 text':<8} {'v2 math':<8} {'v2 total':<10} {'hires text':<11} {'hires math':<11} {'hires total':<12}")
for page_key, models in summary.items():
    v2 = models["v2"]["conf>=0.25"]
    h = models["v2_hires_w0"]["conf>=0.25"]
    print(f"{page_key:<10} {v2['text']:<8} {v2['math']:<8} {v2['total']:<10} {h['text']:<11} {h['math']:<11} {h['total']:<12}")
