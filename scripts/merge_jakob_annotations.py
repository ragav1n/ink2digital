"""Merge jakob_annotations (old) + jakob_annotations_v2 (new, class-flipped) into a single YOLO dataset.

Old: text=0, math=1, 52 slides, 153 boxes
New: math=0, text=1, 113 slides (112 with boxes + 1 negative), 747 boxes — class IDs flipped on copy

Output: data/processed/jakob_detection_v2/{images,labels}/{train,val}/
"""
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = ROOT / "data/jakob_annotations"
NEW_DIR = ROOT / "data/jakob_annotations_v2"
OUT_DIR = ROOT / "data/processed/jakob_detection_v2"

VAL_FRAC = 0.2
SEED = 42


def collect_old():
    """Old format: jpg and txt sit side-by-side at top of folder, classes already text=0, math=1."""
    pairs = []
    for jpg in OLD_DIR.glob("*.jpg"):
        txt = jpg.with_suffix(".txt")
        if txt.exists():
            pairs.append((jpg, txt, "old"))
    return pairs


def collect_new():
    """New format: images/<name>.jpg + labels/<name>.txt, classes are math=0/text=1 — must flip."""
    pairs = []
    for jpg in (NEW_DIR / "images").glob("*.jpg"):
        txt = NEW_DIR / "labels" / (jpg.stem + ".txt")
        if txt.exists():
            pairs.append((jpg, txt, "new"))
    return pairs


def write_label(src_txt: Path, dst_txt: Path, flip_classes: bool):
    """Copy a YOLO label file. If flip_classes, swap class id 0<->1."""
    lines = []
    for raw in src_txt.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        if cls in ("text", "math"):
            continue  # stray classes.txt-style noise — skip
        if flip_classes:
            cls = "1" if cls == "0" else "0"
        lines.append(" ".join([cls] + parts[1:]))
    dst_txt.write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    pairs = collect_old() + collect_new()
    random.Random(SEED).shuffle(pairs)

    n_val = max(1, int(len(pairs) * VAL_FRAC))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    print(f"Total slides:      {len(pairs)}")
    print(f"  old: {sum(1 for _,_,s in pairs if s=='old')}")
    print(f"  new: {sum(1 for _,_,s in pairs if s=='new')}")
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    # Wipe and recreate output directories
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = OUT_DIR / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    text_n, math_n = 0, 0
    for split, group in (("train", train_pairs), ("val", val_pairs)):
        for jpg, txt, source in group:
            # Use slide stem for name; new-batch UUIDs already unique, old names unique too
            stem = jpg.stem
            dst_img = OUT_DIR / "images" / split / f"{stem}.jpg"
            dst_lbl = OUT_DIR / "labels" / split / f"{stem}.txt"
            shutil.copy2(jpg, dst_img)
            write_label(txt, dst_lbl, flip_classes=(source == "new"))

            for line in dst_lbl.read_text().splitlines():
                if line.startswith("0 "):
                    text_n += 1
                elif line.startswith("1 "):
                    math_n += 1

    print(f"\nFinal merged box counts (text=0, math=1):")
    print(f"  text: {text_n}")
    print(f"  math: {math_n}")
    print(f"  total: {text_n + math_n}")
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
