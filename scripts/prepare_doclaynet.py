"""
Prepare DocLayNet dataset for handwriting detection training.

DocLayNet contains 80,863 annotated document pages (COCO format).
For our task we need:
- Images containing text/math regions
- Convert to YOLO format (class_id cx cy w h normalized)
- Map relevant DocLayNet classes to our 2-class schema: [text, math]

DocLayNet classes we use:
  - Text, Caption, Footnote, List-item, Paragraph, Section-header, Title -> 'text' (0)
  - Formula -> 'math' (1)
  (Others: Figure, Page-header, Page-footer, Table -> ignored)

STORAGE NOTE:
  DocLayNet_core.zip is 28GB. This script uses streaming extraction:
  it reads images directly from the zip and converts to YOLO format,
  so only the selected subset needs to be written to disk.

Usage:
    python scripts/prepare_doclaynet.py \
        --doclaynet-dir data/DocLayNet \
        --output-dir data/processed/detection \
        --max-images-per-split 5000   # Use 5000 per split for Phase 1

Output structure:
    data/processed/detection/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── stats.json
    └── dataset.yaml
"""

import argparse
import io
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from loguru import logger
from PIL import Image


# ---------------------------------------------------------------------------
# DocLayNet class mapping
# ---------------------------------------------------------------------------

DOCLAYNET_CLASS_MAP: Dict[str, Optional[int]] = {
    'Caption': 0,
    'Footnote': 0,
    'Formula': 1,
    'List-item': 0,
    'Page-footer': None,
    'Page-header': None,
    'Picture': None,
    'Section-header': 0,
    'Table': None,
    'Text': 0,
    'Title': 0,
}

CLASS_NAMES = ['text', 'math']


# ---------------------------------------------------------------------------
# COCO to YOLO conversion
# ---------------------------------------------------------------------------

def coco_bbox_to_yolo(
    bbox: List[float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    """Convert COCO [x, y, w, h] -> YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


# ---------------------------------------------------------------------------
# Streaming conversion (read from zip without full extraction)
# ---------------------------------------------------------------------------

def convert_split_streaming(
    zip_path: Path,
    annotation_member: str,
    output_images_dir: Path,
    output_labels_dir: Path,
    class_map: Dict[str, Optional[int]],
    max_images: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """
    Convert a DocLayNet split to YOLO format by streaming from zip.

    Reads images and annotations directly from the zip file,
    writing only the selected subset to disk. No full extraction needed.

    Args:
        zip_path: Path to DocLayNet_core.zip.
        annotation_member: Member name of COCO JSON inside zip.
        output_images_dir: Directory to write selected images.
        output_labels_dir: Directory to write YOLO label files.
        class_map: DocLayNet class -> our class ID mapping.
        max_images: Maximum images to process per split.
        seed: Random seed for reproducible subset selection.

    Returns:
        Stats dict.
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(zip_path)) as zf:
        # Load annotations
        logger.info(f"Loading annotations: {annotation_member}")
        with zf.open(annotation_member) as f:
            coco = json.load(f)

        # Build category ID -> our class ID mapping
        cat_id_map: Dict[int, Optional[int]] = {}
        for cat in coco.get('categories', []):
            name = cat['name']
            cat_id_map[cat['id']] = class_map.get(name)

        # Build image ID -> info mapping
        images: Dict[int, dict] = {img['id']: img for img in coco.get('images', [])}

        # Group annotations by image
        ann_by_image: Dict[int, List[dict]] = {}
        for ann in coco.get('annotations', []):
            ann_by_image.setdefault(ann['image_id'], []).append(ann)

        # Filter images that have at least one valid annotation
        valid_image_ids = []
        for img_id, anns in ann_by_image.items():
            has_valid = any(cat_id_map.get(a['category_id']) is not None for a in anns)
            if has_valid:
                valid_image_ids.append(img_id)

        logger.info(f"Found {len(valid_image_ids)} images with valid annotations")

        # Subsample if needed
        random.seed(seed)
        random.shuffle(valid_image_ids)
        if max_images:
            valid_image_ids = valid_image_ids[:max_images]
        logger.info(f"Processing {len(valid_image_ids)} images (max={max_images})")

        # Build a set of PNG member names for fast lookup
        all_members = set(zf.namelist())

        n_processed = 0
        n_skipped = 0
        n_boxes = 0

        for img_id in valid_image_ids:
            img_info = images[img_id]
            img_filename = img_info['file_name']
            img_w = img_info['width']
            img_h = img_info['height']

            # The PNG filename in the zip: PNG/{filename}
            img_stem = Path(img_filename).name
            zip_member = f"PNG/{img_stem}"

            if zip_member not in all_members:
                n_skipped += 1
                continue

            # Build YOLO label lines
            annotations = ann_by_image.get(img_id, [])
            yolo_lines = []
            for ann in annotations:
                our_class = cat_id_map.get(ann['category_id'])
                if our_class is None:
                    continue
                cx, cy, nw, nh = coco_bbox_to_yolo(ann['bbox'], img_w, img_h)
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                if nw < 0.001 or nh < 0.001:
                    continue
                yolo_lines.append(f"{our_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                n_boxes += 1

            if not yolo_lines:
                n_skipped += 1
                continue

            # Extract and save image
            stem = Path(img_stem).stem
            dest_img = output_images_dir / f"{stem}.png"
            if not dest_img.exists():
                with zf.open(zip_member) as img_f:
                    img_data = img_f.read()
                dest_img.write_bytes(img_data)

            # Write label file
            label_file = output_labels_dir / f"{stem}.txt"
            label_file.write_text('\n'.join(yolo_lines))
            n_processed += 1

            if n_processed % 500 == 0:
                logger.info(f"  Progress: {n_processed}/{len(valid_image_ids)} images")

    return {
        'n_processed': n_processed,
        'n_skipped': n_skipped,
        'n_boxes': n_boxes,
    }


def write_dataset_yaml(output_dir: Path, class_names: List[str]) -> None:
    """Write YOLO dataset configuration YAML."""
    yaml_content = f"""# DocLayNet detection dataset (YOLO format)
# Generated by prepare_doclaynet.py

path: {output_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = output_dir / 'dataset.yaml'
    yaml_path.write_text(yaml_content)
    logger.info(f"Dataset YAML written -> {yaml_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_doclaynet(
    doclaynet_dir: Path,
    output_dir: Path,
    max_images_per_split: Optional[int] = None,
) -> None:
    """Full DocLayNet preparation pipeline using streaming extraction."""
    logger.info("=== DocLayNet Detection Data Preparation ===")

    zip_path = doclaynet_dir / 'DocLayNet_core.zip'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        logger.error(f"DocLayNet zip not found: {zip_path}")
        return

    # Verify zip is valid
    try:
        with zipfile.ZipFile(str(zip_path)) as zf:
            members = zf.namelist()
        logger.info(f"DocLayNet zip is valid. Total entries: {len(members)}")
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid DocLayNet zip: {e}")
        return

    # DocLayNet COCO annotation members
    split_annotations = {
        'train': 'COCO/train.json',
        'val': 'COCO/val.json',
        'test': 'COCO/test.json',
    }

    all_stats = {}
    for split_name, ann_member in split_annotations.items():
        logger.info(f"\nProcessing {split_name} split...")
        img_out = output_dir / 'images' / split_name
        lbl_out = output_dir / 'labels' / split_name

        # Check if already done
        existing = len(list(img_out.glob('*.png'))) if img_out.exists() else 0
        if existing > 0:
            logger.info(f"  {split_name}: already has {existing} images, skipping")
            all_stats[split_name] = {'n_processed': existing, 'skipped': True}
            continue

        stats = convert_split_streaming(
            zip_path=zip_path,
            annotation_member=ann_member,
            output_images_dir=img_out,
            output_labels_dir=lbl_out,
            class_map=DOCLAYNET_CLASS_MAP,
            max_images=max_images_per_split,
        )
        all_stats[split_name] = stats
        logger.info(f"  {split_name}: {stats['n_processed']} images, "
                    f"{stats['n_boxes']} boxes, {stats['n_skipped']} skipped")

    # Write dataset YAML
    write_dataset_yaml(output_dir, CLASS_NAMES)

    # Save stats
    stats_out = {
        'splits': all_stats,
        'class_names': CLASS_NAMES,
        'max_images_per_split': max_images_per_split,
    }
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats_out, f, indent=2)

    logger.info("\n=== DocLayNet Preparation Complete ===")
    for split, s in all_stats.items():
        logger.info(f"  {split}: {s.get('n_processed', 0)} images, "
                    f"{s.get('n_boxes', 0)} boxes")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DocLayNet for detection training")
    parser.add_argument('--doclaynet-dir', type=Path, default=Path('data/DocLayNet'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/detection'))
    parser.add_argument('--max-images-per-split', type=int, default=5000,
                        help='Max images per split (default 5000 for Phase 1). '
                             'Use None for full dataset.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    prepare_doclaynet(
        doclaynet_dir=args.doclaynet_dir,
        output_dir=args.output_dir,
        max_images_per_split=args.max_images_per_split,
    )
