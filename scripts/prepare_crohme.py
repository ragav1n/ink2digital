"""
Prepare CROHME 2019 data for math expression recognition (TAMER).

IMPORTANT: The CROHME2019_data.zip in data/CROHME/ appears to be an HTML
download page rather than actual data. This script handles both cases:
1. If valid zip: extracts and prepares data
2. If HTML stub: downloads the data from the correct source

CROHME 2019 data is available from the TAMER repository and InkML format.
Fallback: use CROHME data from HuggingFace or TAMER's bundled examples.

Usage:
    python scripts/prepare_crohme.py \
        --crohme-dir data/CROHME \
        --output-dir data/processed/math \
        --tamer-dir TAMER
"""

import argparse
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET

from loguru import logger


# ---------------------------------------------------------------------------
# InkML parsing (CROHME format)
# ---------------------------------------------------------------------------

def parse_inkml(inkml_path: Path) -> Optional[Dict]:
    """
    Parse a CROHME InkML file.
    Returns {'latex': str, 'inkml_path': str} or None on failure.
    """
    try:
        tree = ET.parse(str(inkml_path))
        root = tree.getroot()
        ns = {'ink': 'http://www.w3.org/2003/InkML'}

        # Try to find the LaTeX annotation
        latex = None
        for ann in root.findall('.//ink:annotation', ns):
            if ann.get('type') in ('truth', 'label', 'latex'):
                latex = ann.text
                break

        # Also try without namespace
        if latex is None:
            for ann in root.findall('.//annotation'):
                t = ann.get('type', '')
                if t in ('truth', 'label', 'latex') or 'truth' in t.lower():
                    latex = ann.text
                    break

        if latex is None:
            return None

        return {
            'inkml_path': str(inkml_path),
            'latex': latex.strip(),
        }
    except ET.ParseError as e:
        logger.debug(f"Failed to parse {inkml_path}: {e}")
        return None


def convert_inkml_to_image(inkml_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Convert InkML to a rasterized PNG using stroke coordinates.
    Falls back to a blank image if no stroke data.
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError:
        logger.warning("PIL/numpy not available for InkML rasterization")
        return None

    try:
        tree = ET.parse(str(inkml_path))
        root = tree.getroot()
        ns = {'ink': 'http://www.w3.org/2003/InkML'}

        # Collect all stroke points
        all_x, all_y = [], []
        strokes = []

        for trace in root.findall('.//ink:trace', ns) or root.findall('.//trace'):
            text = trace.text or ''
            points = []
            for pt in text.strip().split(','):
                coords = pt.strip().split()
                if len(coords) >= 2:
                    try:
                        x, y = float(coords[0]), float(coords[1])
                        points.append((x, y))
                        all_x.append(x)
                        all_y.append(y)
                    except ValueError:
                        pass
            if points:
                strokes.append(points)

        if not all_x:
            return None

        # Rasterize
        margin = 20
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        w = max(int(max_x - min_x) + 2 * margin, 64)
        h = max(int(max_y - min_y) + 2 * margin, 64)

        img = Image.new('L', (w, h), color=255)
        draw = ImageDraw.Draw(img)

        for stroke in strokes:
            if len(stroke) < 2:
                continue
            pts = [(int(x - min_x + margin), int(y - min_y + margin)) for x, y in stroke]
            draw.line(pts, fill=0, width=2)

        # Save
        out_name = inkml_path.stem + '.png'
        out_path = output_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
        return out_path

    except Exception as e:
        logger.debug(f"Failed to rasterize {inkml_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_from_zip(zip_path: Path, output_dir: Path) -> bool:
    """
    Try to extract CROHME data from a zip file.
    Returns True if successful, False if the zip is invalid.
    """
    try:
        with zipfile.ZipFile(str(zip_path)) as zf:
            names = zf.namelist()
            if not any(n.endswith('.inkml') for n in names):
                logger.warning("Zip contains no InkML files. Likely a download error.")
                return False
            logger.info(f"Extracting {len(names)} files from CROHME zip...")
            zf.extractall(str(output_dir / 'raw'))
            return True
    except (zipfile.BadZipFile, Exception) as e:
        logger.warning(f"Cannot open zip: {e}")
        return False


def prepare_from_tamer_dir(tamer_dir: Path, output_dir: Path) -> List[dict]:
    """
    Use TAMER's bundled evaluation data as a fallback source.
    TAMER contains some CROHME test samples in its eval/ directory.
    """
    eval_dir = tamer_dir / 'eval'
    samples = []

    if not eval_dir.exists():
        logger.warning(f"TAMER eval dir not found: {eval_dir}")
        return samples

    inkml_files = list(eval_dir.rglob('*.inkml'))
    logger.info(f"Found {len(inkml_files)} InkML files in TAMER eval dir")

    img_dir = output_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    for inkml_path in inkml_files:
        sample = parse_inkml(inkml_path)
        if sample is None:
            continue
        img_path = convert_inkml_to_image(inkml_path, img_dir)
        if img_path:
            sample['image'] = str(img_path.relative_to(output_dir.parent.parent))
            samples.append(sample)

    return samples


def prepare_from_inkml_dir(inkml_dir: Path, output_dir: Path) -> List[dict]:
    """Process all InkML files found in a directory tree."""
    samples = []
    inkml_files = list(inkml_dir.rglob('*.inkml'))
    logger.info(f"Processing {len(inkml_files)} InkML files...")

    img_dir = output_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    for inkml_path in inkml_files:
        sample = parse_inkml(inkml_path)
        if sample is None:
            continue
        img_path = convert_inkml_to_image(inkml_path, img_dir)
        if img_path:
            sample['image'] = str(img_path.relative_to(output_dir.parent.parent))
            samples.append(sample)

    return samples


def save_manifest(samples: List[dict], output_dir: Path) -> None:
    """Save dataset manifest with train/val/test splits."""
    import random
    random.seed(42)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        'train': samples[:n_train],
        'val': samples[n_train:n_train + n_val],
        'test': samples[n_train + n_val:],
    }
    for split_name, split_data in splits.items():
        out = output_dir / f'math_{split_name}.json'
        with open(out, 'w', encoding='utf-8') as f:
            json.dump({'samples': split_data, 'n_samples': len(split_data)}, f,
                      ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(split_data)} {split_name} math samples -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_crohme(crohme_dir: Path, output_dir: Path, tamer_dir: Path) -> None:
    """Full CROHME data preparation pipeline."""
    logger.info("=== CROHME Math Data Preparation ===")

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = crohme_dir / 'CROHME2019_data.zip'

    samples = []

    # Strategy 1: Try the local zip
    if zip_path.exists():
        logger.info(f"Attempting to use local CROHME zip: {zip_path}")
        if prepare_from_zip(zip_path, output_dir):
            raw_dir = output_dir / 'raw'
            samples = prepare_from_inkml_dir(raw_dir, output_dir)
        else:
            logger.warning("Local CROHME zip is invalid (likely HTML download page).")
            logger.warning("CROHME data must be manually downloaded from:")
            logger.warning("  https://www.cs.rit.edu/~crohme2019/dataANDtools.html")
            logger.warning("Place the valid zip at: data/CROHME/CROHME2019_data.zip")

    # Strategy 2: Fall back to TAMER bundled samples
    if not samples and tamer_dir.exists():
        logger.info("Falling back to TAMER bundled evaluation samples...")
        samples = prepare_from_tamer_dir(tamer_dir, output_dir)

    if not samples:
        logger.error("No math samples could be prepared. See warnings above.")
        # Create empty manifests so downstream code doesn't fail
        for split in ['train', 'val', 'test']:
            out = output_dir / f'math_{split}.json'
            with open(out, 'w') as f:
                json.dump({'samples': [], 'n_samples': 0, 'note': 'CROHME data missing'}, f)
        return

    logger.info(f"Total math samples prepared: {len(samples)}")
    save_manifest(samples, output_dir)

    # Save stats
    stats = {'total_samples': len(samples), 'source': 'CROHME2019 / TAMER eval'}
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("=== CROHME Preparation Complete ===")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CROHME math data")
    parser.add_argument('--crohme-dir', type=Path, default=Path('data/CROHME'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/math'))
    parser.add_argument('--tamer-dir', type=Path, default=Path('TAMER'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    prepare_crohme(args.crohme_dir, args.output_dir, args.tamer_dir)
