"""
Prepare IAM Handwriting Database for German OCR training.

This script:
1. Extracts IAM line images (lines.tgz) and transcriptions (ascii.tgz)
2. Identifies German native speakers from writers.xml
3. Maps form IDs to writer IDs via XML metadata
4. Outputs a JSON manifest and organized image directory for training

Usage:
    python scripts/prepare_iam_german.py \
        --iam-dir data/iam_downloads \
        --output-dir data/processed/german_text

Output structure:
    data/processed/german_text/
    ├── images/           # Extracted line images
    ├── german_text_train.json
    ├── german_text_val.json
    └── german_text_test.json
"""

import argparse
import json
import random
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Optional
import shutil

from loguru import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# IAM writer IDs are 3-digit (000-xxx); writers.xml uses 10000+ format
# writer "000" -> writers.xml "10000", etc.
def iam_writer_id_to_xml(iam_id: str) -> str:
    """Convert IAM form writer ID (e.g. '000') to writers.xml ID (e.g. '10000')."""
    return str(10000 + int(iam_id))


# ---------------------------------------------------------------------------
# Parse writers.xml to find German speakers
# ---------------------------------------------------------------------------

def get_german_writer_ids(writers_xml_path: Path) -> Set[str]:
    """
    Parse writers.xml and return set of writer IDs (in XML format, e.g. '10000')
    who are native German or Swiss German speakers.
    """
    tree = ET.parse(str(writers_xml_path))
    root = tree.getroot()
    german_ids: Set[str] = set()

    for writer in root.findall('Writer'):
        wid = writer.get('name', '')
        native = writer.get('NativeLanguage', '')
        if 'German' in native:
            german_ids.add(wid)

    logger.info(f"Found {len(german_ids)} German native speakers in writers.xml")
    return german_ids


# ---------------------------------------------------------------------------
# Parse IAM XML to map form IDs to writer IDs
# ---------------------------------------------------------------------------

def parse_form_writer_mapping(xml_archive_path: Path) -> Dict[str, str]:
    """
    Extract form_id -> xml_writer_id mapping from IAM xml.tgz.
    E.g. {'a01-000u': '10000', 'a01-000x': '10001', ...}
    """
    mapping: Dict[str, str] = {}
    with tarfile.open(str(xml_archive_path), 'r:gz') as tar:
        for member in tar.getmembers():
            if not member.name.endswith('.xml'):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            try:
                content = f.read()
                root = ET.fromstring(content)
                form_id = root.get('id', '')
                iam_writer = root.get('writer-id', '')
                if form_id and iam_writer:
                    xml_writer_id = iam_writer_id_to_xml(iam_writer)
                    mapping[form_id] = xml_writer_id
            except ET.ParseError:
                continue

    logger.info(f"Parsed {len(mapping)} form->writer mappings from XML archive")
    return mapping


# ---------------------------------------------------------------------------
# Parse ASCII transcriptions
# ---------------------------------------------------------------------------

def parse_lines_transcriptions(ascii_archive_path: Path) -> Dict[str, str]:
    """
    Parse lines.txt from ascii.tgz.
    Returns dict: {line_id: transcription_text}
    E.g. {'a01-000u-00': 'A MOVE to stop Mr. Gaitskell from', ...}
    """
    transcriptions: Dict[str, str] = {}

    with tarfile.open(str(ascii_archive_path), 'r:gz') as tar:
        # Try with and without ./ prefix
        f = None
        for candidate in ['lines.txt', './lines.txt']:
            try:
                f = tar.extractfile(candidate)
                if f:
                    break
            except KeyError:
                continue
        if f is None:
            raise FileNotFoundError("lines.txt not found in ascii.tgz")
        for raw_line in f:
            line = raw_line.decode('utf-8', errors='replace').strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(' ')
            if len(parts) < 9:
                continue
            line_id = parts[0]
            # Transcription is the last field; words separated by |
            text = parts[-1].replace('|', ' ')
            transcriptions[line_id] = text

    logger.info(f"Parsed {len(transcriptions)} line transcriptions")
    return transcriptions


# ---------------------------------------------------------------------------
# Extract line images
# ---------------------------------------------------------------------------

def extract_line_images(
    lines_archive_path: Path,
    output_dir: Path,
    form_ids: Set[str],
) -> Dict[str, Path]:
    """
    Extract line images for specific form IDs from lines.tgz.
    Returns dict: {line_id: local_image_path}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: Dict[str, Path] = {}

    with tarfile.open(str(lines_archive_path), 'r:gz') as tar:
        for member in tar.getmembers():
            if not member.name.endswith('.png'):
                continue
            # Path format: ./a02/a02-000/a02-000-00.png
            parts = Path(member.name).parts
            if len(parts) < 3:
                continue
            filename = parts[-1]  # e.g. a02-000-00.png
            line_id = filename.replace('.png', '')  # e.g. a02-000-00
            # Form ID = first two components: a02-000
            form_id = '-'.join(line_id.split('-')[:2])  # e.g. a02-000

            if form_id not in form_ids:
                continue

            dest = output_dir / filename
            if not dest.exists():
                f = tar.extractfile(member)
                if f:
                    dest.write_bytes(f.read())
            extracted[line_id] = dest

    logger.info(f"Extracted {len(extracted)} line images for German writers")
    return extracted


# ---------------------------------------------------------------------------
# Build dataset manifest
# ---------------------------------------------------------------------------

def build_manifest(
    transcriptions: Dict[str, str],
    images: Dict[str, Path],
    form_writer_map: Dict[str, str],
    german_writer_ids: Set[str],
    output_dir: Path,
) -> List[dict]:
    """
    Build a list of sample dicts for German writers only.
    Each sample: {'image': str, 'text': str, 'language': 'de', 'writer_id': str}
    """
    samples: List[dict] = []
    skipped = 0

    for line_id, img_path in images.items():
        if line_id not in transcriptions:
            skipped += 1
            continue

        # Derive form_id from line_id (e.g. 'a02-000-00' -> 'a02-000')
        form_id = '-'.join(line_id.split('-')[:2])
        writer_xml_id = form_writer_map.get(form_id, '')

        if writer_xml_id not in german_writer_ids:
            continue

        text = transcriptions[line_id]
        samples.append({
            'image': str(img_path.relative_to(output_dir.parent.parent)),
            'text': text,
            'language': 'de',
            'writer_id': writer_xml_id,
            'line_id': line_id,
        })

    if skipped > 0:
        logger.warning(f"Skipped {skipped} images with no transcription")
    return samples


def split_and_save(samples: List[dict], output_dir: Path) -> None:
    """Split samples into train/val/test and save as JSON."""
    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]

    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        out_path = output_dir / f"german_text_{split_name}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'samples': split_data, 'n_samples': len(split_data)}, f,
                      ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(split_data)} {split_name} samples -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_iam_german(iam_dir: Path, output_dir: Path) -> None:
    """Full IAM German data preparation pipeline."""
    logger.info("=== IAM German Data Preparation ===")

    # Paths
    writers_xml = iam_dir / 'writers.xml'
    xml_archive = iam_dir / 'xml.tgz'
    ascii_archive = iam_dir / 'ascii.tgz'
    lines_archive = iam_dir / 'lines.tgz'
    images_dir = output_dir / 'images'

    # Validate inputs
    for p in [writers_xml, xml_archive, ascii_archive, lines_archive]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find German writers
    german_writer_ids = get_german_writer_ids(writers_xml)

    # Step 2: Map forms to writers
    logger.info("Parsing form->writer mapping from XML archive...")
    form_writer_map = parse_form_writer_mapping(xml_archive)

    # Find forms belonging to German writers
    german_form_ids: Set[str] = {
        fid for fid, wid in form_writer_map.items()
        if wid in german_writer_ids
    }
    logger.info(f"Found {len(german_form_ids)} forms from German writers")

    # Step 3: Parse transcriptions
    logger.info("Parsing line transcriptions...")
    transcriptions = parse_lines_transcriptions(ascii_archive)

    # Step 4: Extract line images for German forms
    logger.info(f"Extracting line images from {lines_archive}...")
    images = extract_line_images(lines_archive, images_dir, german_form_ids)

    if not images:
        logger.warning("No images extracted. Check that lines.tgz contains data.")
        return

    # Step 5: Build manifest
    logger.info("Building dataset manifest...")
    samples = build_manifest(transcriptions, images, form_writer_map,
                             german_writer_ids, output_dir)
    logger.info(f"Total German samples: {len(samples)}")

    # Step 6: Split and save
    split_and_save(samples, output_dir)

    # Save statistics
    stats = {
        'total_samples': len(samples),
        'n_german_writers': len(german_writer_ids),
        'n_german_forms': len(german_form_ids),
        'splits': {
            'train': int(len(samples) * TRAIN_SPLIT),
            'val': int(len(samples) * VAL_SPLIT),
            'test': int(len(samples) * TEST_SPLIT),
        }
    }
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved -> {output_dir / 'stats.json'}")
    logger.info("=== IAM German Data Preparation Complete ===")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare IAM German OCR data")
    parser.add_argument('--iam-dir', type=Path, default=Path('data/iam_downloads'),
                        help='Directory containing IAM archives and writers.xml')
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/german_text'),
                        help='Output directory for processed data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    prepare_iam_german(args.iam_dir, args.output_dir)
