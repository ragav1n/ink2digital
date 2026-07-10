"""
Convert data/jakob_math/transcriptions.json into TAMER's training format.

Output: data/jakob_math_tamer/
    dictionary.txt           (copied from TAMER/lightning_logs/version_1)
    train/images.pkl         (dict fname -> grayscale uint8 np.ndarray [H,W])
    train/caption.txt        (lines: "fname tok1 tok2 ... tokN")
    val/images.pkl
    val/caption.txt

Auto-sanitizes 21 of the 30 entries to use only HME100K vocab tokens.
The other 9 entries (with \\prod / \\pmod / \\bmod / \\exists / \\substack)
are dropped — TAMER's vocab can't represent them.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from loguru import logger


# Sanitization rules — applied in order ---------------------------------------

DROP_PLAIN = ['\\left', '\\right', '\\quad', '\\qquad', '\\,', '\\;', '\\!', '\\:']
UNWRAP = ['\\text', '\\operatorname', '\\mathbb', '\\mathrm']  # \cmd{x} -> x
SUBST = [
    (r'\\le\b',     r'\\leq'),
    (r'\\iff\b',    r'\\Leftrightarrow'),
    (r'\\ell\b',    'l'),
    (r'\\ldots\b',  r'\\cdots'),
    (r'\\\\',       ' '),
    (r"'",          r' \\prime '),
]
HARD_OOV = ['\\prod', '\\pmod', '\\bmod', '\\exists', '\\substack']


def sanitize(s: str) -> str:
    for cmd in DROP_PLAIN:
        s = s.replace(cmd, ' ')
    for cmd in UNWRAP:
        s = re.sub(re.escape(cmd) + r'\{([^{}]*)\}', r'\1', s)
        s = s.replace(cmd, '')
    for pat, rep in SUBST:
        s = re.sub(pat, rep, s)
    s = re.sub(r'\\substack\s*\{[^{}]*\}', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# Tokenizer — TAMER captions are whitespace-separated --------------------------

TOK_RE = re.compile(r'\\[A-Za-z]+\*?|\\.|\{|\}|\[|\]|\(|\)|[\^_+\-=<>/.,!|;:\']|[0-9]|[A-Za-z]')


def tokenize(latex: str) -> list[str]:
    return TOK_RE.findall(latex)


def load_image_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=Path, default=Path('data/jakob_math'))
    p.add_argument('--dst', type=Path, default=Path('data/jakob_math_tamer'))
    p.add_argument('--dict', type=Path,
                   default=Path('TAMER/lightning_logs/version_1/dictionary.txt'))
    p.add_argument('--val-frac', type=float, default=0.18)  # 4 of 21
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    with open(args.dict) as f:
        vocab = {line.strip() for line in f if line.strip()}

    with open(args.src / 'transcriptions.json') as f:
        data = json.load(f)

    # Auto-sanitize and filter
    usable, dropped = [], []
    for e in data:
        cleaned = sanitize(e['latex'])
        tokens = tokenize(cleaned)
        oov = [t for t in tokens if t not in vocab]
        hard = [t for t in tokens if any(h in t for h in HARD_OOV)]
        if hard or oov:
            dropped.append({'image': e['image'], 'orig': e['latex'],
                            'cleaned': cleaned, 'oov': sorted(set(oov))})
        else:
            usable.append({'image': e['image'], 'latex_orig': e['latex'],
                           'latex_clean': cleaned, 'tokens': tokens})

    logger.info(f"Usable: {len(usable)}, dropped: {len(dropped)}")
    for d in dropped:
        logger.info(f"  DROPPED {d['image']}: OOV={d['oov']}")

    if not usable:
        raise SystemExit("No usable entries — aborting.")

    # Split
    rng = random.Random(args.seed)
    indices = list(range(len(usable)))
    rng.shuffle(indices)
    n_val = max(1, round(len(usable) * args.val_frac))
    val_idx = set(indices[:n_val])
    train = [usable[i] for i in range(len(usable)) if i not in val_idx]
    val = [usable[i] for i in sorted(val_idx)]
    logger.info(f"Split: train={len(train)} val={len(val)}")

    # Write TAMER format
    args.dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.dict, args.dst / 'dictionary.txt')

    for split_name, split_data in [('train', train), ('val', val)]:
        d = args.dst / split_name
        d.mkdir(exist_ok=True)
        images = {}
        with open(d / 'caption.txt', 'w', encoding='utf-8') as f:
            for entry in split_data:
                img_path = args.src.parent / entry['image']
                fname = Path(entry['image']).stem  # e.g. math_001
                arr = load_image_gray(img_path)
                images[fname] = arr
                f.write(fname + ' ' + ' '.join(entry['tokens']) + '\n')
        with open(d / 'images.pkl', 'wb') as f:
            pickle.dump(images, f)
        logger.info(f"  Wrote {split_name}: {len(images)} samples -> {d}")

    # Write a summary alongside
    with open(args.dst / 'preparation_summary.json', 'w') as f:
        json.dump({
            'usable_count': len(usable),
            'dropped_count': len(dropped),
            'train_count': len(train),
            'val_count': len(val),
            'dropped_entries': dropped,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Done. Output -> {args.dst}")


if __name__ == '__main__':
    main()
