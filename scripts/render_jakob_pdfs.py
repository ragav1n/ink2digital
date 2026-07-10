"""Render the 24 Jakob lecture PDFs to per-page JPGs at 4000x2250.

Output layout matches the existing `11 Extremwertberechnungen` JPG dir
(confirmed via Image.open(...).size == (4000, 2250)), so the batch infer
driver can treat every lecture identically.

Idempotent: a lecture is skipped if its output dir already contains a JPG
count matching the source PDF's page count.

Usage:
    python scripts/render_jakob_pdfs.py
    python scripts/render_jakob_pdfs.py --only "04 Laplace-Transformation_kommentiert.pdf"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF

SRC_DIRS = [
    Path("data/Dr_Judith_Jakob_Slides/Mathematics for Computer Science 2"),
    Path("data/Dr_Judith_Jakob_Slides/Mathematics for Computer Science 4"),
]
DEST_ROOT = Path("data/jakob_full_corpus")
TARGET_W, TARGET_H = 4000, 2250


def slug_for(pdf_path: Path) -> str:
    stem = pdf_path.stem
    if stem.endswith("_kommentiert"):
        stem = stem[: -len("_kommentiert")]
    return stem.replace(" ", "_")


def render_pdf(pdf_path: Path, out_dir: Path) -> tuple[int, int]:
    """Render every page of `pdf_path` to `out_dir/<slug>_page-XXXX.jpg`.

    Returns (rendered, skipped).
    """
    slug = out_dir.name
    doc = fitz.open(pdf_path)
    n_pages = doc.page_count

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(f"{slug}_page-*.jpg"))
    if len(existing) == n_pages:
        doc.close()
        return 0, n_pages

    rendered = 0
    for i, page in enumerate(doc, start=1):
        out_path = out_dir / f"{slug}_page-{i:04d}.jpg"
        if out_path.exists():
            continue
        rect = page.rect
        matrix = fitz.Matrix(TARGET_W / rect.width, TARGET_H / rect.height)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(out_path, jpg_quality=92)
        rendered += 1
    doc.close()
    return rendered, n_pages - rendered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="Render only the named PDF (filename, with extension).")
    args = ap.parse_args()

    pdfs: list[Path] = []
    for d in SRC_DIRS:
        if not d.is_dir():
            print(f"WARN: missing source dir {d}", file=sys.stderr)
            continue
        pdfs.extend(sorted(d.glob("*.pdf")))

    if args.only:
        pdfs = [p for p in pdfs if p.name == args.only]
        if not pdfs:
            print(f"ERROR: no PDF matching --only {args.only!r}", file=sys.stderr)
            return 2

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(pdfs)} PDF(s) to consider.")
    for pdf in pdfs:
        slug = slug_for(pdf)
        out_dir = DEST_ROOT / slug
        rendered, skipped = render_pdf(pdf, out_dir)
        print(f"  {slug}: rendered={rendered}, skipped={skipped} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
