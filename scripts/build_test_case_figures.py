#!/usr/bin/env python
"""Build side-by-side input/output figures for every test_cases/ slot.

For each slot under test_cases/NN_slug/:
  - input  = test_cases/NN_slug/slide.jpg
  - output = outputs/test_cases/NN_slug/slide.jpg
The two are pasted side-by-side at a uniform height with a header strip
labelling the variant (from notes.md's H1) and the source. Result is written
to figures/test_cases/NN_slug.png and a contact-sheet figures/test_cases/all.png.

Usage:
  python scripts/build_test_case_figures.py
"""
from __future__ import annotations
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "test_cases"
OUT_ROOT = ROOT / "outputs" / "test_cases"
FIG_ROOT = ROOT / "figures" / "test_cases"

PANEL_HEIGHT = 720
HEADER_HEIGHT = 80
GUTTER = 24
PADDING = 20
BG = (255, 255, 255)
LABEL_BG = (245, 245, 245)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def slot_title(notes_path: Path) -> str:
    if not notes_path.exists():
        return notes_path.parent.name
    for line in notes_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"#\s*\d+\s*[—-]\s*(.+)", line)
        if m:
            return m.group(1).strip()
    return notes_path.parent.name


def resize_to_height(img: Image.Image, h: int) -> Image.Image:
    w = max(1, round(img.width * (h / img.height)))
    return img.resize((w, h), Image.LANCZOS)


def make_panel(img_path: Path, label: str, h: int) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    img = resize_to_height(img, h)
    panel = Image.new("RGB", (img.width, h + 36), LABEL_BG)
    panel.paste(img, (0, 36))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), label, fill=(60, 60, 60), font=load_font(20))
    return panel


def build_pair(slot_dir: Path) -> Path | None:
    slug = slot_dir.name
    input_img = slot_dir / "slide.jpg"
    output_img = OUT_ROOT / slug / "slide.jpg"
    if not input_img.exists():
        return None
    title = slot_title(slot_dir / "notes.md")

    have_output = output_img.exists()
    left = make_panel(input_img, "Input", PANEL_HEIGHT)
    if have_output:
        right = make_panel(output_img, "Pipeline output", PANEL_HEIGHT)
    else:
        right = Image.new("RGB", (left.width, left.height), (255, 245, 245))
        d = ImageDraw.Draw(right)
        d.text((10, 8), "Pipeline output", fill=(60, 60, 60), font=load_font(20))
        d.text(
            (left.width // 2 - 120, left.height // 2 - 10),
            "(no output yet)",
            fill=(150, 60, 60),
            font=load_font(22),
        )

    total_w = PADDING * 2 + left.width + GUTTER + right.width
    total_h = HEADER_HEIGHT + left.height + PADDING
    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((PADDING, 18), f"{slug}", fill=(20, 20, 20), font=load_font(28))
    draw.text((PADDING, 52), title, fill=(80, 80, 80), font=load_font(22))
    canvas.paste(left, (PADDING, HEADER_HEIGHT))
    canvas.paste(right, (PADDING + left.width + GUTTER, HEADER_HEIGHT))

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = FIG_ROOT / f"{slug}.png"
    canvas.save(out_path, optimize=True)
    return out_path


def build_contact_sheet(fig_paths: list[Path]) -> Path | None:
    if not fig_paths:
        return None
    figs = [Image.open(p).convert("RGB") for p in fig_paths]
    target_w = max(f.width for f in figs)
    scaled = []
    for f in figs:
        if f.width != target_w:
            ratio = target_w / f.width
            f = f.resize((target_w, round(f.height * ratio)), Image.LANCZOS)
        scaled.append(f)
    total_h = sum(f.height for f in scaled) + GUTTER * (len(scaled) - 1)
    sheet = Image.new("RGB", (target_w, total_h), BG)
    y = 0
    for f in scaled:
        sheet.paste(f, (0, y))
        y += f.height + GUTTER
    out_path = FIG_ROOT / "all.png"
    sheet.save(out_path, optimize=True)
    return out_path


def main() -> None:
    slots = sorted(d for d in CASES.iterdir() if d.is_dir() and re.match(r"\d{2}_", d.name))
    figs: list[Path] = []
    for slot in slots:
        out = build_pair(slot)
        if out is not None:
            print(f"  wrote {out.relative_to(ROOT)}")
            figs.append(out)
    sheet = build_contact_sheet(figs)
    if sheet:
        print(f"  wrote {sheet.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
