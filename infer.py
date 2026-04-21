"""
infer.py — German lecture slide OCR inference

Detects handwritten text/math regions and replaces them with typeset equivalents.

Usage:
    python infer.py --image slide.png --output result.png

    # With professor adaptation (n-shot):
    python infer.py --image slide.png --output result.png \
        --adapt-samples professor_samples.json --n-shot 5

    # Process a directory:
    python infer.py --image-dir slides/ --output-dir results/

    # Use a specific OCR checkpoint:
    python infer.py --image slide.png --output result.png \
        --meta-checkpoint checkpoint/maml_ocr/meta_checkpoint_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Typesetting helpers
# ---------------------------------------------------------------------------

def _get_font(size: int):
    """Return a PIL font at the given size, falling back to default."""
    from PIL import ImageFont
    candidates = [
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
        'C:/Windows/Fonts/arial.ttf',
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _fit_font_size(text: str, box_w: int, box_h: int, min_size: int = 8, max_size: int = 36) -> int:
    """Find largest font size where text fits inside box_w x box_h."""
    from PIL import ImageDraw, Image as PILImage
    dummy = PILImage.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy)
    for size in range(max_size, min_size - 1, -2):
        font = _get_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= box_w and th <= box_h:
            return size
    return min_size


def render_text_in_box(
    image_pil,
    x1: int, y1: int, x2: int, y2: int,
    text: str,
    bg_color=(255, 255, 255),
    text_color=(10, 10, 120),
    padding: int = 4,
):
    """White out box and render typeset text inside it."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image_pil)

    # White out the region
    draw.rectangle([x1, y1, x2, y2], fill=bg_color, outline=(200, 200, 200))

    box_w = max(x2 - x1 - 2 * padding, 10)
    box_h = max(y2 - y1 - 2 * padding, 10)
    font_size = _fit_font_size(text, box_w, box_h)
    font = _get_font(font_size)

    # Wrap text if needed
    words = text.split()
    lines = []
    current = ''
    draw2 = draw  # reuse
    for word in words:
        test = (current + ' ' + word).strip()
        bb = draw2.textbbox((0, 0), test, font=font)
        if bb[2] - bb[0] <= box_w:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if not lines:
        lines = [text]

    # Draw lines
    line_h = font_size + 2
    for i, line in enumerate(lines):
        ty = y1 + padding + i * line_h
        if ty + line_h > y2:
            break
        draw.text((x1 + padding, ty), line, fill=text_color, font=font)


def render_latex_in_box(
    image_pil,
    x1: int, y1: int, x2: int, y2: int,
    latex: str,
    bg_color=(255, 245, 255),
    padding: int = 4,
):
    """Render LaTeX math into box using matplotlib, with text fallback."""
    box_w = x2 - x1
    box_h = y2 - y1

    rendered = _render_latex_to_array(latex, box_w, box_h)
    if rendered is not None:
        from PIL import Image as PILImage
        latex_img = PILImage.fromarray(rendered).resize((box_w, box_h), PILImage.LANCZOS)
        from PIL import ImageDraw
        ImageDraw.Draw(image_pil).rectangle([x1, y1, x2, y2], fill=bg_color)
        image_pil.paste(latex_img, (x1, y1))
    else:
        # Fallback: render as plain text
        display = f'$  {latex}  $'
        render_text_in_box(image_pil, x1, y1, x2, y2, display,
                           bg_color=bg_color, text_color=(120, 0, 120), padding=padding)


def _render_latex_to_array(latex: str, width: int, height: int) -> Optional[np.ndarray]:
    """Render LaTeX using matplotlib. Returns RGB array or None on failure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        dpi = 100
        fig_w = max(width / dpi, 0.5)
        fig_h = max(height / dpi, 0.3)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.axis('off')
        fig.patch.set_facecolor('white')

        expr = latex.strip()
        if not (expr.startswith('$') or expr.startswith('\\begin')):
            expr = f'${expr}$'

        ax.text(0.5, 0.5, expr, transform=ax.transAxes,
                ha='center', va='center', fontsize=12,
                usetex=False)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return buf
    except Exception as e:
        logger.debug(f"LaTeX render failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Line segmentation for multi-line handwritten regions
# ---------------------------------------------------------------------------

def segment_lines(region: np.ndarray, min_line_height: int = 15, gap_threshold: int = 3) -> List[tuple]:
    """
    Split a multi-line handwriting region into (y_start, y_end) row spans
    using horizontal projection profiles.

    Returns list of (y0, y1) tuples in region-local coordinates.
    """
    import cv2
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    proj = np.sum(binary > 0, axis=1).astype(int)

    # Smooth with a small window to bridge tiny gaps within a line
    kernel = np.ones(gap_threshold * 2 + 1) / (gap_threshold * 2 + 1)
    proj_smooth = np.convolve(proj, kernel, mode='same')

    lines = []
    in_line = False
    start = 0
    threshold = max(region.shape[1] * 0.005, 3)  # at least 0.5% of row width has ink

    for i, val in enumerate(proj_smooth):
        if not in_line and val >= threshold:
            in_line = True
            start = i
        elif in_line and val < threshold:
            in_line = False
            if i - start >= min_line_height:
                # Add a small vertical padding
                pad = max(4, int((i - start) * 0.1))
                lines.append((max(0, start - pad), min(region.shape[0], i + pad)))
    if in_line and region.shape[0] - start >= min_line_height:
        pad = max(4, int((region.shape[0] - start) * 0.1))
        lines.append((max(0, start - pad), region.shape[0]))

    return lines


def is_multiline(bbox: List[float], img_h: int, img_w: int) -> bool:
    """Heuristic: region is likely multi-line if its height exceeds ~5% of image height."""
    x1, y1, x2, y2 = bbox
    region_h = y2 - y1
    region_w = x2 - x1
    aspect = region_h / max(region_w, 1)
    return region_h > img_h * 0.05 and aspect > 0.15


def is_likely_handwriting(region: np.ndarray, bbox: List[float], img_h: int, img_w: int) -> bool:
    """
    Filter out non-handwriting regions: QR codes, diagrams, photos.
    Returns False if region looks like a QR code or large image block.
    """
    x1, y1, x2, y2 = bbox
    region_h = y2 - y1
    region_w = x2 - x1
    area_frac = (region_h * region_w) / max(img_h * img_w, 1)
    aspect = region_h / max(region_w, 1)

    # Skip very square large regions — likely QR codes or images
    if area_frac > 0.08 and 0.6 < aspect < 1.7:
        # Check pixel variance: handwriting has moderate variance, QR codes are extreme
        import cv2
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        black_frac = np.mean(binary == 0)
        # QR codes have ~30-50% black pixels in a noisy high-frequency pattern
        # Handwriting has <20% black pixels on white background
        if black_frac > 0.25:
            return False

    return True


def detect_ink_regions(
    img_np: np.ndarray,
    existing_bboxes: List[List[float]],
    min_area_frac: float = 0.003,
    max_area_frac: float = 0.55,
    ink_threshold: int = 180,
    margin: int = 10,
) -> List[dict]:
    """
    Fallback detector: find handwritten regions by ink density.

    Finds connected dark-pixel blobs on a light background, merges nearby
    strokes into coherent text blocks, and excludes regions already covered
    by YOLOv8 detections or identified as diagrams/photos.

    Returns list of detection dicts (same format as YOLOv8 output).
    """
    import cv2
    img_h, img_w = img_np.shape[:2]
    total_pixels = img_h * img_w

    # Threshold to binary (dark ink on white background)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, ink_threshold, 255, cv2.THRESH_BINARY_INV)

    # Merge strokes within the same text line (horizontal), then nearby lines (vertical).
    # Keep horizontal kernel modest so wide headers don't fuse into slide-spanning blobs.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 4))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 18))
    merged = cv2.dilate(binary, h_kernel, iterations=2)
    merged = cv2.dilate(merged, v_kernel, iterations=2)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_dets = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area_frac = (bw * bh) / total_pixels

        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue

        # Skip horizontal rules / thin decorative lines
        aspect = bh / max(bw, 1)
        if aspect < 0.02:
            continue

        # Skip slide-spanning blobs (full-width header/footer merge artifacts)
        if bw > img_w * 0.85:
            continue

        # Add margin
        x1 = max(0, bx - margin)
        y1 = max(0, by - margin)
        x2 = min(img_w, bx + bw + margin)
        y2 = min(img_h, by + bh + margin)

        # Skip if mostly covered by an existing YOLOv8 detection
        skip = False
        for eb in existing_bboxes:
            ex1, ey1, ex2, ey2 = [int(v) for v in eb]
            inter_w = max(0, min(x2, ex2) - max(x1, ex1))
            inter_h = max(0, min(y2, ey2) - max(y1, ey1))
            inter_area = inter_w * inter_h
            region_area = (x2 - x1) * (y2 - y1)
            if region_area > 0 and inter_area / region_area > 0.6:
                skip = True
                break
        if skip:
            continue

        # Require minimum ink density — filters out blank / near-blank regions
        region_crop = binary[y1:y2, x1:x2]
        ink_density = np.mean(region_crop > 0)
        if ink_density < 0.005:
            continue

        # Filter out diagrams / photos using two checks:
        # 1. Grayscale diversity: handwriting on white paper is mostly pure white (>200)
        #    or pure black (<60). Diagrams/photos have many intermediate gray values.
        region_gray = gray[y1:y2, x1:x2]
        mid_gray_frac = np.mean((region_gray > 60) & (region_gray < 200))
        if mid_gray_frac > 0.10:
            logger.debug(f"  Ink detector: skipping likely diagram/photo at "
                         f"({x1},{y1},{x2},{y2}) mid_gray={mid_gray_frac:.2f}")
            continue

        # 2. Edge complexity: diagrams have many connected curved edges;
        #    handwriting text lines are mostly horizontal with sparse edges.
        edges = cv2.Canny(region_gray, 50, 150)
        edge_density = np.mean(edges > 0)
        raw_ink = np.mean(region_crop > 0)
        if edge_density > 0.12 and raw_ink > 0.18:
            logger.debug(f"  Ink detector: skipping likely diagram at ({x1},{y1},{x2},{y2}) "
                         f"edge={edge_density:.2f} ink={raw_ink:.2f}")
            continue

        new_dets.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': 0.5,
            'class': 'text',
            'source': 'ink_detector',
        })

    logger.debug(f"  Ink detector found {len(new_dets)} additional regions")
    return new_dets


def _looks_like_math(text: str, region: np.ndarray) -> bool:
    """
    Heuristic: does this OCR result (or region image) look like a math expression?
    Uses text character ratios and basic image structure.
    """
    if not text:
        return False
    alpha = sum(c.isalpha() for c in text)
    total = len(text)
    math_chars = sum(1 for c in text if c in '+-=/<>^_{}[]()\\|∫∑∏√·×÷')
    # Looks like math if more than 25% math chars, or very short and has digits
    if total > 0 and math_chars / total > 0.25:
        return True
    if total < 20 and sum(c.isdigit() for c in text) / max(total, 1) > 0.3:
        return True
    return False


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_pipeline(
    detector_path: str,
    meta_checkpoint: Optional[str],
    ocr_fallback: str,
    device: str,
):
    """Load detector + OCR + math OCR models."""
    from baseline.baseline_pipeline import YOLOv8Detector
    from models.math_ocr_tamer import TAMERMathOCR

    detector = YOLOv8Detector(weights=detector_path, device=device)

    # OCR: prefer meta-learned model, fall back to fine-tuned TrOCR
    ocr = _load_ocr(meta_checkpoint, ocr_fallback, device)

    # Math OCR: TAMER with pix2tex fallback
    math_ocr = TAMERMathOCR(device=device)

    return detector, ocr, math_ocr


def _load_ocr(meta_checkpoint: Optional[str], fallback_path: str, device: str):
    """Load MAMLOCRWrapper with checkpoint, or plain TrOCR as fallback."""
    from models.meta_learning_ocr import MAMLOCRWrapper
    import torch

    wrapper = MAMLOCRWrapper(base_model_path=fallback_path, device=device)

    if meta_checkpoint and Path(meta_checkpoint).exists():
        ckpt = torch.load(meta_checkpoint, map_location=device, weights_only=False)
        wrapper.meta_model.load_state_dict(ckpt['meta_model_state'])
        epoch = ckpt.get('epoch', '?')
        val_cer = ckpt.get('val_cer', float('nan'))
        logger.info(f"Loaded meta-checkpoint: epoch={epoch+1 if isinstance(epoch,int) else epoch}, "
                    f"val_CER={val_cer*100:.2f}%")
    else:
        logger.info(f"No meta-checkpoint found; using base OCR model from {fallback_path}")

    return wrapper


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_infer(
    image_path: Path,
    output_path: Path,
    detector,
    ocr,
    math_ocr,
    adapt_samples: Optional[List[dict]] = None,
    n_shot: int = 5,
    conf_threshold: float = 0.35,
    annotate_only: bool = False,
    force_tamer: bool = False,
) -> List[dict]:
    """
    Run full OCR pipeline on one slide image and write typeset output.

    Args:
        image_path:     Input slide image.
        output_path:    Where to save the result.
        detector:       YOLOv8Detector instance.
        ocr:            MAMLOCRWrapper instance.
        math_ocr:       TAMERMathOCR instance.
        adapt_samples:  Optional list of {image, text} dicts for professor adaptation.
        n_shot:         Adaptation steps if adapt_samples is provided.
        conf_threshold: Detection confidence cutoff.
        annotate_only:  If True, draw boxes instead of replacing content.

    Returns:
        List of detection result dicts.
    """
    from PIL import Image as PILImage
    from utils.image_utils import load_image, extract_region

    logger.info(f"Processing: {image_path}")
    t0 = time.time()

    img_np = load_image(image_path, mode='rgb')
    img_pil = PILImage.fromarray(img_np)

    # Step 1: Detect
    detector.conf_threshold = conf_threshold
    detections = detector.detect(img_np)
    logger.info(f"  Detected {len(detections)} regions "
                f"({sum(1 for d in detections if d['class']=='text')} text, "
                f"{sum(1 for d in detections if d['class']=='math')} math)")

    # Fallback: ink-based detection for handwriting YOLOv8 misses
    existing_bboxes = [d['bbox'] for d in detections]
    ink_dets = detect_ink_regions(img_np, existing_bboxes)
    if ink_dets:
        logger.info(f"  Ink detector added {len(ink_dets)} region(s) missed by YOLOv8")
        detections = detections + ink_dets

    if not detections:
        img_pil.save(str(output_path))
        logger.info(f"  No regions found. Saved copy -> {output_path}")
        return []

    # Step 2: Optional professor adaptation
    if adapt_samples:
        logger.info(f"  Adapting to professor with {len(adapt_samples[:n_shot])} samples...")
        ocr.adapt(adapt_samples[:n_shot], steps=n_shot)

    text_dets = [d for d in detections if d['class'] == 'text']
    math_dets = [d for d in detections if d['class'] == 'math']

    img_h, img_w = img_np.shape[:2]
    results = []
    use_adapted = adapt_samples is not None

    # Step 3: Text OCR — split multi-line regions into individual lines first
    for det in text_dets:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        region = img_np[y1:y2, x1:x2]

        if not is_likely_handwriting(region, det['bbox'], img_h, img_w):
            logger.debug(f"  Skipping non-handwriting region at ({x1},{y1},{x2},{y2})")
            continue

        if is_multiline(det['bbox'], img_h, img_w):
            line_spans = segment_lines(region)
            logger.debug(f"  Multi-line region split into {len(line_spans)} lines")
        else:
            line_spans = [(0, region.shape[0])]

        line_texts = []
        line_crops = [region[ly0:ly1, :] for ly0, ly1 in line_spans if ly1 > ly0]

        if line_crops:
            texts = ocr.predict(line_crops, use_adapted=use_adapted, postprocess_german=True)
            line_texts = texts

        full_text = ' '.join(line_texts)

        # If force_tamer is set, route through TAMER math OCR
        is_math = force_tamer
        if is_math and math_ocr is not None:
            try:
                latex = math_ocr.recognize(region)
                if latex:
                    logger.debug(f"  Math re-route: TrOCR='{full_text[:30]}' → TAMER='{latex[:40]}'")
                    results.append({**det, 'text': latex, 'lines': len(line_spans), 'type': 'math'})
                    if not annotate_only:
                        render_latex_in_box(img_pil, x1, y1, x2, y2, latex)
                    continue
            except Exception as e:
                logger.debug(f"  TAMER re-route failed: {e}")

        results.append({**det, 'text': full_text, 'lines': len(line_spans)})

        if not annotate_only:
            if len(line_spans) > 1 and line_texts:
                # White out the whole region first, then render each line
                from PIL import ImageDraw
                ImageDraw.Draw(img_pil).rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
                for (ly0, ly1), text in zip(line_spans, line_texts):
                    render_text_in_box(img_pil, x1, y1 + ly0, x2, y1 + ly1, text,
                                       bg_color=(255, 255, 255))
            else:
                render_text_in_box(img_pil, x1, y1, x2, y2, full_text)

    # Step 4: Math OCR (sequential)
    for det in math_dets:
        crop = extract_region(img_np, det['bbox'])
        try:
            latex = math_ocr.recognize(crop)
        except Exception as e:
            logger.debug(f"  Math OCR failed: {e}")
            latex = ''
        results.append({**det, 'text': latex})
        if not annotate_only:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            render_latex_in_box(img_pil, x1, y1, x2, y2, latex or '[math]')

    if annotate_only:
        _draw_annotations(img_pil, results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(str(output_path))
    elapsed = time.time() - t0
    logger.info(f"  Saved -> {output_path} ({elapsed:.1f}s)")

    return results


def _draw_annotations(img_pil, results: List[dict]):
    """Draw colored boxes + labels for annotate-only mode."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img_pil)
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r['bbox']]
        color = (0, 180, 0) if r['class'] == 'text' else (180, 0, 180)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{r['class']}: {r.get('text','')[:25]}"
        font = _get_font(12)
        draw.text((x1 + 2, max(0, y1 - 14)), label, fill=color, font=font)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Lecture slide OCR: replace handwriting with typeset text'
    )

    in_group = parser.add_mutually_exclusive_group(required=True)
    in_group.add_argument('--image', type=Path, help='Single input image')
    in_group.add_argument('--image-dir', type=Path, help='Directory of images to process')

    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument('--output', type=Path, default=None,
                           help='Output path for single image (default: <input>_out.<ext>)')
    out_group.add_argument('--output-dir', type=Path, default=None,
                           help='Output directory for batch mode')

    parser.add_argument('--detector-path', type=str,
                        default='runs/detect/runs/detect/baseline_v1_r2/weights/best.pt',
                        help='YOLOv8 checkpoint path')
    parser.add_argument('--meta-checkpoint', type=str,
                        default='checkpoint/maml_ocr/meta_checkpoint_best.pt',
                        help='Reptile/MAML meta-learned OCR checkpoint')
    parser.add_argument('--ocr-fallback', type=str,
                        default='checkpoint/trocr_german/best',
                        help='Fine-tuned TrOCR fallback path')
    parser.add_argument('--adapt-samples', type=Path, default=None,
                        help='JSON file with professor samples [{image, text}, ...] for adaptation')
    parser.add_argument('--n-shot', type=int, default=5,
                        help='Number of adaptation samples to use')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--annotate-only', action='store_true',
                        help='Draw detection boxes instead of replacing content')
    parser.add_argument('--force-tamer', action='store_true',
                        help='Route ALL regions through TAMER math OCR (for math-heavy slides)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-json', action='store_true',
                        help='Also save OCR results as JSON alongside output image')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load models once
    logger.info("Loading pipeline models...")
    detector, ocr, math_ocr = load_pipeline(
        detector_path=args.detector_path,
        meta_checkpoint=args.meta_checkpoint,
        ocr_fallback=args.ocr_fallback,
        device=args.device,
    )

    # Load optional adaptation samples
    adapt_samples = None
    if args.adapt_samples and args.adapt_samples.exists():
        with open(args.adapt_samples) as f:
            adapt_samples = json.load(f)
        logger.info(f"Loaded {len(adapt_samples)} professor adaptation samples")

    # Collect input/output pairs
    EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    if args.image:
        pairs = [(args.image, args.output or args.image.with_stem(args.image.stem + '_out'))]
    else:
        images = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in EXTS)
        out_dir = args.output_dir or args.image_dir / 'infer_out'
        pairs = [(img, out_dir / img.name) for img in images]
        logger.info(f"Found {len(pairs)} images in {args.image_dir}")

    if not pairs:
        logger.error("No images found to process.")
        sys.exit(1)

    # Process
    all_results = {}
    for img_path, out_path in pairs:
        try:
            results = run_infer(
                image_path=img_path,
                output_path=out_path,
                detector=detector,
                ocr=ocr,
                math_ocr=math_ocr,
                adapt_samples=adapt_samples,
                n_shot=args.n_shot,
                conf_threshold=args.conf,
                annotate_only=args.annotate_only,
                force_tamer=args.force_tamer,
            )
            all_results[str(img_path)] = results
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    if args.save_json:
        json_path = out_path.with_suffix('.json') if args.image else (
            (args.output_dir or args.image_dir / 'infer_out') / 'results.json'
        )
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Results saved -> {json_path}")

    logger.info(f"Done. Processed {len(all_results)} image(s).")


if __name__ == '__main__':
    main()
