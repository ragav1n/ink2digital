"""
Image utilities for the German Lecture Slide OCR project.
Handles image loading, preprocessing, augmentation, and region extraction.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_image(path: Union[str, Path], mode: str = "rgb") -> np.ndarray:
    """
    Load an image from disk.

    Args:
        path: Path to the image file.
        mode: One of 'rgb', 'gray', 'bgr'. Default 'rgb'.

    Returns:
        Numpy array (H, W, C) or (H, W) for gray.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if mode == "rgb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img  # bgr


def save_image(img: np.ndarray, path: Union[str, Path]) -> None:
    """Save an image (RGB or BGR) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 3 and img.shape[2] == 3:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        out = img
    cv2.imwrite(str(path), out)


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy RGB array."""
    return np.array(img.convert("RGB"))


def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert numpy RGB array to PIL Image."""
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def resize_image(
    img: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keep_aspect: bool = True,
) -> np.ndarray:
    """Resize image to target size."""
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    if keep_aspect:
        if width is not None and height is not None:
            scale = min(width / w, height / h)
            new_w, new_h = int(w * scale), int(h * scale)
        elif width is not None:
            scale = width / w
            new_w, new_h = width, int(h * scale)
        else:
            scale = height / h
            new_w, new_h = int(w * scale), height
    else:
        new_w = width if width is not None else w
        new_h = height if height is not None else h
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def pad_to_size(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    fill_value: int = 255,
) -> np.ndarray:
    """Pad image (on right/bottom) to reach target size."""
    h, w = img.shape[:2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if img.ndim == 3:
        return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=fill_value)
    return np.pad(img, ((0, pad_h), (0, pad_w)), constant_values=fill_value)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize uint8 [0,255] to float32 [0,1]."""
    return img.astype(np.float32) / 255.0


def binarize(img: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Binarize a grayscale or RGB image.

    Args:
        img: Input image.
        method: 'otsu' or 'adaptive'.

    Returns:
        Binary image (uint8, 0 or 255).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
        )
    else:
        raise ValueError(f"Unknown binarization method: {method}")
    return binary


def deskew(img: np.ndarray) -> np.ndarray:
    """Deskew a handwriting image using Hough line detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    return rotated


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------

def extract_region(
    img: np.ndarray,
    bbox: Union[List[float], Tuple[float, ...]],
    padding: int = 0,
    format: str = "xyxy",
) -> np.ndarray:
    """
    Extract a bounding-box region from an image.

    Args:
        img: Source image (H, W, C) or (H, W).
        bbox: Bounding box. Supports 'xyxy' (x1,y1,x2,y2) or 'xywh' (x,y,w,h).
        padding: Extra pixels to add around the region.
        format: 'xyxy' or 'xywh'.

    Returns:
        Cropped image region.
    """
    h, w = img.shape[:2]
    if format == "xywh":
        x1, y1, bw, bh = [int(v) for v in bbox]
        x2, y2 = x1 + bw, y1 + bh
    else:
        x1, y1, x2, y2 = [int(v) for v in bbox]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Augmentation (handwriting-specific)
# ---------------------------------------------------------------------------

def augment_handwriting(
    img: np.ndarray,
    prob: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply handwriting-specific augmentations for training robustness.

    Augmentations applied with probability `prob`:
    - Elastic distortion
    - Random rotation (±5°)
    - Brightness/contrast jitter
    - Gaussian noise
    - Ink smearing (morphological)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pil = numpy_to_pil(img)

    # Brightness jitter
    if random.random() < prob:
        factor = random.uniform(0.8, 1.2)
        pil = ImageEnhance.Brightness(pil).enhance(factor)

    # Contrast jitter
    if random.random() < prob:
        factor = random.uniform(0.8, 1.3)
        pil = ImageEnhance.Contrast(pil).enhance(factor)

    # Slight blur (simulate ink smear)
    if random.random() < prob * 0.5:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    img = pil_to_numpy(pil)

    # Small rotation
    if random.random() < prob:
        angle = random.uniform(-5, 5)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # Gaussian noise
    if random.random() < prob * 0.5:
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def augment_batch(images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
    """Apply augmentation to a list of images."""
    return [augment_handwriting(img, **kwargs) for img in images]


# ---------------------------------------------------------------------------
# Slide-specific utilities
# ---------------------------------------------------------------------------

def split_slide_into_tiles(
    img: np.ndarray,
    tile_h: int = 512,
    tile_w: int = 512,
    overlap: float = 0.1,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Split a large slide image into overlapping tiles for dense detection.

    Returns:
        List of (tile_image, (x1, y1, x2, y2)) tuples in original coordinates.
    """
    h, w = img.shape[:2]
    stride_h = int(tile_h * (1 - overlap))
    stride_w = int(tile_w * (1 - overlap))
    tiles = []
    for y in range(0, h, stride_h):
        for x in range(0, w, stride_w):
            x2 = min(x + tile_w, w)
            y2 = min(y + tile_h, h)
            tile = img[y:y2, x:x2]
            tiles.append((tile, (x, y, x2, y2)))
    return tiles


def letterbox(
    img: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    fill_value: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with padding to target size (letterbox style for YOLO).

    Returns:
        (letterboxed_image, scale_factor, (pad_w, pad_h))
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    out = np.full((target_h, target_w, img.shape[2] if img.ndim == 3 else 1),
                  fill_value, dtype=np.uint8)
    if img.ndim == 2:
        out = out.squeeze(-1)
    out[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return out, scale, (pad_w, pad_h)


if __name__ == "__main__":
    # Quick smoke test
    test_img = np.ones((100, 200, 3), dtype=np.uint8) * 200
    resized = resize_image(test_img, width=100)
    assert resized.shape == (50, 100, 3), f"Expected (50,100,3) got {resized.shape}"
    region = extract_region(test_img, [10, 10, 50, 50])
    assert region.shape == (40, 40, 3)
    lb, scale, pad = letterbox(test_img)
    assert lb.shape == (640, 640, 3)
    print("image_utils: all checks passed")
