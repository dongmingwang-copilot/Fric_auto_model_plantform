from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_uint8_gray(path: Path) -> Image.Image:
    with Image.open(path) as img:
        arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    arr = arr.astype(np.float32)
    if arr.size == 0:
        raise ValueError(f"Empty image: {path}")
    if arr.max() <= 255 and arr.min() >= 0:
        out = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        lo, hi = np.percentile(arr, [0.5, 99.5])
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
        out = ((arr - lo) / max(hi - lo, 1e-6) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def tile_positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def save_preview_png(source: Path, destination: Path) -> tuple[int, int]:
    image = load_uint8_gray(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination)
    return image.size


def save_binary_mask(mask: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(destination)


def save_probability(prob: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, prob.astype(np.float32))


def save_overlay(image: Image.Image, mask: np.ndarray, destination: Path, alpha: float = 0.34) -> None:
    base = np.array(image.convert("RGB"), dtype=np.uint8)
    color = np.array([255, 55, 40], dtype=np.float32)
    out = base.astype(np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + color * alpha
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out.clip(0, 255).astype(np.uint8), mode="RGB").save(destination)


def decode_mask_data_url(data_url: str) -> Image.Image:
    marker = "base64,"
    payload = data_url.split(marker, 1)[1] if marker in data_url else data_url
    raw = base64.b64decode(payload)
    return Image.open(BytesIO(raw)).convert("RGBA")


def rgba_to_binary_mask(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGBA"))
    alpha = arr[..., 3] > 0
    colored = arr[..., :3].max(axis=2) > 0
    return alpha & colored


def refine_mask(mask: np.ndarray, close_kernel: int = 0) -> np.ndarray:
    if close_kernel <= 1:
        return mask
    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
