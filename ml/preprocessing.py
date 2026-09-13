"""Deterministic image preprocessing shared by training and the web app (no torch import)."""
from typing import BinaryIO, Union
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

Image.MAX_IMAGE_PIXELS = 40_000_000


def load_grayscale(source: Union[str, Path, BinaryIO]) -> Image.Image:
    with Image.open(source) as im:
        im.load()
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            background = Image.new("RGBA", im.size, (0, 0, 0, 255))
            im = Image.alpha_composite(background, im)
        return im.convert("L")


def pad_and_resize(gray: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Pad to a square on a black background (MRI background is black), then resize."""
    w, h = gray.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(gray, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BICUBIC)


def preprocess(source: Union[str, Path, BinaryIO], size: int = IMG_SIZE) -> Image.Image:
    return pad_and_resize(load_grayscale(source), size)


def to_model_input(gray: Image.Image) -> np.ndarray:
    """Grayscale PIL image -> float32 array of shape (1, 3, H, W), ImageNet-normalised."""
    x = np.asarray(gray, dtype=np.float32) / 255.0
    x = np.repeat(x[..., None], 3, axis=2)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def colour_fraction(source: Union[str, Path, BinaryIO]) -> float:
    """Mean channel disagreement in [0, 1]; MRI scans are ~0, colour photos are much higher."""
    with Image.open(source) as im:
        rgb = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    return float(spread.mean() / 255.0)
