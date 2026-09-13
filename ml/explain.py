"""Torch-free heat-map rendering and attention description, shared by evaluation and the web app."""
import numpy as np
from PIL import Image

_ANCHORS = np.array([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
_COLOURS = np.array([[0, 0, 128], [0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0], [128, 0, 0]], dtype=float)
_LUT = np.stack([np.interp(np.linspace(0, 1, 256), _ANCHORS, _COLOURS[:, c]) for c in range(3)], axis=1).astype(np.uint8)


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(np.asarray(cam, dtype=np.float32), 0)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def upsample(cam01: np.ndarray, size: int) -> np.ndarray:
    heat = Image.fromarray(cam01.astype(np.float32), mode="F").resize((size, size), Image.BICUBIC)
    return np.clip(np.asarray(heat), 0.0, 1.0)


def overlay(gray: Image.Image, heat: np.ndarray, alpha: float = 0.5) -> Image.Image:
    base = np.repeat(np.asarray(gray, dtype=np.float32)[..., None], 3, axis=2)
    colour = _LUT[(heat * 255).astype(np.uint8)].astype(np.float32)
    weight = (alpha * heat)[..., None]
    return Image.fromarray((base * (1 - weight) + colour * weight).astype(np.uint8), mode="RGB")


def describe_attention(heat: np.ndarray, hot: float = 0.5) -> dict:
    mask = heat >= hot
    weights = np.where(mask, heat, 0.0) ** 2
    h, w = heat.shape
    ys, xs = np.mgrid[0:h, 0:w]
    total = weights.sum()
    cy, cx = ((ys * weights).sum() / total / h, (xs * weights).sum() / total / w) if total > 0 else (0.5, 0.5)
    vertical = "upper" if cy < 0.4 else "lower" if cy > 0.6 else "middle"
    horizontal = "left" if cx < 0.4 else "right" if cx > 0.6 else "centre"
    if vertical == "middle" and horizontal == "centre":
        area = "central part"
    elif vertical == "middle":
        area = f"{horizontal} side"
    elif horizontal == "centre":
        area = f"{vertical} part"
    else:
        area = f"{vertical}-{horizontal} part"
    coverage = float(mask.mean())
    return {
        "area": area,
        "centroid": {"x": round(float(cx), 3), "y": round(float(cy), 3)},
        "coverage": round(coverage, 3),
        "text": f"The model's strongest attention is in the {area} of the image "
                f"(about {coverage * 100:.0f}% of the image area).",
    }
