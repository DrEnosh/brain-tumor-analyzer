"""Out-of-distribution check: Mahalanobis distance of penultimate features to the nearest class mean."""
import numpy as np
import torch
from matplotlib import cbook
from PIL import Image, ImageDraw
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader

from ml.dataset import MRIDataset
from ml.models import Classifier

VAL_PERCENTILE = 99.0


@torch.no_grad()
def penultimate_features(model: Classifier, x: torch.Tensor) -> torch.Tensor:
    return model.pool(model.features(x)).flatten(1)


def split_features(model: Classifier, split: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats, labels = [], []
    for x, y in DataLoader(MRIDataset(split), batch_size=64, num_workers=4):
        feats.append(penultimate_features(model, x.to(device)).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def fit(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(labels)
    means = np.stack([features[labels == k].mean(axis=0) for k in classes])
    precision = LedoitWolf().fit(features - means[labels]).precision_
    return means.astype(np.float32), precision.astype(np.float32)


def scores(features: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    diff = features[:, None, :] - means[None]
    return np.einsum("nkd,de,nke->nk", diff, precision, diff).min(axis=1)


def synthetic_ood_images(seed: int = 0) -> dict[str, list[Image.Image]]:
    """Non-MRI inputs a user might upload by mistake, for measuring detection."""
    rng = np.random.default_rng(seed)
    phantoms = []
    for _ in range(40):
        im = Image.new("L", (256, 256), 0)
        draw = ImageDraw.Draw(im)
        rx, ry = rng.integers(60, 110, 2)
        draw.ellipse((128 - rx, 128 - ry, 128 + rx, 128 + ry), fill=int(rng.integers(60, 160)))
        for _ in range(rng.integers(0, 3)):
            cx, cy, r = rng.integers(70, 190), rng.integers(70, 190), rng.integers(8, 30)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=int(rng.integers(150, 255)))
        phantoms.append(im)
    ramp = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    with cbook.get_sample_data("grace_hopper.jpg") as fh:
        portrait = Image.open(fh).convert("L")
    w, h = portrait.size
    photos = [portrait, portrait.rotate(90, expand=True), portrait.transpose(Image.FLIP_LEFT_RIGHT)] + [
        portrait.crop((int(rng.integers(0, w // 3)), int(rng.integers(0, h // 3)),
                       int(rng.integers(2 * w // 3, w)), int(rng.integers(2 * h // 3, h)))) for _ in range(12)]
    documents = []
    for i in range(10):
        im = Image.new("L", (256, 256), 255 if i % 2 else 0)
        draw = ImageDraw.Draw(im)
        for row in range(8):
            draw.text((10, 10 + row * 30), f"Patient report line {row}", fill=0 if i % 2 else 255)
        documents.append(im)
    return {
        "Cartoon head phantoms": phantoms,
        "Random noise": [Image.fromarray(rng.integers(0, 255, (256, 256), dtype=np.uint8)) for _ in range(20)],
        "Gradients": [Image.fromarray(ramp if i % 2 else ramp.T.copy()) for i in range(10)],
        "Portrait photo (and crops)": photos,
        "Text documents": documents,
    }
