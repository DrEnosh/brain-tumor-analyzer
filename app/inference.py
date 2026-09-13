import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

from ml.explain import describe_attention, normalize_cam, overlay, upsample
from ml.preprocessing import IMAGENET_MEAN, IMAGENET_STD, colour_fraction, load_grayscale, pad_and_resize, to_model_input

DISPLAY_SIZE = 448
ALLOWED_FORMATS = {"JPEG", "PNG"}
COLOUR_WARNING_THRESHOLD = 0.05
LOW_CONFIDENCE_THRESHOLD = 0.60


class InvalidImage(ValueError):
    pass


@dataclass
class Analysis:
    label: str
    display_name: str
    confidence: float
    probabilities: dict[str, float]
    attention: dict
    colour_warning: bool
    atypical_input: bool
    ood_score: float
    inference_ms: float
    display_image: Image.Image
    overlay_image: Image.Image


class TumorClassifier:
    def __init__(self, model_path: Path, meta_path: Path):
        self.meta = json.loads(meta_path.read_text())
        if (not np.allclose(self.meta["mean"], IMAGENET_MEAN) or not np.allclose(self.meta["std"], IMAGENET_STD)
                or self.meta["img_size"] != 224):
            raise RuntimeError("model metadata does not match the preprocessing used by the app")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.classes: list[str] = self.meta["classes"]
        self.version: str = self.meta["version"]

    def analyze(self, data: bytes) -> Analysis:
        start = time.perf_counter()
        try:
            with Image.open(io.BytesIO(data)) as probe:
                if probe.format not in ALLOWED_FORMATS:
                    raise InvalidImage("Only JPEG and PNG images are supported.")
                probe.verify()
            gray = load_grayscale(io.BytesIO(data))
        except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
            raise InvalidImage("The file is not a readable image.") from exc
        if min(gray.size) < 64:
            raise InvalidImage("The image is too small (minimum 64 x 64 pixels).")

        probs, cams, ood_score = self.session.run(
            None, {"image": to_model_input(pad_and_resize(gray, self.meta["img_size"]))})
        probs = probs[0]
        k = int(probs.argmax())
        heat = upsample(normalize_cam(cams[0, k]), DISPLAY_SIZE)
        display = pad_and_resize(gray, DISPLAY_SIZE)
        return Analysis(
            label=self.classes[k],
            display_name=self.meta["display_names"][self.classes[k]],
            confidence=float(probs[k]),
            probabilities={c: float(p) for c, p in zip(self.classes, probs)},
            attention=describe_attention(heat),
            colour_warning=colour_fraction(io.BytesIO(data)) > COLOUR_WARNING_THRESHOLD,
            atypical_input=float(ood_score[0]) > self.meta["ood_threshold"],
            ood_score=float(ood_score[0]),
            inference_ms=(time.perf_counter() - start) * 1000,
            display_image=display,
            overlay_image=overlay(display, heat),
        )
