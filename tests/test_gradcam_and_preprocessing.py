import json

import numpy as np
import pytest
import torch
from PIL import Image

from ml.config import ARTIFACTS_DIR, MODELS_DIR, PROJECT_ROOT
from ml.explain import describe_attention, normalize_cam, overlay, upsample
from ml.gradcam import CamExport, gradcam
from ml.models import build_model
from ml.preprocessing import IMAGENET_MEAN, IMAGENET_STD, load_grayscale, pad_and_resize, preprocess, to_model_input


@pytest.mark.parametrize("arch", ["baseline_cnn", "efficientnet_b0"])
def test_exported_cam_equals_autograd_gradcam(arch):
    torch.manual_seed(0)
    model = build_model(arch, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    reference, logits = gradcam(model, x)
    k = int(logits.argmax())
    dim = model.fc.in_features
    with torch.no_grad():
        probs, cams, _ = CamExport(model, torch.zeros(4, dim), torch.eye(dim))(x)
    torch.testing.assert_close(probs, torch.softmax(logits, 1), rtol=1e-4, atol=1e-5)
    fast = torch.relu(cams[0, k])
    # Grad-CAM channel weights are w_ck / (h * w), so the maps differ only by that constant.
    torch.testing.assert_close(reference * reference.numel(), fast, rtol=1e-3, atol=1e-4)


def test_exported_ood_score_is_min_mahalanobis_distance_to_class_means():
    from ml import ood

    torch.manual_seed(0)
    model = build_model("baseline_cnn", pretrained=False).eval()
    dim = model.fc.in_features
    rng = np.random.default_rng(0)
    means = rng.normal(size=(4, dim)).astype(np.float32)
    a = rng.normal(size=(dim, dim)).astype(np.float32)
    precision = (a @ a.T / dim + np.eye(dim)).astype(np.float32)
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        _, _, score = CamExport(model, torch.from_numpy(means), torch.from_numpy(precision))(x)
        features = ood.penultimate_features(model, x).numpy()
    np.testing.assert_allclose(score.numpy(), ood.scores(features, means, precision), rtol=1e-4)


def test_pad_and_resize_keeps_aspect_ratio_on_black_background():
    wide = Image.new("L", (300, 100), 255)
    out = np.asarray(pad_and_resize(wide, 224))
    assert out.shape == (224, 224)
    assert out[:60].max() == 0 and out[-60:].max() == 0
    assert out[112, 20:204].min() > 240


def test_transparent_pixels_become_black(tmp_path):
    rgba = Image.new("RGBA", (80, 80), (255, 255, 255, 0))
    rgba.paste((255, 255, 255, 255), (20, 20, 60, 60))
    path = tmp_path / "scan.png"
    rgba.save(path)
    gray = np.asarray(load_grayscale(path))
    assert gray[0, 0] == 0 and gray[40, 40] == 255


def test_model_input_shape_and_normalisation():
    x = to_model_input(Image.new("L", (224, 224), 0))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    np.testing.assert_allclose(x[0, :, 0, 0], -IMAGENET_MEAN / IMAGENET_STD, rtol=1e-6)


def test_attention_description_locates_hot_spot():
    cam = np.zeros((7, 7))
    cam[0:2, 0:2] = 1.0
    heat = upsample(normalize_cam(cam), 224)
    info = describe_attention(heat)
    assert info["area"] == "upper-left part"
    assert 0 < info["coverage"] < 0.2
    assert overlay(Image.new("L", (224, 224), 128), heat).mode == "RGB"


def test_empty_cam_stays_zero():
    assert normalize_cam(np.full((7, 7), -1.0)).max() == 0


@pytest.mark.skipif(not (ARTIFACTS_DIR / "model.onnx").exists(), reason="run `python -m ml.export_onnx` first")
def test_onnx_artifact_matches_its_pytorch_checkpoint():
    import onnxruntime as ort
    from ml.dataset import read_split
    from ml.models import load_checkpoint

    meta = json.loads((ARTIFACTS_DIR / "model_meta.json").read_text())
    model, ckpt = load_checkpoint(MODELS_DIR / meta["arch"] / "best.pt")
    assert meta["version"].startswith(f"{meta['arch']}-epoch{ckpt['epoch']}-")
    session = ort.InferenceSession(str(ARTIFACTS_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    for rel in read_split("test")["path"].head(5):
        x = to_model_input(preprocess(PROJECT_ROOT / rel))
        probs, cams, _ = session.run(None, {"image": x})
        k = int(probs[0].argmax())
        with torch.no_grad():
            expected = torch.softmax(model(torch.from_numpy(x)), 1).numpy()
        np.testing.assert_allclose(probs, expected, atol=1e-4)
        reference, _ = gradcam(model, torch.from_numpy(x), k)
        np.testing.assert_allclose(reference.numpy() * reference.numel(), np.maximum(cams[0, k], 0),
                                   rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not (ARTIFACTS_DIR / "model.onnx").exists(), reason="run `python -m ml.export_onnx` first")
def test_ood_threshold_passes_real_scans_and_flags_non_mri_images():
    import onnxruntime as ort
    from ml import ood
    from ml.dataset import read_split

    meta = json.loads((ARTIFACTS_DIR / "model_meta.json").read_text())
    session = ort.InferenceSession(str(ARTIFACTS_DIR / "model.onnx"), providers=["CPUExecutionProvider"])

    def flagged(gray) -> bool:
        return float(session.run(None, {"image": to_model_input(gray)})[2][0]) > meta["ood_threshold"]

    real = read_split("test").sample(n=150, random_state=1)["path"]
    assert np.mean([flagged(preprocess(PROJECT_ROOT / rel)) for rel in real]) <= 0.05
    synthetic = ood.synthetic_ood_images(seed=123)
    for name in ("Random noise", "Portrait photo (and crops)", "Text documents"):
        assert all(flagged(pad_and_resize(im)) for im in synthetic[name][:5]), name
