import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
import torch

from ml import ood
from ml.config import ARTIFACTS_DIR, CLASSES, DISPLAY_NAMES, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR
from ml.dataset import read_split
from ml.gradcam import CamExport, gradcam
from ml.models import load_checkpoint
from ml.preprocessing import pad_and_resize, preprocess, to_model_input

ONNX_PATH = ARTIFACTS_DIR / "model.onnx"
META_PATH = ARTIFACTS_DIR / "model_meta.json"


def selected_arch() -> str:
    selection = ARTIFACTS_DIR / "selected_model.json"
    if selection.exists():
        return json.loads(selection.read_text())["arch"]
    return "efficientnet_b0"


def fit_ood(arch: str) -> tuple[np.ndarray, np.ndarray, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(MODELS_DIR / arch / "best.pt", device=device)
    train_f, train_y = ood.split_features(model, "train", device)
    means, precision = ood.fit(train_f, train_y)
    val_scores = ood.scores(ood.split_features(model, "val", device)[0], means, precision)
    test_f, test_y = ood.split_features(model, "test", device)
    test_scores = ood.scores(test_f, means, precision)
    threshold = float(np.percentile(val_scores, ood.VAL_PERCENTILE))
    with torch.no_grad():
        test_pred = model.fc(torch.from_numpy(test_f).to(device)).argmax(1).cpu().numpy()
    test_flagged = test_scores > threshold

    synthetic = {}
    for name, images in ood.synthetic_ood_images().items():
        x = torch.from_numpy(np.concatenate([to_model_input(pad_and_resize(im)) for im in images])).to(device)
        with torch.no_grad():
            s = ood.scores(ood.penultimate_features(model, x).cpu().numpy(), means, precision)
            confidence = torch.softmax(model(x), 1).max(1).values.cpu().numpy()
        synthetic[name] = {"n": len(images), "detected": float(np.mean(s > threshold)),
                           "median_model_confidence": float(np.median(confidence))}
    report = {
        "method": "Mahalanobis distance of penultimate features to the nearest class mean (Ledoit-Wolf covariance)",
        "threshold": threshold,
        "threshold_rule": f"{ood.VAL_PERCENTILE:g}th percentile of validation-set scores",
        "val_flag_rate": float(np.mean(val_scores > threshold)),
        "test_flag_rate": float(np.mean(test_flagged)),
        "test_images": int(len(test_scores)),
        "test_flagged_images": int(test_flagged.sum()),
        "test_accuracy_flagged": float(np.mean(test_pred[test_flagged] == test_y[test_flagged])),
        "test_accuracy_not_flagged": float(np.mean(test_pred[~test_flagged] == test_y[~test_flagged])),
        "synthetic_detection": synthetic,
    }
    return means, precision, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the selected classifier (+ CAM head + OOD score) to ONNX.")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--check-images", type=int, default=25)
    args = parser.parse_args()
    arch = args.arch or selected_arch()

    means, precision, ood_report = fit_ood(arch)
    model, ckpt = load_checkpoint(MODELS_DIR / arch / "best.pt", device="cpu")
    wrapper = CamExport(model, torch.from_numpy(means), torch.from_numpy(precision)).eval()
    dummy = torch.zeros(1, 3, ckpt["img_size"], ckpt["img_size"])
    torch.onnx.export(wrapper, (dummy,), str(ONNX_PATH), input_names=["image"],
                      output_names=["probabilities", "cams", "ood_score"], opset_version=18, dynamo=True,
                      external_data=False)

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    frame = read_split("val").sample(n=args.check_images, random_state=0)
    max_prob_diff, min_cam_corr, max_ood_rel_diff = 0.0, 1.0, 0.0
    for rel in frame["path"]:
        x = to_model_input(preprocess(PROJECT_ROOT / rel))
        probs, cams, ood_score = session.run(None, {"image": x})
        with torch.no_grad():
            t_probs, _, _ = wrapper(torch.from_numpy(x))
            features = ood.penultimate_features(model, torch.from_numpy(x)).numpy()
        pred = int(probs[0].argmax())
        reference, _ = gradcam(model, torch.from_numpy(x), pred)
        onnx_cam = np.maximum(cams[0, pred], 0).ravel()
        ref_cam = reference.numpy().ravel()
        expected_ood = ood.scores(features, means, precision)[0]
        max_prob_diff = max(max_prob_diff, float(np.abs(probs - t_probs.numpy()).max()))
        max_ood_rel_diff = max(max_ood_rel_diff, abs(float(ood_score[0]) - expected_ood) / expected_ood)
        if onnx_cam.std() > 0 and ref_cam.std() > 0:
            min_cam_corr = min(min_cam_corr, float(np.corrcoef(onnx_cam, ref_cam)[0, 1]))
    print(f"ONNX vs PyTorch: max probability diff={max_prob_diff:.2e}, "
          f"min CAM correlation with autograd Grad-CAM={min_cam_corr:.6f}, max OOD score rel. diff={max_ood_rel_diff:.2e}")
    if max_prob_diff > 1e-3 or min_cam_corr < 0.999 or max_ood_rel_diff > 1e-3:
        raise SystemExit("ONNX export does not match the PyTorch model")

    digest = hashlib.sha256(ONNX_PATH.read_bytes()).hexdigest()
    meta = {
        "arch": arch,
        "version": f"{arch}-epoch{ckpt['epoch']}-{digest[:8]}",
        "classes": list(CLASSES),
        "display_names": DISPLAY_NAMES,
        "img_size": ckpt["img_size"],
        "mean": ckpt["mean"],
        "std": ckpt["std"],
        "val_macro_f1": ckpt["val_macro_f1"],
        "ood_threshold": ood_report["threshold"],
        "sha256": digest,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "ood.json").write_text(json.dumps({"model_version": meta["version"], **ood_report}, indent=2))
    print(json.dumps(ood_report, indent=2))


if __name__ == "__main__":
    main()
