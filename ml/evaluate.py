import json
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ml.config import (  # noqa: E402
    ARCHITECTURES, ARTIFACTS_DIR, CLASSES, DISPLAY_NAMES, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR, SEED, SPLITS_DIR,
)
from ml.dataset import MRIDataset, read_split  # noqa: E402
from ml.explain import normalize_cam, overlay, upsample  # noqa: E402
from ml.gradcam import gradcam  # noqa: E402
from ml.models import load_checkpoint  # noqa: E402
from ml.preprocessing import preprocess, to_model_input  # noqa: E402
from ml.train import DEFAULTS, predict  # noqa: E402

DEFAULT_BASELINE_EPOCHS = DEFAULTS["baseline_cnn"]["epochs"]

FIG_DIR = REPORTS_DIR / "figures"
ARCH_NAMES = {"baseline_cnn": "Custom CNN (baseline, from scratch)",
              "efficientnet_b0": "EfficientNet-B0 (ImageNet transfer)",
              "resnet50": "ResNet50 (ImageNet transfer)"}
K = len(CLASSES)
# Validation macro-F1 gap (about 2 of 1,010 images) treated as a tie; ties go to the smaller model for CPU serving.
SELECTION_TOLERANCE = 0.002


def macro_f1_from(targets: np.ndarray, pred: np.ndarray) -> float:
    cm = np.bincount(targets * K + pred, minlength=K * K).reshape(K, K).astype(float)
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(K), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(K), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(K), where=(precision + recall) > 0)
    return float(f1.mean())


def grouped_bootstrap(targets, pred, groups, n: int = 2000, seed: int = SEED) -> dict:
    """95% CIs resampling duplicate groups (not single images), since images in a group are correlated."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = [np.nonzero(groups == g)[0] for g in unique]
    accs, f1s = [], []
    for _ in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        accs.append(float((pred[idx] == targets[idx]).mean()))
        f1s.append(macro_f1_from(targets[idx], pred[idx]))
    return {"accuracy_95ci": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            "macro_f1_95ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]}


def split_metrics(probs: np.ndarray, targets: np.ndarray, groups: np.ndarray | None) -> dict:
    pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(targets, pred, labels=list(range(K)), zero_division=0)
    auc = [float(roc_auc_score(targets == k, probs[:, k])) for k in range(K)]
    out = {
        "n": int(len(targets)),
        "accuracy": float((pred == targets).mean()),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "macro_roc_auc": float(np.mean(auc)),
        "per_class": {c: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]),
                          "roc_auc": auc[k], "support": int(s[k])} for k, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(targets, pred, labels=list(range(K))).tolist(),
    }
    if groups is not None:
        out.update(grouped_bootstrap(targets, pred, groups))
    return out


def evaluate_arch(arch: str, device: torch.device) -> dict | None:
    ckpt_path = MODELS_DIR / arch / "best.pt"
    if not ckpt_path.exists():
        return None
    model, ckpt = load_checkpoint(ckpt_path, device=device)
    result = {"arch": arch, "name": ARCH_NAMES[arch], "best_epoch": ckpt["epoch"],
              "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}
    for split in ("val", "test"):
        loader = DataLoader(MRIDataset(split), batch_size=64, num_workers=4)
        logits, targets = predict(model, loader, device, amp=False)
        probs = torch.softmax(logits, 1).numpy()
        groups = read_split(split)["group_id"].to_numpy() if split == "test" else None
        result[split] = split_metrics(probs, targets.numpy(), groups)
        if split == "test":
            result["_test_probs"] = probs
            result["_test_targets"] = targets.numpy()
    history_path = MODELS_DIR / arch / "history.json"
    result["_history"] = json.loads(history_path.read_text())["history"] if history_path.exists() else []
    return result


def plot_confusions(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.6))
    axes = np.atleast_1d(axes)
    labels = [DISPLAY_NAMES[c] for c in CLASSES]
    for ax, res in zip(axes, results):
        cm = np.array(res["test"]["confusion_matrix"])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else "black")
        ax.set_xticks(range(K), labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(K), labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(ARCH_NAMES[res["arch"]], fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_roc(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for k, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(res["_test_targets"] == k, res["_test_probs"][:, k])
        ax.plot(fpr, tpr, label=f"{DISPLAY_NAMES[c]} (AUC {res['test']['per_class'][c]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"One-vs-rest ROC, test set — {res['arch']}", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_training(results: list[dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for res in results:
        h = res["_history"]
        if not h:
            continue
        epochs = [r["epoch"] for r in h]
        ax1.plot(epochs, [r["val_macro_f1"] for r in h], marker="o", markersize=3, label=res["arch"])
        line, = ax2.plot(epochs, [r["train_loss"] for r in h], label=f"{res['arch']} train")
        ax2.plot(epochs, [r["val_loss"] for r in h], "--", color=line.get_color(), label=f"{res['arch']} val")
    ax1.set(xlabel="Epoch", ylabel="Validation macro-F1", title="Validation macro-F1")
    ax2.set(xlabel="Epoch", ylabel="Cross-entropy (label smoothing 0.1)", title="Loss")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def gradcam_gallery(arch: str, probs: np.ndarray, targets: np.ndarray, device: torch.device) -> dict:
    model, _ = load_checkpoint(MODELS_DIR / arch / "best.pt", device=device)
    paths = read_split("test")["path"].tolist()
    pred = probs.argmax(1)
    rng = np.random.default_rng(SEED)
    correct = [int(i) for k in range(K)
               for i in rng.choice(np.nonzero((targets == k) & (pred == k))[0], size=3, replace=False)]
    wrong = np.nonzero(pred != targets)[0]
    wrong = [int(i) for i in wrong[np.argsort(-probs[wrong].max(1))][:8]]

    def render(indices, filename, title):
        if not indices:
            return
        cols = 3 if len(indices) >= 3 else len(indices)
        rows = int(np.ceil(len(indices) / cols))
        fig, axes = plt.subplots(rows, cols * 2, figsize=(2.3 * cols * 2, 2.6 * rows))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for n, i in enumerate(indices):
            gray = preprocess(PROJECT_ROOT / paths[i])
            x = torch.from_numpy(to_model_input(gray)).to(device)
            cam, _ = gradcam(model, x, int(pred[i]))
            heat = upsample(normalize_cam(cam.cpu().numpy()), gray.size[0])
            r, c = divmod(n, cols)
            axes[r, 2 * c].imshow(gray, cmap="gray")
            axes[r, 2 * c + 1].imshow(overlay(gray, heat))
            axes[r, 2 * c].set_title(f"true: {DISPLAY_NAMES[CLASSES[targets[i]]]}", fontsize=7)
            axes[r, 2 * c + 1].set_title(f"pred: {DISPLAY_NAMES[CLASSES[pred[i]]]} ({probs[i].max():.0%})",
                                         fontsize=7, color="black" if pred[i] == targets[i] else "red")
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=150)
        plt.close(fig)

    render(correct, "gradcam_correct.png", f"Grad-CAM, correctly classified test images ({arch})")
    render(wrong, "gradcam_errors.png", f"Grad-CAM, most confident misclassifications ({arch})")
    return {"correct_examples": [paths[i] for i in correct], "error_examples": [paths[i] for i in wrong],
            "total_errors": int((pred != targets).sum())}


def fmt_ci(value: float, ci: list[float]) -> str:
    return f"{value:.2%} ({ci[0]:.2%}–{ci[1]:.2%})"


def write_report(results: list[dict], selected: dict, gallery: dict) -> None:
    split_summary = json.loads((SPLITS_DIR / "summary.json").read_text())
    lines = [
        "# Evaluation Report — Brain Tumor MRI Classification",
        "",
        f"Generated by `python -m ml.evaluate` on {date.today().isoformat()}. All test numbers come from the "
        "held-out test split, which was not used for training, early stopping or model selection.",
        "",
        "## 1. Data and split",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Raw images (Training + Testing folders) | {split_summary['raw_images']:,} |",
        f"| Byte-identical copies removed | {split_summary['exact_duplicate_copies_removed']:,} |",
        f"| Unique images used | {split_summary['unique_images']:,} |",
        f"| Near-duplicate groups (dHash ≤ {split_summary['near_duplicate_threshold_bits_of_256']}/256 bits) "
        f"| {split_summary['groups']:,} groups, largest {split_summary['largest_group']} images |",
        f"| Duplicate groups that crossed the original Training/Testing folders | "
        f"{split_summary['near_duplicate_groups_crossing_original_training_testing']:,} |",
        "",
        "Whole duplicate groups are assigned to one split, so no image (or re-encoded / contrast-shifted copy "
        "of it) appears in more than one split.",
        "",
        "| Split | Images | Share | " + " | ".join(DISPLAY_NAMES[c] for c in CLASSES) + " |",
        "|---|---|---|" + "---|" * K,
    ]
    for name, s in split_summary["splits"].items():
        lines.append(f"| {name} | {s['images']:,} | {s['fraction']:.1%} | "
                     + " | ".join(str(s["per_class"].get(c, 0)) for c in CLASSES) + " |")

    lines += [
        "",
        f"## 2. Model comparison (test set, n = {results[0]['test']['n']:,})",
        "",
        "95% confidence intervals come from a bootstrap that resamples duplicate groups.",
        "",
        "| Model | Params (M) | Best epoch | Val macro-F1 | Test accuracy (95% CI) | Test macro-F1 (95% CI) | Test macro ROC-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        t = r["test"]
        lines.append(f"| {r['name']} | {r['parameters_millions']} | {r['best_epoch']} | {r['val']['macro_f1']:.4f} | "
                     f"{fmt_ci(t['accuracy'], t['accuracy_95ci'])} | {fmt_ci(t['macro_f1'], t['macro_f1_95ci'])} | "
                     f"{t['macro_roc_auc']:.4f} |")
    lines += [
        "",
        f"**Deployed model: {selected['name']}** (validation macro-F1 {selected['val']['macro_f1']:.4f}). Selection used "
        f"validation data only: models within {SELECTION_TOLERANCE} macro-F1 of the best (about 2 validation images) "
        "count as tied, and the tie goes to the model with fewer parameters because the app serves predictions on a CPU. "
        "Test results were not used for the choice.",
        "",
        "## 3. Deployed model — per-class test results",
        "",
        "| Class | Precision | Recall | F1 | ROC-AUC | Support |",
        "|---|---|---|---|---|---|",
    ]
    for c in CLASSES:
        m = selected["test"]["per_class"][c]
        lines.append(f"| {DISPLAY_NAMES[c]} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                     f"{m['roc_auc']:.4f} | {m['support']} |")
    lines += [
        "",
        "![Confusion matrices](figures/confusion_matrices.png)",
        "",
        "![ROC curves](figures/roc_curves.png)",
        "",
        "![Training curves](figures/training_curves.png)",
        "",
        "## 4. Explainability (Grad-CAM)",
        "",
        f"The deployed model misclassified {gallery['total_errors']} of {selected['test']['n']:,} test images. "
        "Heat-maps below are computed on the final convolutional feature map for the predicted class.",
        "",
        "![Grad-CAM correct](figures/gradcam_correct.png)",
        "",
        "![Grad-CAM errors](figures/gradcam_errors.png)",
        "",
    ]
    review = REPORTS_DIR / "gradcam_review.md"
    if review.exists():
        lines += [review.read_text(encoding="utf-8").strip(), ""]

    ood_path = REPORTS_DIR / "ood.json"
    lines += ["## 5. Unusual-input detection", ""]
    if ood_path.exists():
        o = json.loads(ood_path.read_text())
        lines += [
            "A classifier always outputs one of its four classes, even for a photo or a document, often with high "
            f"confidence. The app therefore also computes an out-of-distribution score ({o['method']}). "
            "Scores above the threshold "
            f"({o['threshold_rule']}) show an \"Unusual input\" warning on the report.",
            "",
            "| Input | Images | Flagged as unusual | Median model confidence |",
            "|---|---|---|---|",
            f"| Real MRI, validation split (sets the threshold) | 1,010 | {o['val_flag_rate']:.1%} | — |",
            f"| Real MRI, test split | {o['test_images']:,} | {o['test_flag_rate']:.1%} | — |",
        ]
        for name, s in o["synthetic_detection"].items():
            lines.append(f"| {name} | {s['n']} | {s['detected']:.0%} | {s['median_model_confidence']:.0%} |")
        lines += [
            "",
            f"On real test scans the flag also marks less reliable predictions: accuracy is "
            f"{o['test_accuracy_flagged']:.1%} on the {o['test_flagged_images']} flagged scans versus "
            f"{o['test_accuracy_not_flagged']:.1%} on the rest, so the warning in the app covers both non-MRI uploads "
            "and unusual real scans. The check has not been tested on MRI of other body parts or other brain "
            "conditions, which it may not flag.",
            "",
        ]
    else:
        lines += ["Not computed yet — run `python -m ml.export_onnx`.", ""]

    latency_path = REPORTS_DIR / "latency.json"
    lines += ["## 6. Application response time", ""]
    if latency_path.exists():
        lat = json.loads(latency_path.read_text())
        lines += [
            f"Measured with `python -m ml.benchmark_app` against `{lat['base_url']}` on {lat['environment']}: "
            f"{lat['requests']} uploads of test-split images.",
            "",
            "| Measurement | Mean (ms) | Median (ms) | p95 (ms) |",
            "|---|---|---|---|",
        ]
        for key, label in (("upload_to_report_ms", "Upload → report page (end to end)"),
                           ("api_predict_ms", "`POST /api/predict` round trip"),
                           ("server_inference_ms", "Server-side preprocessing + model + heat-map")):
            s = lat[key]
            lines.append(f"| {label} | {s['mean']:.0f} | {s['median']:.0f} | {s['p95']:.0f} |")
        lines.append("")
    else:
        lines += ["Not measured yet — run `python -m ml.benchmark_app` against a running app.", ""]

    lines += [
        "## 7. Limitations",
        "",
        "- The dataset has no patient identifiers. Grouping by perceptual hash removes duplicates and many "
        "adjacent slices of the same scan, but images of the same patient that look different could still "
        "fall into different splits.",
        "- A single public dataset was used; there is no external validation on data from another hospital or scanner.",
        "- The \"no tumor\" images come from different sources than the tumour images: only about 1% of them are "
        "512 × 512 pixels, against 85–100% for each tumour class. Grad-CAM shows the model partly uses such image-level "
        "cues (framing, background), so accuracy on scans from a single hospital is likely lower than the test "
        "numbers above.",
        f"- The baseline CNN was still improving when it reached its {DEFAULT_BASELINE_EPOCHS}-epoch limit, so the "
        "gap between it and the transfer-learning models may be slightly overstated.",
        "- There are no tumour segmentation masks, so Grad-CAM localisation is reviewed qualitatively, not scored.",
        "- Research prototype for education only — not a medical device, and not for diagnosis.",
        "",
    ]
    (REPORTS_DIR / "EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [r for r in (evaluate_arch(a, device) for a in ARCHITECTURES) if r is not None]
    if not results:
        raise SystemExit("no trained checkpoints found under artifacts/models")
    best_val_f1 = max(r["val"]["macro_f1"] for r in results)
    tied = [r for r in results if best_val_f1 - r["val"]["macro_f1"] <= SELECTION_TOLERANCE]
    selected = min(tied, key=lambda r: r["parameters_millions"])

    plot_confusions(results)
    plot_roc(selected)
    plot_training(results)
    gallery = gradcam_gallery(selected["arch"], selected["_test_probs"], selected["_test_targets"], device)

    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(
        {"selected_arch": selected["arch"], "models": public, "gradcam": gallery}, indent=2))
    (ARTIFACTS_DIR / "selected_model.json").write_text(json.dumps(
        {"arch": selected["arch"], "val_macro_f1": selected["val"]["macro_f1"],
         "rule": f"highest validation macro-F1; within {SELECTION_TOLERANCE} counts as a tie, won by fewer parameters"},
        indent=2))
    write_report(results, selected, gallery)
    for r in public:
        t = r["test"]
        print(f"{r['arch']:16s} val_f1={r['val']['macro_f1']:.4f} test_acc={t['accuracy']:.4f} "
              f"test_f1={t['macro_f1']:.4f} auc={t['macro_roc_auc']:.4f}")
    print("selected:", selected["arch"])


if __name__ == "__main__":
    main()
