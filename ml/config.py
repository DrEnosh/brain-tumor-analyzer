from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIRS = ("Training", "Testing")
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}

SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
# 256-bit dHash distance at or below which two images are treated as the same scan.
# 20 is the largest value that merged no images across classes on this dataset.
NEAR_DUPLICATE_THRESHOLD = 20

ARCHITECTURES = ("baseline_cnn", "efficientnet_b0", "resnet50")
