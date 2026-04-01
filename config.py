"""
YOLO CAM Attribution 실험 프레임워크 — 공통 설정 및 유틸리티
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT.parent / "datasets" / "datasets_2classes"
DATASET_IMAGES = DATASET_ROOT / "images"

TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
FAILURE_BANK_DIR = PROJECT_ROOT / "failure_bank"
CAM_RESULTS_DIR = PROJECT_ROOT / "cam_results"
METRICS_RESULTS_DIR = PROJECT_ROOT / "metrics_results"
INTERVENTION_RESULTS_DIR = PROJECT_ROOT / "intervention_results"

YOLO_MODEL_NAME = "yolo11n.pt"
PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Failure bank categories
# ---------------------------------------------------------------------------
CATEGORIES = [
    "easy_success",
    "hard_success",
    "missed_detection",
    "false_positive",
    "complex_background",
    "illumination_change",
    "occlusion",
    "small_object",
]

EASY_CONF_THRESHOLD = 0.70
SMALL_OBJECT_AREA_RATIO = 0.02

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_yolo_model(model_name: str | None = None):
    """Load a YOLO model via ultralytics."""
    from ultralytics import YOLO
    return YOLO(model_name or YOLO_MODEL_NAME)


def get_image_paths(directory: Path) -> list[Path]:
    """Return sorted image paths from *directory*."""
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_dirs():
    """Create all output directories."""
    for d in (
        TEST_IMAGES_DIR,
        FAILURE_BANK_DIR,
        FAILURE_BANK_DIR / "annotated",
        CAM_RESULTS_DIR,
        METRICS_RESULTS_DIR,
        INTERVENTION_RESULTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
