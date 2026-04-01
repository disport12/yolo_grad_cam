"""
Grad-CAM 전용 하위 파이프라인 설정.

test_images / failure_bank 는 상위 yolo_cam_attribution 과 공유하고,
cam_results / intervention_results / metrics_results 만 grad_cam/ 아래에 둔다.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = PROJECT_ROOT.parent

DATASET_ROOT = PARENT_ROOT.parent / "datasets" / "datasets_2classes"
DATASET_IMAGES = DATASET_ROOT / "images"

TEST_IMAGES_DIR = PARENT_ROOT / "test_images"
FAILURE_BANK_DIR = PROJECT_ROOT / "failure_bank"

CAM_RESULTS_DIR = PROJECT_ROOT / "cam_results"
METRICS_RESULTS_DIR = PROJECT_ROOT / "metrics_results"
INTERVENTION_RESULTS_DIR = PROJECT_ROOT / "intervention_results"

YOLO_MODEL_NAME = "yolo11n.pt"
PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.25

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

CAM_METHOD = "gradcam"


def load_yolo_model(model_name: str | None = None):
    """Load YOLO; prefer weights next to 상위 yolo_cam_attribution."""
    from ultralytics import YOLO

    name = model_name or YOLO_MODEL_NAME
    p = PARENT_ROOT / name
    if p.is_file():
        return YOLO(str(p))
    return YOLO(name)


def get_image_paths(directory: Path) -> list[Path]:
    """Return sorted image paths from *directory*."""
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_dirs():
    for d in (
        FAILURE_BANK_DIR,
        FAILURE_BANK_DIR / "annotated",
        CAM_RESULTS_DIR,
        METRICS_RESULTS_DIR,
        INTERVENTION_RESULTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
