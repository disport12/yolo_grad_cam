"""
Task 5: 개입(Intervention) 실험

test_images/의 이미지에 대해 4가지 개입을 수행하고,
각 개입 전후의 detection score 및 CAM 변화를 비교한다.

개입 방법:
  A) 배경 blur      — bbox 외부에 Gaussian blur (kernel=51)
  B) 객체 외부 마스킹 — bbox 외부를 회색(128,128,128)으로 채움
  C) 객체만 crop     — bbox 영역만 검은 배경에 배치
  D) 배경 교체       — bbox 외부를 다른 이미지의 배경으로 교체

Usage:
    python 04_intervention.py
    python 04_intervention.py --methods bg_blur obj_mask
    python 04_intervention.py --methods bg_blur obj_mask obj_crop bg_replace
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

import torch.nn as nn

from config import (
    CAM_METHOD,
    CONF_THRESHOLD,
    INTERVENTION_RESULTS_DIR,
    PERSON_CLASS_ID,
    TEST_IMAGES_DIR,
    ensure_dirs,
    get_image_paths,
    load_yolo_model,
)


class YOLOWrapper(nn.Module):
    """Ultralytics DetectionModel은 층 간 연결이 m.f 그래프이므로 _predict_once와 동일하게
    순전파하되, 마지막 Detect 층은 실행하지 않는다 (GradCAM 역전파 호환)."""

    def __init__(self, detection_model):
        super().__init__()
        self.seq = detection_model.model
        self.save = list(detection_model.save)

    def forward(self, x):
        y = []
        last_i = len(self.seq) - 1
        for m in self.seq:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i == last_i - 1:
                return x[0] if isinstance(x, (tuple, list)) else x
        return x[0] if isinstance(x, (tuple, list)) else x

class TotalSumTarget:
    """Truncated YOLO 출력이 [B,C,H,W]일 때 GradCAM용 스칼라 loss."""

    def __call__(self, model_output):
        return model_output.float().sum()


ALL_METHODS = ["bg_blur", "obj_mask", "obj_crop", "bg_replace"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="개입 실험")
    parser.add_argument(
        "--methods", nargs="+", default=["bg_blur", "obj_mask"],
        choices=ALL_METHODS,
        help="수행할 개입 방법 (기본: bg_blur obj_mask)",
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD,
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="처리할 최대 이미지 수 (0=전체)",
    )
    parser.add_argument(
        "--blur-kernel", type=int, default=51,
        help="배경 blur 커널 크기",
    )
    return parser.parse_args()


# ─── Intervention functions ───────────────────────────────────────────────


def create_person_mask(
    img_shape: tuple[int, int],
    boxes: list[list[int]],
) -> np.ndarray:
    """person bbox 영역을 1, 나머지를 0으로 하는 마스크 생성."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = 1
    return mask


def intervene_bg_blur(
    img: np.ndarray,
    boxes: list[list[int]],
    kernel_size: int = 51,
) -> np.ndarray:
    """A) 배경에만 Gaussian blur 적용."""
    mask = create_person_mask(img.shape, boxes)
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    result = img.copy()
    result[mask == 0] = blurred[mask == 0]
    return result


def intervene_obj_mask(
    img: np.ndarray,
    boxes: list[list[int]],
    fill_value: int = 128,
) -> np.ndarray:
    """B) bbox 외부를 회색으로 채움."""
    mask = create_person_mask(img.shape, boxes)
    result = np.full_like(img, fill_value)
    result[mask == 1] = img[mask == 1]
    return result


def intervene_obj_crop(
    img: np.ndarray,
    boxes: list[list[int]],
) -> np.ndarray:
    """C) bbox 영역만 검은 배경에 원본 위치 유지."""
    mask = create_person_mask(img.shape, boxes)
    result = np.zeros_like(img)
    result[mask == 1] = img[mask == 1]
    return result


def intervene_bg_replace(
    img: np.ndarray,
    boxes: list[list[int]],
    bg_img: np.ndarray,
) -> np.ndarray:
    """D) bbox 외부를 다른 이미지의 배경으로 교체."""
    mask = create_person_mask(img.shape, boxes)
    h, w = img.shape[:2]
    bg_resized = cv2.resize(bg_img, (w, h))
    result = bg_resized.copy()
    result[mask == 1] = img[mask == 1]
    return result


# ─── Detection + CAM helper ──────────────────────────────────────────────


def run_detection(model, img_bgr: np.ndarray, conf: float) -> list[dict]:
    """YOLO person 검출 실행."""
    results = model.predict(
        source=img_bgr,
        conf=conf,
        classes=[PERSON_CLASS_ID],
        verbose=False,
        imgsz=640,
    )
    detections: list[dict] = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            conf_val = float(boxes.conf[i].cpu())
            detections.append({"xyxy": xyxy, "conf": conf_val})
    return detections


def compute_cam(cam_model, img_rgb: np.ndarray) -> np.ndarray:
    """CAM heatmap 계산 (640x640 스케일)."""
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_float = np.float32(img_resized) / 255.0
    device = next(cam_model.activations_and_grads.model.parameters()).device
    input_tensor = transforms.ToTensor()(img_float).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    with torch.enable_grad():
        grayscale_cam = cam_model(input_tensor=input_tensor, targets=[TotalSumTarget()])[0, :, :]
    return grayscale_cam


def compute_inside_ratio(
    cam: np.ndarray,
    boxes: list[list[int]],
    img_w: int,
    img_h: int,
) -> float:
    """CAM bbox 내부 비율 계산 (겹치는 bbox는 union 처리)."""
    total = cam.sum()
    if total < 1e-8:
        return 0.0

    h_cam, w_cam = cam.shape
    sx, sy = 640 / img_w, 640 / img_h
    mask = np.zeros((h_cam, w_cam), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        sx1 = max(0, int(x1 * sx))
        sy1 = max(0, int(y1 * sy))
        sx2 = min(w_cam, int(x2 * sx))
        sy2 = min(h_cam, int(y2 * sy))
        mask[sy1:sy2, sx1:sx2] = True

    inside = cam[mask].sum()
    return float(inside / total)


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    ensure_dirs()

    images = get_image_paths(TEST_IMAGES_DIR)
    if not images:
        print("[ERROR] test_images/에 이미지가 없습니다.")
        return

    if args.max_images > 0:
        images = images[:args.max_images]

    # 배경 교체용 이미지 (리스트에서 마지막 이미지를 배경으로 사용)
    bg_source = None
    if "bg_replace" in args.methods and len(images) > 1:
        bg_path = images[-1]
        bg_source = cv2.imread(str(bg_path))

    print(f"[INFO] {len(images)}장에 대해 개입 실험: {args.methods}")

    # 모델 로드
    yolo = load_yolo_model()

    # dummy predict로 device 확정
    dummy_img = cv2.imread(str(images[0]))
    yolo.predict(source=dummy_img, conf=args.conf, verbose=False, imgsz=640)

    from pytorch_grad_cam import GradCAM
    wrapped_model = YOLOWrapper(yolo.model)
    wrapped_model.eval()
    for p in wrapped_model.parameters():
        if p.dtype.is_floating_point:
            p.requires_grad_(True)
    target_layers = [yolo.model.model[-2]]
    cam_model = GradCAM(model=wrapped_model, target_layers=target_layers)

    from pytorch_grad_cam.utils.image import show_cam_on_image

    processed = 0
    for idx, img_path in enumerate(images, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        # 원본 검출
        orig_dets = run_detection(yolo, img_bgr, args.conf)
        if not orig_dets:
            print(f"  [{idx}/{len(images)}] {img_path.name}: 원본 검출 없음 (스킵)")
            continue

        orig_max_conf = max(d["conf"] for d in orig_dets)
        orig_boxes = [d["xyxy"] for d in orig_dets]

        # 원본 CAM
        orig_cam = compute_cam(cam_model, img_rgb)
        orig_inside = compute_inside_ratio(orig_cam, orig_boxes, w, h)

        stem_dir = INTERVENTION_RESULTS_DIR / img_path.stem
        stem_dir.mkdir(parents=True, exist_ok=True)

        # 원본 저장
        cv2.imwrite(str(stem_dir / "original.png"), img_bgr)

        comparison: dict = {}

        # --- 각 개입 방법 수행 ---

        for method in args.methods:
            if method == "bg_blur":
                perturbed_bgr = intervene_bg_blur(img_bgr, orig_boxes, args.blur_kernel)
            elif method == "obj_mask":
                perturbed_bgr = intervene_obj_mask(img_bgr, orig_boxes)
            elif method == "obj_crop":
                perturbed_bgr = intervene_obj_crop(img_bgr, orig_boxes)
            elif method == "bg_replace":
                if bg_source is None:
                    continue
                perturbed_bgr = intervene_bg_replace(img_bgr, orig_boxes, bg_source)
            else:
                continue

            # 변형 이미지 저장
            cv2.imwrite(str(stem_dir / f"{method}.png"), perturbed_bgr)

            # 변형 이미지에 대한 재검출
            pert_dets = run_detection(yolo, perturbed_bgr, args.conf)
            pert_max_conf = max((d["conf"] for d in pert_dets), default=0.0)

            # 변형 이미지에 대한 CAM
            pert_rgb = cv2.cvtColor(perturbed_bgr, cv2.COLOR_BGR2RGB)
            pert_cam = compute_cam(cam_model, pert_rgb)
            pert_inside = compute_inside_ratio(
                pert_cam, orig_boxes, w, h,
            )

            # CAM overlay 저장
            pert_resized = cv2.resize(pert_rgb, (640, 640))
            pert_float = np.float32(pert_resized) / 255.0
            cam_vis = show_cam_on_image(pert_float, pert_cam, use_rgb=True)
            cv2.imwrite(
                str(stem_dir / f"{method}_cam.png"),
                cv2.cvtColor(cam_vis, cv2.COLOR_RGB2BGR),
            )

            delta_conf = orig_max_conf - pert_max_conf
            delta_cam = orig_inside - pert_inside

            comparison[method] = {
                "original_conf": round(orig_max_conf, 4),
                "perturbed_conf": round(pert_max_conf, 4),
                "delta_conf": round(delta_conf, 4),
                "original_cam_inside": round(orig_inside, 4),
                "perturbed_cam_inside": round(pert_inside, 4),
                "delta_cam_inside": round(delta_cam, 4),
                "perturbed_num_detections": len(pert_dets),
            }

        # 비교 결과 저장 (cam_method 메타 + 03_cam_metrics 호환 키 유지)
        out_obj = {"cam_method": CAM_METHOD}
        out_obj.update(comparison)
        with open(stem_dir / "comparison.json", "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2, ensure_ascii=False)

        processed += 1
        methods_str = ", ".join(
            f"{m}: dconf={comparison[m]['delta_conf']:+.3f}"
            for m in args.methods if m in comparison
        )
        print(f"  [{idx}/{len(images)}] {img_path.name}: {methods_str}")

    print(f"\n[INFO] 완료: {processed}장 처리됨")
    print(f"[INFO] 결과 저장: {INTERVENTION_RESULTS_DIR}")

    # --- 간단한 요약 통계 ---
    if processed > 0:
        print("\n[개입 실험 요약]")
        for method in args.methods:
            conf_drops = []
            cam_deltas = []
            for stem_dir in INTERVENTION_RESULTS_DIR.iterdir():
                comp_path = stem_dir / "comparison.json"
                if not comp_path.exists():
                    continue
                with open(comp_path, encoding="utf-8") as f:
                    comp = json.load(f)
                if method in comp and isinstance(comp.get(method), dict):
                    conf_drops.append(comp[method]["delta_conf"])
                    cam_deltas.append(comp[method]["delta_cam_inside"])

            if conf_drops:
                print(f"\n  {method}:")
                print(f"    Confidence Drop  — mean={np.mean(conf_drops):+.4f}, "
                      f"std={np.std(conf_drops):.4f}, "
                      f"range=[{min(conf_drops):+.4f}, {max(conf_drops):+.4f}]")
                print(f"    CAM Inside Delta — mean={np.mean(cam_deltas):+.4f}, "
                      f"std={np.std(cam_deltas):.4f}")


if __name__ == "__main__":
    main()
