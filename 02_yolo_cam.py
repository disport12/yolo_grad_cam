"""
Task 3: YOLO에 CAM 적용

yolov11n.pt로 사람 검출 후, 각 검출 결과에 대해
EigenCAM / GradCAM heatmap을 생성하고 시각화를 저장한다.

EigenCAM은 gradient-free 방법으로, YOLO처럼 출력이 non-differentiable한
detection 모델에 안정적으로 적용할 수 있다.

Usage:
    python 02_yolo_cam.py
    python 02_yolo_cam.py --cam-method eigencam
    python 02_yolo_cam.py --cam-method gradcam
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
    CAM_RESULTS_DIR,
    CONF_THRESHOLD,
    PERSON_CLASS_ID,
    TEST_IMAGES_DIR,
    ensure_dirs,
    get_image_paths,
    load_yolo_model,
)


class YOLOWrapper(nn.Module):
    """DetectionModel을 감싸서 pytorch-grad-cam이 처리할 수 있는
    단일 Tensor를 반환하도록 한다.
    DetectionModel.forward()는 tuple을 반환하므로 첫 번째 원소만 꺼낸다."""

    def __init__(self, detection_model):
        super().__init__()
        self.model = detection_model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO + CAM 시각화")
    parser.add_argument(
        "--cam-method", type=str, default="eigencam",
        choices=["eigencam", "gradcam", "gradcampp", "layercam"],
        help="CAM 방법 선택 (기본: eigencam)",
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD,
        help="YOLO confidence threshold",
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="처리할 최대 이미지 수 (0=전체)",
    )
    return parser.parse_args()


def get_cam_class(method: str):
    """CAM 클래스를 동적으로 로드."""
    from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, LayerCAM
    mapping = {
        "eigencam": EigenCAM,
        "gradcam": GradCAM,
        "gradcampp": GradCAMPlusPlus,
        "layercam": LayerCAM,
    }
    return mapping[method]


def renormalize_cam_in_bboxes(
    grayscale_cam: np.ndarray,
    boxes: list[list[int]],
) -> np.ndarray:
    """bbox 내부에서 CAM을 재정규화하고, 외부는 0으로 설정."""
    from pytorch_grad_cam.utils.image import scale_cam_image
    renorm = np.zeros_like(grayscale_cam, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        region = grayscale_cam[y1:y2, x1:x2]
        if region.size > 0:
            renorm[y1:y2, x1:x2] = scale_cam_image(region.copy())
    if renorm.max() > 0:
        renorm = scale_cam_image(renorm)
    return renorm


def draw_detection_image(
    img_rgb: np.ndarray,
    detections: list[dict],
) -> np.ndarray:
    """검출 결과를 이미지에 그린다."""
    vis = img_rgb.copy()
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        color = (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"person {d['conf']:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 8, 15)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return vis


def process_single_image(
    img_path: Path,
    yolo_model,
    cam_model,
    conf_threshold: float,
    out_dir: Path,
) -> dict | None:
    """단일 이미지에 대해 YOLO 검출 + CAM을 수행하고 결과를 저장."""
    from pytorch_grad_cam.utils.image import show_cam_on_image

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # --- YOLO detection ---
    results = yolo_model.predict(
        source=img_bgr,
        conf=conf_threshold,
        classes=[PERSON_CLASS_ID],
        verbose=False,
        imgsz=640,
    )

    detections: list[dict] = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            conf = float(boxes.conf[i].cpu())
            detections.append({"xyxy": xyxy, "conf": conf, "cls": "person"})

    if not detections:
        return None

    # --- CAM computation ---
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_float = np.float32(img_resized) / 255.0
    device = next(cam_model.activations_and_grads.model.parameters()).device
    input_tensor = transforms.ToTensor()(img_float).unsqueeze(0).to(device)

    grayscale_cam = cam_model(input_tensor=input_tensor, targets=None)[0, :, :]

    # bbox 좌표를 640x640 스케일로 변환
    sx, sy = 640 / w, 640 / h
    scaled_boxes = []
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        scaled_boxes.append([
            int(x1 * sx), int(y1 * sy),
            int(x2 * sx), int(y2 * sy),
        ])

    # 전체 CAM overlay
    cam_overlay_640 = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # bbox 내부 재정규화 CAM
    renorm_cam = renormalize_cam_in_bboxes(grayscale_cam, scaled_boxes)
    renorm_overlay_640 = show_cam_on_image(img_float, renorm_cam, use_rgb=True)

    # detection 시각화 (원본 스케일)
    detection_vis = draw_detection_image(img_rgb, detections)

    # --- 저장 ---
    stem_dir = out_dir / img_path.stem
    stem_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(stem_dir / "detection.png"),
        cv2.cvtColor(detection_vis, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(stem_dir / "cam_overlay.png"),
        cv2.cvtColor(cam_overlay_640, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(stem_dir / "cam_renorm.png"),
        cv2.cvtColor(renorm_overlay_640, cv2.COLOR_RGB2BGR),
    )

    # grayscale CAM 자체도 저장 (metrics 계산용, npz)
    np.savez_compressed(
        str(stem_dir / "cam_data.npz"),
        grayscale_cam=grayscale_cam,
        renorm_cam=renorm_cam,
        scale_x=sx,
        scale_y=sy,
    )

    cam_stats = {
        "mean": float(np.mean(grayscale_cam)),
        "std": float(np.std(grayscale_cam)),
        "max": float(np.max(grayscale_cam)),
    }

    metadata = {
        "image_name": img_path.name,
        "image_size": [w, h],
        "cam_size": [640, 640],
        "boxes": [
            {
                "xyxy": d["xyxy"],
                "xyxy_scaled": sb,
                "conf": round(d["conf"], 4),
                "cls": d["cls"],
            }
            for d, sb in zip(detections, scaled_boxes)
        ],
        "cam_stats": cam_stats,
    }

    with open(stem_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def main() -> None:
    args = parse_args()
    ensure_dirs()

    images = get_image_paths(TEST_IMAGES_DIR)
    if not images:
        print("[ERROR] test_images/ 에 이미지가 없습니다.")
        print("        먼저 01_build_failure_bank.py를 실행하세요.")
        return

    if args.max_images > 0:
        images = images[:args.max_images]

    print(f"[INFO] {len(images)}장에 대해 {args.cam_method.upper()} 적용")

    # --- YOLO 모델 로드 ---
    yolo = load_yolo_model()

    # 첫 번째 이미지로 dummy predict 수행하여 모델 device를 확정시킴
    # (ultralytics는 첫 predict에서 자동으로 CUDA/CPU 결정)
    dummy_img = cv2.imread(str(images[0]))
    yolo.predict(source=dummy_img, conf=args.conf, verbose=False, imgsz=640)

    # --- CAM 모델 설정 ---
    # DetectionModel을 Wrapper로 감싸서 단일 Tensor 출력으로 변환
    wrapped_model = YOLOWrapper(yolo.model)
    wrapped_model.eval()

    # target layer: DetectionModel 내부 Sequential의 두 번째 마지막 레이어
    target_layers = [yolo.model.model[-2]]

    CamClass = get_cam_class(args.cam_method)
    cam = CamClass(model=wrapped_model, target_layers=target_layers)

    processed = 0
    for i, img_path in enumerate(images, 1):
        meta = process_single_image(
            img_path, yolo, cam, args.conf, CAM_RESULTS_DIR,
        )
        if meta is not None:
            n_boxes = len(meta["boxes"])
            max_conf = max(b["conf"] for b in meta["boxes"])
            print(f"  [{i}/{len(images)}] {img_path.name}: "
                  f"{n_boxes} person(s), max_conf={max_conf:.3f}, "
                  f"cam_mean={meta['cam_stats']['mean']:.4f}")
            processed += 1
        else:
            print(f"  [{i}/{len(images)}] {img_path.name}: 검출 없음 (스킵)")

    print(f"\n[INFO] 완료: {processed}/{len(images)}장 처리됨")
    print(f"[INFO] 결과 저장 위치: {CAM_RESULTS_DIR}")


if __name__ == "__main__":
    main()
