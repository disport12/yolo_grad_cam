"""
Task 2: 실패 사례 은행(Failure Bank) 구축

datasets_2classes에서 50장을 선별하여 yolov11n.pt로 사람 검출을 수행하고,
검출 결과를 confidence / bbox 크기 기반으로 자동 분류한 뒤
failure_bank/failure_bank.csv 와 annotated 시각화를 생성한다.

Usage:
    python 01_build_failure_bank.py
    python 01_build_failure_bank.py --num-images 40
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from config import (
    CATEGORIES,
    CONF_THRESHOLD,
    DATASET_IMAGES,
    EASY_CONF_THRESHOLD,
    FAILURE_BANK_DIR,
    IMAGE_EXTENSIONS,
    PERSON_CLASS_ID,
    SMALL_OBJECT_AREA_RATIO,
    TEST_IMAGES_DIR,
    ensure_dirs,
    load_yolo_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="실패 사례 은행 구축")
    parser.add_argument(
        "--num-images", type=int, default=50,
        help="선별할 이미지 수 (기본 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="이미지 샘플링 시드",
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD,
        help="YOLO confidence threshold",
    )
    return parser.parse_args()


def collect_source_images(num_images: int, seed: int) -> list[Path]:
    """datasets_2classes의 val+test에서 num_images장을 무작위 선별."""
    candidates: list[Path] = []
    for split in ("val", "test", "train"):
        split_dir = DATASET_IMAGES / split
        if split_dir.exists():
            candidates.extend(
                p for p in split_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
    random.seed(seed)
    selected = random.sample(candidates, min(num_images, len(candidates)))
    return sorted(selected)


def copy_to_test_images(sources: list[Path]) -> list[Path]:
    """선별된 이미지를 test_images/에 복사하고 경로 목록 반환."""
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in sources:
        dst = TEST_IMAGES_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def classify_image(
    detections: list[dict],
    img_h: int,
    img_w: int,
) -> tuple[str, str]:
    """
    검출 결과에 따라 카테고리와 실패 원인 메모를 반환.
    Returns (category, failure_reason_memo)
    """
    img_area = img_h * img_w

    if not detections:
        return "missed_detection", "사람 검출 없음 (헬멧 데이터셋이므로 사람 존재 가능성 높음)"

    max_conf = max(d["conf"] for d in detections)
    min_area_ratio = min(d["area"] / img_area for d in detections)

    if min_area_ratio < SMALL_OBJECT_AREA_RATIO:
        return "small_object", f"최소 bbox 면적 비율 {min_area_ratio:.4f} < {SMALL_OBJECT_AREA_RATIO}"

    if max_conf >= EASY_CONF_THRESHOLD:
        return "easy_success", f"max_conf={max_conf:.3f} >= {EASY_CONF_THRESHOLD}"

    return "hard_success", f"max_conf={max_conf:.3f}, 낮은 confidence로 검출"


def draw_annotated(
    img_bgr: np.ndarray,
    detections: list[dict],
    category: str,
    image_name: str,
) -> np.ndarray:
    """Detection bbox와 카테고리 라벨을 이미지에 그린다."""
    vis = img_bgr.copy()
    color_map = {
        "easy_success": (0, 200, 0),
        "hard_success": (0, 200, 200),
        "missed_detection": (0, 0, 255),
        "false_positive": (255, 0, 255),
        "complex_background": (255, 165, 0),
        "illumination_change": (200, 200, 0),
        "occlusion": (128, 0, 128),
        "small_object": (255, 100, 100),
    }
    color = color_map.get(category, (255, 255, 255))

    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"person {d['conf']:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 8, 15)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    tag = f"[{category}] {image_name}"
    cv2.putText(vis, tag, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def main() -> None:
    args = parse_args()
    ensure_dirs()

    # 1) 이미지 선별 및 복사
    print(f"[INFO] datasets_2classes에서 {args.num_images}장 선별 중 ...")
    sources = collect_source_images(args.num_images, args.seed)
    images = copy_to_test_images(sources)
    print(f"[INFO] {len(images)}장을 test_images/에 복사 완료")

    # 2) YOLO 모델 로드
    model = load_yolo_model()
    print(f"[INFO] 모델 로드 완료: {model.model_name}")

    # 3) 각 이미지에 대해 검출 및 분류
    annotated_dir = FAILURE_BANK_DIR / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    csv_path = FAILURE_BANK_DIR / "failure_bank.csv"
    rows: list[dict] = []

    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] 이미지 로드 실패: {img_path}")
            continue

        h, w = img_bgr.shape[:2]

        results = model.predict(
            source=img_bgr,
            conf=args.conf,
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
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                detections.append({
                    "xyxy": xyxy,
                    "conf": conf,
                    "area": bw * bh,
                })

        category, memo = classify_image(detections, h, w)

        bbox_info = ";".join(
            f"{d['xyxy']},conf={d['conf']:.3f}" for d in detections
        ) if detections else ""

        rows.append({
            "image_name": img_path.name,
            "category": category,
            "num_detections": len(detections),
            "max_confidence": f"{max((d['conf'] for d in detections), default=0.0):.4f}",
            "failure_reason_memo": memo,
            "bbox_info": bbox_info,
        })

        vis = draw_annotated(img_bgr, detections, category, img_path.name)
        cv2.imwrite(str(annotated_dir / f"{img_path.stem}_annotated.jpg"), vis)

    # 4) CSV 저장
    fieldnames = [
        "image_name", "category", "num_detections",
        "max_confidence", "failure_reason_memo", "bbox_info",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] failure_bank.csv 저장 완료: {csv_path}")

    # 5) 카테고리별 통계
    from collections import Counter
    cat_counts = Counter(r["category"] for r in rows)
    print("\n[카테고리별 분포]")
    for cat in CATEGORIES:
        cnt = cat_counts.get(cat, 0)
        print(f"  {cat:25s}: {cnt}")
    print(f"  {'합계':25s}: {len(rows)}")


if __name__ == "__main__":
    main()
