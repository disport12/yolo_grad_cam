# README example images

GradCAM 트랙 (`grad_cam/02_yolo_cam.py`) 및 `failure_bank/annotated/` 에서 복사한 대표 이미지.

## 포함된 파일

| 파일 | 카테고리 | 원본 이미지 | 설명 |
|------|----------|------------|------|
| `easy_success_detection.png` | easy_success | `image1219.jpg` | 검출 bbox (conf=0.95) |
| `easy_success_cam.png` | easy_success | `image1219.jpg` | GradCAM 오버레이 |
| `easy_success_eigen_cam.png` | easy_success | `image1219.jpg` | EigenCAM 오버레이 |
| `hard_success_detection.png` | hard_success | `image233.jpg` | 검출 bbox (conf=0.32) |
| `hard_success_cam.png` | hard_success | `image233.jpg` | GradCAM 오버레이 |
| `hard_success_eigen_cam.png` | hard_success | `image233.jpg` | EigenCAM 오버레이 |
| `hard_success2_detection.png` | hard_success | `image528.jpg` | 검출 bbox (conf=0.38) |
| `hard_success2_cam.png` | hard_success | `image528.jpg` | GradCAM 오버레이 |
| `hard_success3_detection.png` | hard_success | `image573.jpg` | 검출 bbox (conf=0.51, 중복 2개) |
| `hard_success3_cam.png` | hard_success | `image573.jpg` | GradCAM 오버레이 |
| `small_object_detection.png` | small_object | `image1024.jpeg` | 검출 bbox (3명, 작은 bbox 포함) |
| `small_object_cam.png` | small_object | `image1024.jpeg` | GradCAM 오버레이 |
| `small_object2_detection.png` | small_object | `image1121.jpg` | 검출 bbox (16명 군중) |
| `small_object2_cam.png` | small_object | `image1121.jpg` | GradCAM 오버레이 |
| `missed_detection_annotated.jpg` | missed_detection | `image751.jpg` | 검출 실패 (annotated, CAM 없음) |

## 재생성 방법

파이프라인을 재실행한 뒤 `grad_cam/cam_results/` 및 `failure_bank/annotated/` 에서 동일 stem의 파일을 복사하면 된다.
