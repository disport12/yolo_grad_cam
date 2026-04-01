# `grad_cam/` — GradCAM 전용 트랙

상위 [`../README.md`](../README.md)에 **방법론·한계·EigenCAM과의 비교**가 정리되어 있다. 이 파일은 실행 요약만 둔다.

## 요약

- **입력 이미지:** 상위 `../test_images/` 공유  
- **산출:** `failure_bank/`, `cam_results/`, `intervention_results/`, `metrics_results/` 는 **이 폴더 안**에만 생성  
- **CAM:** GradCAM 고정, Detect 헤드 생략 래퍼 + `TotalSumTarget` (상위 README 참고)

## 실행 (`grad_cam/` 에서)

```bash
pip install -r requirements.txt
python 01_build_failure_bank.py
python 02_yolo_cam.py
python 04_intervention.py
python 03_cam_metrics.py
```

상위 디렉터리에 `yolo11n.pt` 가 있으면 `config.load_yolo_model()` 이 자동으로 사용한다.
