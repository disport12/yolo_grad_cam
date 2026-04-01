# YOLO CAM Attribution 실험 프레임워크

YOLO 객체탐지 모델(`yolo11n.pt`)로 **사람(person) 검출**을 하고, 검출 결과를 바탕으로 **설명 가능성(attribution)** 실험을 수행한다.  
상위 폴더는 **EigenCAM**(기본) 파이프라인, 하위 **`grad_cam/`** 은 동일 실험을 **GradCAM**으로 재현하는 병렬 트랙이다.

---

## 목차

1. [실험 개요](#실험-개요)
2. [폴더 구조](#폴더-구조)
3. [실행 순서](#실행-순서)
4. [Grad-CAM 서브프로젝트 (`grad_cam/`) 분석](#grad-cam-서브프로젝트-grad_cam-분석)
5. [EigenCAM vs GradCAM 트랙 비교](#eigencam-vs-gradcam-트랙-비교)
6. [모델·지표·개입](#모델지표개입)
7. [GitHub에 올리기](#github에-올리기)

---

## 실험 개요

| 단계 | 스크립트 | 설명 |
|------|----------|------|
| Task 1 | `comparison_summary.md` | jacobgil vs kazuto Grad-CAM 구현 비교 요약 |
| Task 2 | `01_build_failure_bank.py` | 데이터셋에서 이미지 선별 → `test_images/`, `failure_bank/` |
| Task 3 | `02_yolo_cam.py` | 검출 + CAM 히트맵 (`cam_results/`) |
| Task 4 | `03_cam_metrics.py` | Inside Ratio, Focus Score, 개입 연동 지표 (`metrics_results/`) |
| Task 5 | `04_intervention.py` | 배경 blur / 객체 외부 마스킹 등 (`intervention_results/`) |

**`grad_cam/`** 안에는 위와 **동일 단계**를 GradCAM 전용으로 둔 스크립트(`config.py`, `01`…`04`)가 있다. `test_images/`만 상위와 공유하고, `failure_bank`·CAM·개입·지표 결과는 `grad_cam/` 아래에 따로 쌓인다.

---

## 폴더 구조

```
yolo_cam_attribution/
├── README.md                 # 본 문서
├── requirements.txt
├── comparison_summary.md
├── config.py
├── 01_build_failure_bank.py
├── 02_yolo_cam.py            # 기본 CAM: EigenCAM (--cam-method 로 변경 가능)
├── 03_cam_metrics.py
├── 04_intervention.py
├── test_images/              # 01 실행 시 생성 (gitignore 권장)
├── failure_bank/
├── cam_results/
├── metrics_results/
├── intervention_results/
└── grad_cam/                 # GradCAM 전용 트랙
    ├── README.md
    ├── config.py
    ├── 01_build_failure_bank.py
    ├── 02_yolo_cam.py       # GradCAM 고정
    ├── 03_cam_metrics.py
    ├── 04_intervention.py
    ├── failure_bank/
    ├── cam_results/
    ├── metrics_results/
    └── intervention_results/
```

---

## 실행 순서

### 상위 (EigenCAM 기본)

```bash
cd yolo_cam_attribution
pip install -r requirements.txt
# yolo11n.pt 를 이 폴더에 두거나 Ultralytics 가 받아오도록 둠

python 01_build_failure_bank.py
python 02_yolo_cam.py                    # 기본 eigencam
python 04_intervention.py
python 03_cam_metrics.py
```

### `grad_cam/` (GradCAM)

```bash
cd yolo_cam_attribution/grad_cam
pip install -r requirements.txt

python 01_build_failure_bank.py          # grad_cam/failure_bank/ + 상위 test_images
python 02_yolo_cam.py
python 04_intervention.py
python 03_cam_metrics.py
```

---

## Grad-CAM 서브프로젝트 (`grad_cam/`) 분석

### 왜 별도 폴더인가

Ultralytics YOLO의 **기본 `02_yolo_cam.py`는 EigenCAM을 기본값**으로 둔다. EigenCAM은 그래디언트 없이 활성화 맵에 PCA 성격의 투영을 쓰기 때문에, **NMS·검출 헤드가 미분하기 어려운 구조**에서도 비교적 안정적으로 동작한다.  
반면 **GradCAM**은 출력에 대해 **역전파**가 필요하므로, 그대로 전체 `DetectionModel.forward()`를 쓰면 다음 문제가 생긴다.

1. **Detect 헤드의 inference 전용 텐서**  
   최신 Ultralytics 검출 헤드는 `Inference tensors cannot be saved for backward` 류의 오류를 유발할 수 있다.
2. **`targets=None` 가정**  
   `pytorch-grad-cam`의 GradCAM은 분류 로짓 `[B, num_classes]` 에 가까운 출력을 가정할 때 `targets=None` 처리가 자연스럽다. 잘린 특징 맵 `[B, C, H, W]` 만 있으면 **스칼라 loss를 직접 정의**해야 한다.

### 이 레포에서 쓰는 해결책

| 요소 | 역할 |
|------|------|
| **`YOLOWrapper`** | `DetectionModel.model` 층을 **Ultralytics와 동일한 `m.f` 연결 규칙**으로 순전파하되, **마지막 Detect 모듈은 실행하지 않는다**. 그래프가 헤드 이전에서 끊겨 역전파가 가능해진다. |
| **`TotalSumTarget`** | 잘린 출력 텐서에 대해 `output.float().sum()` 으로 **스칼라 loss**를 만들어 GradCAM의 `backward` 경로를 통과시킨다. (특정 클래스 “사람” 로짓과 동일하지 않음 — **해석은 부분적**.) |
| **`input_tensor.requires_grad_(True)` + `torch.enable_grad()`** | 입력·파라미터 쪽 그래디언트 경로를 명시적으로 연다. |
| **타깃 레이어** | 상위와 동일하게 `model.model[-2]` (Detect 직전 블록). |

### 해석 시 주의

- GradCAM 히트맵은 **“잘린 특징 + 합(sum) loss”** 에 대한 민감도에 가깝다. **실제 person confidence나 박스 회귀에 대한 클래스별 Grad-CAM**과는 다르다.
- 검출 박스는 여전히 **`YOLO.predict()`** 로 얻으며, CAM은 **별도 640×640 입력 경로**로 계산한다(상위 스크립트와 동일한 설계). 레터박스와의 미세 불일치 가능성은 문서화만 해둔다.

---

## EigenCAM vs GradCAM 트랙 비교

| 항목 | 상위 `02_yolo_cam.py` | `grad_cam/02_yolo_cam.py` |
|------|------------------------|---------------------------|
| 기본 CAM | EigenCAM (`--cam-method`로 GradCAM 등 선택 가능) | GradCAM 고정 |
| Wrapper | tuple 첫 요소만 반환하는 얕은 래퍼 | Detect 생략 + `m.f` 그래프 순전파 |
| Loss / target | EigenCAM은 해당 제약 완화 | `TotalSumTarget` (출력 합) |
| 산출물 메타 | (선택) `cam_method` 없을 수 있음 | `metadata.json`에 `cam_method: gradcam` |
| Failure bank | `failure_bank/` | `grad_cam/failure_bank/` (동일 CSV 스키마) |

두 트랙 모두 **M1~M4 지표·개입 JSON 스키마**는 맞추어 두었고, `grad_cam`의 `comparison.json`에는 `cam_method` 필드가 추가되며 `03_cam_metrics.py`가 로드 시 제거한다.

---

## 모델·지표·개입

- **검출 모델:** `yolo11n.pt` (COCO pretrained, **class 0 = person** 만 사용)
- **CAM 라이브러리:** [`jacobgil/pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam)
- **Target layer:** `yolo.model.model[-2]`

### 정량 지표

| 지표 | 의미 |
|------|------|
| M1 Inside Ratio | bbox 내부 CAM 비율 |
| M2 Outside Ratio | 1 − M1 |
| M3 Confidence Drop | 개입 전후 max confidence 차이 (`04` 연동) |
| M4 Focus Score | `max(cam) / mean(cam)` |

### 개입 (`04`)

`bg_blur`, `obj_mask`, `obj_crop`, `bg_replace` 등 — 상위와 `grad_cam` 동일 옵션.

### Failure bank 카테고리

`easy_success`, `hard_success`, `missed_detection`, `small_object` 등 — 규칙은 `01_build_failure_bank.py` 내 `classify_image` 참고.

---

## GitHub에 올리기

이 디렉터리는 **`.gitignore`** 로 대용량·재현 가능 산출물(`test_images/`, `cam_results/`, `*.pt` 등)을 제외하도록 해 두었다. 저장소에는 **코드와 README**가 중심이 되고, 결과는 스크립트로 재생성한다.

```bash
cd yolo_cam_attribution
git init
git add .
git commit -m "Add YOLO CAM attribution framework with grad_cam GradCAM track"

# GitHub에서 빈 저장소 생성 후
git remote add origin https://github.com/<USER>/<REPO>.git
git branch -M main
git push -u origin main
```

GitHub CLI 사용 시:

```bash
gh repo create <REPO> --public --source=. --remote=origin --push
```

(로컬에서 `yolo_cam_attribution`을 루트로 쓰려면 위 `cd` 후 `git init` 하면 된다. 상위 `project` 전체를 올리려면 루트와 `.gitignore`를 따로 조정하면 된다.)

---

## 라이선스·인용

- YOLO: Ultralytics 라이선스 정책을 따른다.
- Grad-CAM: Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- EigenCAM: `pytorch-grad-cam` 문서 및 구현 참고.
