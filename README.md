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
7. [실패 은행 기반 결과 분석](#실패-은행-기반-결과-분석)
8. [GitHub에 올리기](#github에-올리기)

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

## 실패 은행 기반 결과 분석

아래는 `01_build_failure_bank.py`를 **시드 42, 50장, `conf=0.25`** 조건으로 돌렸을 때 생성된 `failure_bank/failure_bank.csv`를 바탕으로 한 해석이다. (이미지를 다시 뽑으면 비율은 달라질 수 있다.)

### 분포 요약

| 카테고리 | 장수 | 비율(50장 기준) | 자동 분류 기준 요약 |
|----------|------|-----------------|---------------------|
| `easy_success` | 38 | 76% | 사람 검출됨, **max confidence ≥ 0.7**, bbox 최소 면적 비율 ≥ 2% |
| `hard_success` | 5 | 10% | 검출은 됐으나 **0.25 ≤ max conf < 0.7** |
| `small_object` | 6 | 12% | 검출은 됐으나 **어떤 bbox든** 이미지 대비 면적 비율 **&lt; 2%** 인 것이 하나라도 있음 |
| `missed_detection` | 1 | 2% | person 클래스 **검출 0건** (`conf` 임계값 미만) |

데이터는 **헬멧 2클래스 분류용 현장 이미지**에서 샘플링되었고, 라벨은 **COCO person 검출기(yolo11n)** 관점에서만 자동 판정된다.

### 쉬운 케이스(`easy_success`)가 쉬운 이유

- **신뢰도가 높다.** 대부분 max confidence가 **0.82~0.95** 구간에 몰려 있어, 특징이 “전형적인 사람” 형태로 잡힌 프레임이다.
- **객체가 프레임에서 충분히 크다.** 규칙상 최소 bbox 면적이 전체의 **2% 이상**이어야 `easy_success`에 들어가므로, **원거리·군집 속 아주 작은 실루엣**은 이 버킷에서 제외된다.
- **단일·소수 인물**이 큰 비중을 차지한다. 다인 영상(`image1181` 등 10명 검출)에서도 **최고 신뢰도 박스**가 0.7을 넘으면 전체 이미지는 `easy_success`로 분류된다(다만 일부 인원은 낮은 conf의 박스로만 잡힐 수 있음).

즉 “쉬움”은 **모델이 person에 대해 강하게 확신**하고, **너무 작지 않게** 잡혔다는 실험실 규칙상의 정의에 가깝다.

### 어렵지만 검출은 된 케이스(`hard_success`)

- **신뢰도만 낮을 뿐, 박스는 있다.** 예: `image233`(max 0.32), `image528`(0.38), `image231`(0.48), `image1363`(0.62), `image573`(0.51) 등.
- 해석 후보:
  - **자세·가림·조명**으로 특징이 약해 logits가 불확실한 경우
  - **박스가 이미지 대부분을 덮는 근접/과대 검출**처럼 맥락이 애매한 경우(`image231`, `image573`처럼 매우 큰 박스와 낮은 conf가 함께 나오는 패턴)
  - **헬멧·작업복** 등 COCO person 학습 분포와 어긋나 **얼굴·윤곽 신호가 약한** 현장 이미지

CAM·개입 실험에서는 이들을 **“검출은 되나 불확실”** 그룹으로 묶어, 배경 개입 시 confidence 변화가 큰지 등을 비교하기 좋다.

### `small_object`로 묶인 이유

- 규칙은 **“검출된 박스 중 하나라도”** 면적/이미지 &lt; **2%** 이면 전체를 `small_object`로 본다.
- 따라서 **원거리 인원**, **군중 속 작은 머리/상반신**, **부분만 보이는 사람**이 섞인 장면(`image1121` 16개 박스, `image1024` 등)이 여기로 간다.
- max confidence가 높아도(`image1024` 0.86 등) **최소 면적 비율** 때문에 `easy_success`가 아니다. “검출 품질”과 “난이도(크기)”를 **분리해 태깅**한 것이다.

### 실패에 가까운 케이스(`missed_detection`)

- 본 CSV에서는 **`image751.jpg` 1건**: person **0건** (임계값 0.25 기준).
- 라벨 상으로는 “실패”이지만, **진짜 사람이 없을 수도** 있고, **아주 작거나 가려져 임계값을 못 넘긴** 경우일 수도 있다. 헬멧 데이터셋 전제(사람 있을 가능성)는 메모에만 반영되어 있고, **정답 bbox 라벨과의 IoU 검증은 이 파이프라인에 없다**.
- CAM 파이프라인(`02`)은 검출이 없으면 **해당 이미지를 스킵**하므로, missed는 attribution 분석에서 **공백**이 된다.

### 성공·실패와 CAM 해석을 엮을 때

- **`easy_success`**: CAM이 bbox 안에 몰리면 “설명이 검출과 정합”에 가깝게 읽기 쉽다. 다만 EigenCAM/GradCAM 모두 **검출 loss와 1:1 대응은 아님**.
- **`hard_success` / `small_object`**: 낮은 conf·작은 박스는 **배경·주변 물체에 반응**할 여지가 커서, M1(Inside Ratio)·개입 실험(M3)과 함께 보는 것이 합리적이다.
- **`missed_detection`**: 설명 맵을 만들 수 없으므로, **한계 사례**로 남기고 데이터 보강·임계값·특화 학습을 논의하는 용도가 맞다.

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
