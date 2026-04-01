# Grad-CAM 구현 비교 결과 정리

> jacobgil/pytorch-grad-cam vs kazuto1011/grad-cam-pytorch 심층 비교 요약

---

## 1. 비교 개요

| 항목 | jacobgil/pytorch-grad-cam | kazuto1011/grad-cam-pytorch |
|------|--------------------------|----------------------------|
| GitHub Stars | 12.7k | 803 |
| CAM 방법 수 | 15+ (GradCAM, GradCAM++, ScoreCAM, EigenCAM 등) | GradCAM only |
| Smoothing | aug_smooth, eigen_smooth 지원 | 없음 |
| Hook 방식 | `register_forward_hook` + `output.register_hook()` | `register_backward_hook` (deprecated) |
| Colormap | `cv2.COLORMAP_JET` | `matplotlib.jet_r` |
| Overlay | `0.5*heatmap + 0.5*img` → `/max` 재정규화 | `(cmap + raw) / 2` 단순 평균 |
| 코드 규모 | 패키지 (20+ 파일) | 단일 파일 (`grad_cam.py`) |
| 유지보수 | 활발 (253 commits) | 정지 (36 commits) |
| Python 호환 | 3.10+ 정상 | 3.10+에서 `collections.Sequence` ImportError |

**테스트 조건:** ResNet50, VGG19_bn, DenseNet121, MobileNetV2 — 동일 이미지(`cat_dog.png`), 동일 레이어에서 비교.

---

## 2. 핵심 발견: Grayscale CAM은 동일

12가지 분석 항목에 걸쳐 두 구현의 **grayscale CAM은 수치적으로 완전히 동일**했다.

```
MAE = 0.000000 | Correlation = 1.000000 | Cosine Similarity = 1.000000
```

| 분석 항목 | 차이 여부 | 비고 |
|-----------|:---------:|------|
| Grayscale CAM | 동일 | 4개 아키텍처 모두 MAE=0 |
| Layer 선택 (layer1~4) | 동일 | 모든 깊이에서 일치 |
| Normalization | 동일 | resize 순서 차이 MAE=0.0018 (미미) |
| Hook 메커니즘 | 동일 | gradient 차이 = 0.0 |
| Backward 신호 | 동일 | logit backward 방식 수학적 동치 |
| Guided Backprop | 동일 | 접근법 다르지만 결과 동일 |
| 아키텍처별 | 동일 | 4개 모델 모두 일치 |
| 배치 처리 | 동일 | per-image norm으로 동일 |

즉, **알고리즘 수준의 차이는 없다.** 두 라이브러리 모두 Grad-CAM 논문(Selvaraju et al., ICCV 2017)의 수식을 정확히 구현하고 있으며, 동일 조건에서 동일한 grayscale activation map을 산출한다.

---

## 3. 시각적 차이 원인 분석

시각적으로 heatmap이 다르게 보이는 원인은 **후처리(post-processing)** 단계에 있었다.

### 3.1 Layer 관점

| Layer | Feature Map 크기 | 특성 |
|-------|-----------------|------|
| layer1 | 56×56 | 엣지, 텍스처 (저수준) — 넓고 흐릿 |
| layer2 | 28×28 | 중간 수준 패턴 |
| layer3 | 14×14 | 객체 윤곽 시작 |
| layer4 | 7×7 | 의미론적 특징 (고수준) — 객체 집중 |

레이어 선택 자체는 두 구현 간 차이를 유발하지 않았다. 다만 최종 heatmap은 7×7 feature map을 입력 해상도(224×224)로 bilinear upsampling한 것이므로, 세밀한 경계는 보간의 산물이며 모델이 실제로 그 해상도로 "본" 것과는 다를 수 있다.

### 3.2 Normalization 관점

| 방식 | 설명 | 영향 |
|------|------|------|
| kazuto | `view(B,-1)` → flatten 후 min-max | 기준 |
| jacobgil | per-image `scale_cam_image()` | kazuto와 동일 결과 |
| resize→norm vs norm→resize | 순서 차이 | MAE=0.0018 (실질적 영향 없음) |
| Percentile clipping | 상하위 2% clip 후 정규화 | 극단값 영향 감소 |

정규화 방식의 차이는 본 실험에서 **실질적 영향이 미미**했다 (MAE=0.0018).

### 3.3 Smoothing 관점

| Smoothing | CAM Mean | Sharpness | 소요시간 |
|-----------|----------|-----------|---------|
| None | 0.2716 | 0.0478 | 1x |
| Aug Smooth (TTA 6회) | 0.2765 | 0.0471 | **6x** |
| Eigen Smooth (PCA) | **0.1334** | 0.0370 | 1.9x |
| Aug + Eigen | 0.1370 | 0.0363 | ~8x |

- **eigen_smooth**가 가장 큰 시각적 변화를 유발: PCA 첫 주성분만 남겨 노이즈를 상당히 줄임
- **aug_smooth**는 소폭 개선에 비해 실행시간 6배 증가로 비효율적
- kazuto 구현에는 smoothing 기능 자체가 없으므로, jacobgil에서 smoothing 옵션을 사용하면 결과가 달라짐

### 3.4 Colormap / Overlay 관점 — 차이의 주된 원인

| 원인 | 설명 | Pixel Difference |
|------|------|:----------------:|
| **Colormap 방향 반전** | `cv2.COLORMAP_JET` (빨강=높음) vs `matplotlib.jet_r` (파랑=높음) | **149.71** |
| **Overlay 방식** | weighted blend + `/max` 재정규화 vs 단순 `(cmap+raw)/2` | **68.57** |

동일한 grayscale CAM 배열이라도:
- jacobgil은 높은 활성화 영역을 **빨간색**으로 표시
- kazuto는 높은 활성화 영역을 **파란색**으로 표시

이는 알고리즘의 차이가 아니라 **시각화 렌즈의 차이**이다. XAI 결과를 보고할 때는 colormap, overlay 방법, normalization을 반드시 명시해야 한다.

---

## 4. YOLO 적용을 위한 구현 선택: jacobgil

본 실험 프레임워크에서는 **jacobgil/pytorch-grad-cam**을 기반으로 한다.

| 선택 근거 | 설명 |
|-----------|------|
| **Hook 안정성** | `output.register_hook()` 방식으로 PyTorch inplace 연산 이슈 회피. YOLO의 복잡한 neck 구조에서 더 안정적 |
| **Detection 모델 지원** | YOLO, Faster R-CNN 등 detection 모델용 target class/box 기반 CAM 계산 지원 |
| **다양한 CAM 방법** | GradCAM 외에 EigenCAM, LayerCAM 등 15+ 방법으로 다각도 분석 가능 |
| **Python 3.10+ 호환** | kazuto는 `collections.Sequence` deprecated로 최신 Python에서 오류 |
| **활발한 유지보수** | 253 commits, 지속 업데이트 중 |

kazuto 구현은 단일 파일에 핵심 로직이 명확하여 **교육 목적으로 우수**하나, YOLO detection 모델에 적용하기 위한 확장성과 안정성 면에서 jacobgil이 적합하다.

---

## 참고문헌

- Selvaraju, R.R. et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
- [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- [kazuto1011/grad-cam-pytorch](https://github.com/kazuto1011/grad-cam-pytorch)
- 상세 분석: `c:\Users\dispo\Desktop\grad_cam\README.md` (12가지 관점 전체 분석)
