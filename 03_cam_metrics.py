"""
Task 4: 정량 지표 정의 및 구현

02_yolo_cam.py의 결과(cam_results/)를 읽어
4가지 CAM attribution 지표를 계산하고
failure_bank.csv의 카테고리별로 비교 시각화한다.

지표:
  M1: bbox 내부 CAM 비율 (Inside Ratio)
  M2: bbox 외부 CAM 비율 (Outside Ratio)  = 1 - M1
  M3: 배경 변형 전후 confidence 변화 (Confidence Drop) — 04_intervention.py 결과 연동
  M4: CAM 집중도 (Focus Score) = max(cam) / mean(cam)

Usage:
    python 03_cam_metrics.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CAM_RESULTS_DIR,
    FAILURE_BANK_DIR,
    INTERVENTION_RESULTS_DIR,
    METRICS_RESULTS_DIR,
    ensure_dirs,
)

plt.rcParams["font.family"] = "DejaVu Sans"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAM 정량 지표 계산")
    return parser.parse_args()


def compute_inside_ratio(
    grayscale_cam: np.ndarray,
    boxes_scaled: list[list[int]],
) -> float:
    """M1: bbox 내부 CAM 합 / 전체 CAM 합. 겹치는 bbox는 union 처리."""
    total = grayscale_cam.sum()
    if total < 1e-8:
        return 0.0

    h, w = grayscale_cam.shape
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes_scaled:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = True

    inside = grayscale_cam[mask].sum()
    return float(inside / total)


def compute_focus_score(grayscale_cam: np.ndarray) -> float:
    """M4: max(cam) / mean(cam). 높을수록 집중적."""
    mean_val = grayscale_cam.mean()
    if mean_val < 1e-8:
        return 0.0
    return float(grayscale_cam.max() / mean_val)


def load_cam_result(stem_dir: Path) -> dict | None:
    """cam_results/{stem}/ 에서 metadata와 CAM 데이터를 로드."""
    meta_path = stem_dir / "metadata.json"
    cam_path = stem_dir / "cam_data.npz"

    if not meta_path.exists() or not cam_path.exists():
        return None

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    data = np.load(str(cam_path))
    grayscale_cam = data["grayscale_cam"]

    return {
        "metadata": metadata,
        "grayscale_cam": grayscale_cam,
    }


def load_failure_bank() -> dict[str, str]:
    """failure_bank.csv에서 image_name -> category 매핑 로드."""
    csv_path = FAILURE_BANK_DIR / "failure_bank.csv"
    mapping: dict[str, str] = {}
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row["image_name"]] = row["category"]
    return mapping


def load_intervention_results() -> dict[str, dict]:
    """intervention_results에서 conf_drop 데이터 로드."""
    results: dict[str, dict] = {}
    if not INTERVENTION_RESULTS_DIR.exists():
        return results

    for stem_dir in sorted(INTERVENTION_RESULTS_DIR.iterdir()):
        if not stem_dir.is_dir():
            continue
        comp_path = stem_dir / "comparison.json"
        if comp_path.exists():
            with open(comp_path, encoding="utf-8") as f:
                results[stem_dir.name] = json.load(f)

    return results


def compute_all_metrics() -> pd.DataFrame:
    """모든 CAM 결과에 대해 M1~M4 지표를 계산."""
    category_map = load_failure_bank()
    intervention_data = load_intervention_results()

    rows: list[dict] = []

    if not CAM_RESULTS_DIR.exists():
        print("[ERROR] cam_results/ 가 없습니다. 02_yolo_cam.py를 먼저 실행하세요.")
        return pd.DataFrame()

    for stem_dir in sorted(CAM_RESULTS_DIR.iterdir()):
        if not stem_dir.is_dir():
            continue

        result = load_cam_result(stem_dir)
        if result is None:
            continue

        meta = result["metadata"]
        cam = result["grayscale_cam"]
        image_name = meta["image_name"]

        boxes_scaled = [b["xyxy_scaled"] for b in meta["boxes"]]
        max_conf = max(b["conf"] for b in meta["boxes"])

        # M1 & M2
        inside_ratio = compute_inside_ratio(cam, boxes_scaled)
        outside_ratio = 1.0 - inside_ratio

        # M4
        focus_score = compute_focus_score(cam)

        # M3: conf_drop (from intervention, if available)
        conf_drop_blur = None
        conf_drop_mask = None
        stem = stem_dir.name
        if stem in intervention_data:
            iv = intervention_data[stem]
            if "bg_blur" in iv:
                conf_drop_blur = iv["bg_blur"].get("delta_conf")
            if "obj_mask" in iv:
                conf_drop_mask = iv["obj_mask"].get("delta_conf")

        category = category_map.get(image_name, "unknown")

        rows.append({
            "image_name": image_name,
            "category": category,
            "num_boxes": len(meta["boxes"]),
            "max_confidence": max_conf,
            "inside_ratio": round(inside_ratio, 4),
            "outside_ratio": round(outside_ratio, 4),
            "focus_score": round(focus_score, 4),
            "cam_mean": round(meta["cam_stats"]["mean"], 4),
            "cam_max": round(meta["cam_stats"]["max"], 4),
            "conf_drop_blur": round(conf_drop_blur, 4) if conf_drop_blur is not None else "",
            "conf_drop_mask": round(conf_drop_mask, 4) if conf_drop_mask is not None else "",
        })

    return pd.DataFrame(rows)


def plot_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    """카테고리별 지표 비교 boxplot 생성."""
    if df.empty:
        return

    categories_present = [c for c in df["category"].unique() if c != "unknown"]
    if not categories_present:
        categories_present = df["category"].unique().tolist()

    df_plot = df[df["category"].isin(categories_present)]

    metrics_to_plot = [
        ("inside_ratio", "M1: BBox Inside CAM Ratio", "Inside Ratio"),
        ("focus_score", "M4: CAM Focus Score (max/mean)", "Focus Score"),
    ]

    # conf_drop이 있으면 추가
    if "conf_drop_blur" in df.columns and df["conf_drop_blur"].replace("", np.nan).dropna().shape[0] > 0:
        metrics_to_plot.append(
            ("conf_drop_blur", "M3: Confidence Drop (BG Blur)", "Conf Drop")
        )

    n_plots = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, (col, title, ylabel) in zip(axes, metrics_to_plot):
        plot_data = df_plot[["category", col]].copy()
        if col in ("conf_drop_blur", "conf_drop_mask"):
            plot_data[col] = pd.to_numeric(plot_data[col], errors="coerce")
            plot_data = plot_data.dropna()

        if plot_data.empty:
            ax.set_title(f"{title}\n(no data)")
            continue

        cats = sorted(plot_data["category"].unique())
        data_by_cat = [
            plot_data[plot_data["category"] == c][col].values
            for c in cats
        ]

        bp = ax.boxplot(data_by_cat, labels=cats, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Category")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(str(out_dir / "metrics_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 개별 scatter: inside_ratio vs focus_score
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for cat in sorted(df_plot["category"].unique()):
        subset = df_plot[df_plot["category"] == cat]
        ax2.scatter(
            subset["inside_ratio"], subset["focus_score"],
            label=cat, alpha=0.7, s=60,
        )
    ax2.set_xlabel("Inside Ratio (M1)")
    ax2.set_ylabel("Focus Score (M4)")
    ax2.set_title("Inside Ratio vs Focus Score by Category")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(str(out_dir / "scatter_inside_vs_focus.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    print("[INFO] CAM 정량 지표 계산 중 ...")
    df = compute_all_metrics()

    if df.empty:
        print("[ERROR] 계산할 데이터가 없습니다.")
        return

    # CSV 저장
    csv_path = METRICS_RESULTS_DIR / "cam_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[INFO] 지표 CSV 저장: {csv_path}")
    print(f"[INFO] 총 {len(df)}개 이미지 처리")

    # 카테고리별 통계 출력
    print("\n[카테고리별 평균 지표]")
    print("-" * 80)
    numeric_cols = ["inside_ratio", "outside_ratio", "focus_score", "max_confidence"]
    for cat in sorted(df["category"].unique()):
        subset = df[df["category"] == cat]
        print(f"\n  {cat} (n={len(subset)}):")
        for col in numeric_cols:
            print(f"    {col:20s}: mean={subset[col].mean():.4f}, std={subset[col].std():.4f}")

    # 시각화
    plot_metrics(df, METRICS_RESULTS_DIR)
    print(f"\n[INFO] 시각화 저장: {METRICS_RESULTS_DIR}")


if __name__ == "__main__":
    main()
