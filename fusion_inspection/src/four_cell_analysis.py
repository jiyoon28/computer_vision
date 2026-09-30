"""ConvNeXt(분류)와 YOLO(segmentation)의 양/불 판정을 네 칸으로 나눠 실제 불량 비율을 본다.

칸: (분류 양, seg 양) / (분류 불, seg 불) / (분류 불, seg 양) / (분류 양, seg 불)
각 칸의 실제 불량 비율을 보고 "바로 불량 / 바로 양품 / 재검수" 규칙을 정한다.
두 모델 모두 data/processed/segmentation/splits 분할로 학습된 체크포인트를 써야 한다.
"""

# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
CONVNEXT_SRC = REPO_ROOT / "convnext_defect_classification" / "src"
YOLO_SRC = REPO_ROOT / "yolo26_defect_segmentation" / "src"
sys.path[:0] = [str(CONVNEXT_SRC), str(YOLO_SRC)]

from dataset import build_transforms  # noqa: E402  (ConvNeXt)
from model import create_model  # noqa: E402  (ConvNeXt)
from inference import load_model, predict_mask  # noqa: E402  (YOLO)
from prepare_data import convert_sample  # noqa: E402  (YOLO)

SPLIT_DIR = REPO_ROOT / "data" / "processed" / "segmentation" / "splits"
DEFAULT_CONVNEXT = (
    REPO_ROOT / "convnext_defect_classification" / "models"
    / "convnextv2_tiny_384_sqrt_seed42_segsplit.pth"
)
DEFAULT_YOLO = REPO_ROOT / "yolo26_defect_segmentation" / "models" / "best.pt"
OUTPUT_DIR = REPO_ROOT / "fusion_inspection" / "outputs"
NORMAL_CLASS = "Free"

# (분류 불량?, seg 불량?) 순서. 표 출력 순서도 이것을 따른다
CELLS = {
    (False, False): "clf 양 / seg 양",
    (True, True): "clf 불 / seg 불",
    (True, False): "clf 불 / seg 양",
    (False, True): "clf 양 / seg 불",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="규칙은 val로 정하고, test는 확인용으로만 쓴다")
    parser.add_argument("--convnext-weights", type=Path, default=DEFAULT_CONVNEXT)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--min-area", type=int, default=1,
                        help="seg 불량으로 볼 최소 결함 픽셀 수 (384x384 기준)")
    parser.add_argument("--ng-ratio", type=float, default=0.8,
                        help="실제 불량 비율이 이 이상인 칸은 바로 불량")
    parser.add_argument("--ok-ratio", type=float, default=0.2,
                        help="실제 불량 비율이 이 이하인 칸은 바로 양품. 그 사이는 재검수")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_convnext(weights, device):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model = create_model(
        num_classes=len(checkpoint["class_names"]),
        model_name=checkpoint["model_name"],
        pretrained=False,  # 가중치는 체크포인트에서 불러오므로 다운로드 불필요
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    _, eval_transform = build_transforms(checkpoint["image_size"])
    return model, eval_transform, checkpoint["class_names"]


def yolo_image_size():
    # best.pt가 학습된 해상도. config.py의 IMAGE_SIZE(256)와 달라서 메타데이터에서 읽는다
    metadata_path = DEFAULT_YOLO.parent / "best_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata["dataset"]["image_size"]


def decide(ratio, ng_ratio, ok_ratio):
    if ratio >= ng_ratio:
        return "불량"
    if ratio <= ok_ratio:
        return "양품"
    return "재검수"


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    with (SPLIT_DIR / f"{args.split}.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    cls_model, cls_transform, class_names = load_convnext(args.convnext_weights, device)
    normal_id = class_names.index(NORMAL_CLASS)
    seg_model = load_model(args.yolo_weights)
    seg_size = yolo_image_size()
    print(f"Split: {args.split} ({len(rows)} images), YOLO size: {seg_size}, device: {device}")

    records = []
    for index, row in enumerate(rows, start=1):
        # ConvNeXt: 학습 때와 같은 평가 전처리 (grayscale 3채널 + 384 resize + ImageNet 정규화)
        with Image.open(REPO_ROOT / row["image_path"]) as src:
            x = cls_transform(src.convert("L")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = cls_model(x).softmax(dim=1)[0].cpu().numpy()
        cls_pred = int(probs.argmax())

        # YOLO: 학습 데이터를 만들 때와 같은 변환 (grayscale -> resize -> RGB)
        image, _ = convert_sample(row, seg_size)
        mask = predict_mask(seg_model, image, seg_size, args.device)
        area = int((mask > 0).sum())

        records.append({
            "image_path": row["image_path"],
            "defect_type": row["defect_type"],
            "gt_defect": int(row["is_defect"]),
            "cls_pred": class_names[cls_pred],
            "cls_p_defect": round(float(1 - probs[normal_id]), 4),
            "cls_ng": int(cls_pred != normal_id),
            "seg_area": area,
            "seg_ng": int(area >= args.min_area),
        })
        if index % 20 == 0:
            print(f"Processed {index}/{len(rows)}")

    # 네 칸 집계
    summary = []
    print(f"\n{'칸':18s} {'장수':>4s} {'실제불량':>6s} {'불량비율':>6s}  제안")
    for key, label in CELLS.items():
        cell = [r for r in records if (bool(r["cls_ng"]), bool(r["seg_ng"])) == key]
        n_defect = sum(r["gt_defect"] for r in cell)
        ratio = n_defect / len(cell) if cell else float("nan")
        action = decide(ratio, args.ng_ratio, args.ok_ratio) if cell else "-"
        note = "  (표본 10장 미만)" if 0 < len(cell) < 10 else ""
        print(f"{label:18s} {len(cell):4d} {n_defect:6d} {ratio:8.2f}  {action}{note}")
        summary.append({"cell": label, "count": len(cell), "true_defect": n_defect,
                        "defect_ratio": None if not cell else ratio, "action": action,
                        "defect_types": sorted({r["defect_type"] for r in cell if r["gt_defect"]})})

    output_dir = OUTPUT_DIR / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    report = {"split": args.split, "convnext_weights": str(args.convnext_weights),
              "yolo_weights": str(args.yolo_weights), "min_area": args.min_area,
              "ng_ratio": args.ng_ratio, "ok_ratio": args.ok_ratio, "cells": summary}
    (output_dir / "four_cell.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
