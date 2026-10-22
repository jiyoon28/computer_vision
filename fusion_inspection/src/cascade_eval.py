"""병렬 결합과 Seg 기반 Clf(cascade)를 같은 이미지에서 비교한다.

병렬:    전체 이미지 Clf + YOLO. Clf 불 -> NG, Clf 양 + Seg 불 -> 재검수, 둘 다 양 -> OK
cascade: YOLO가 찾은 결함 덩어리를 원본에서 크롭 -> 크롭 분류기.
         결함 크롭이 하나라도 있으면 NG, Seg가 못 찾았거나 크롭이 모두 정상이면 OK
"""

# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from crops import CLASS_NAMES, NORMAL_CLASS, expand_box, find_boxes, read_split
# four_cell_analysis가 ConvNeXt/YOLO 코드 경로를 sys.path에 등록한다
from four_cell_analysis import (DEFAULT_CONVNEXT, DEFAULT_YOLO, OUTPUT_DIR, REPO_ROOT,
                                load_convnext, yolo_image_size)
from inference import load_model, predict_mask  # noqa: E402  (YOLO)
from prepare_data import convert_sample  # noqa: E402  (YOLO)

DEFAULT_CROP = REPO_ROOT / "fusion_inspection" / "models" / "convnext_crop_224_seed42.pth"
NORMAL_ID = CLASS_NAMES.index(NORMAL_CLASS)


@torch.no_grad()
def classify(model, transform, images, device):
    x = torch.stack([transform(image) for image in images]).to(device)
    return model(x).softmax(dim=1).cpu().numpy()


def summarize(records, key):
    defects = [r for r in records if r["true"] != NORMAL_CLASS]
    normals = [r for r in records if r["true"] == NORMAL_CLASS]
    flagged = [r for r in defects if r[key] == "NG"]
    return {
        "missed_defects": sum(r[key] == "OK" for r in defects),
        "false_alarms": sum(r[key] == "NG" for r in normals),
        "review": sum(r[key] == "REVIEW" for r in records),
        "type_correct": f"{sum(r[key + '_type'] == r['true'] for r in flagged)}/{len(flagged)}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-weights", type=Path, default=DEFAULT_CROP)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_model, full_tf, names = load_convnext(DEFAULT_CONVNEXT, device)
    crop_model, crop_tf, crop_names = load_convnext(args.crop_weights, device)
    if names != CLASS_NAMES or crop_names != CLASS_NAMES:
        raise ValueError("Class order differs between checkpoints.")
    seg_model = load_model(DEFAULT_YOLO)
    seg_size = yolo_image_size()

    report = {}
    for split in ("val", "test"):
        records = []
        for row in read_split(split):
            with Image.open(REPO_ROOT / row["image_path"]) as src:
                image = src.convert("L")

            # 병렬: 전체 이미지 Clf + YOLO
            full = classify(full_model, full_tf, [image], device)[0]
            seg_image, _ = convert_sample(row, seg_size)
            mask = predict_mask(seg_model, seg_image, seg_size)
            seg_ng = bool((mask > 0).any())
            clf_type = CLASS_NAMES[int(full.argmax())]
            if clf_type != NORMAL_CLASS:
                parallel = "NG"
            else:
                parallel = "REVIEW" if seg_ng else "OK"

            # cascade: YOLO mask(384 좌표)의 덩어리 -> 원본 좌표로 크롭 -> 크롭 분류기
            boxes = [expand_box(b, (seg_size, seg_size), image.size) for b in find_boxes(mask)]
            cascade, cascade_type, crop_preds = "OK", NORMAL_CLASS, []
            if boxes:
                probs = classify(crop_model, crop_tf, [image.crop(b) for b in boxes], device)
                crop_preds = [CLASS_NAMES[int(p.argmax())] for p in probs]
                defect = [i for i, c in enumerate(crop_preds) if c != NORMAL_CLASS]
                if defect:
                    best = max(defect, key=lambda i: probs[i].max())
                    cascade, cascade_type = "NG", crop_preds[best]

            records.append({
                "image": f"{Path(row['image_path']).parent.parent.name}/{Path(row['image_path']).name}",
                "true": row["defect_type"],
                "parallel": parallel, "parallel_type": clf_type,
                "cascade": cascade, "cascade_type": cascade_type,
                "seg_boxes": len(boxes), "crop_preds": ",".join(crop_preds),
            })

        report[split] = {"parallel": summarize(records, "parallel"),
                         "cascade": summarize(records, "cascade")}
        print(f"\n===== {split} ({len(records)} images) =====")
        print(f"{'':16s}{'parallel':>10s}{'cascade':>10s}")
        for key in ("missed_defects", "false_alarms", "review", "type_correct"):
            print(f"{key:16s}{str(report[split]['parallel'][key]):>10s}{str(report[split]['cascade'][key]):>10s}")

        filtered = [r for r in records if r["seg_boxes"] and r["cascade"] == "OK"]
        print(f"Seg가 영역을 찾았지만 크롭 분류기가 모두 정상으로 판단: {len(filtered)}장")
        diff = [r for r in records if (r["parallel"], r["parallel_type"]) != (r["cascade"], r["cascade_type"])]
        print(f"두 방식 결과가 다른 이미지 {len(diff)}장:")
        for r in diff:
            print(f"  {r['image']}: 정답 {r['true']} | 병렬 {r['parallel']}({r['parallel_type']}) | "
                  f"cascade {r['cascade']}({r['cascade_type']}), 크롭 {r['seg_boxes']}개 [{r['crop_preds']}]")

        out_dir = OUTPUT_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "cascade_per_image.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    (OUTPUT_DIR / "cascade_compare.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved:", OUTPUT_DIR / "cascade_compare.json")


if __name__ == "__main__":
    main()
