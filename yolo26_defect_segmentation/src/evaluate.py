"""테스트 평가 및 원본 | 정답 | 예측 PNG 저장. 학습에는 test를 사용하지 않는다."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import config as cfg
from inference import load_dataset, load_model, predict_mask, save_visualization
from metrics import compute_metrics, update_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=cfg.DATA_YAML)
    parser.add_argument("--weights", type=Path, default=cfg.CHECKPOINT_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=cfg.OUTPUT_DIR / "test")
    args = parser.parse_args()
    root, manifest = load_dataset(args.data)
    paths = sorted((root / "images/test").glob("*.png"))
    if not paths or len(paths) != manifest["counts"]["test"]:
        raise ValueError("Test images do not match the prepared dataset manifest.")
    model = load_model(args.weights)
    matrix = np.zeros((len(cfg.CLASS_NAMES), len(cfg.CLASS_NAMES)), dtype=np.int64)

    for index, path in enumerate(paths, start=1):
        with Image.open(path) as src:
            image = src.convert("RGB")
        with Image.open(root / "masks/test" / path.name) as src:
            target = np.asarray(src.convert("L"))
        prediction = predict_mask(model, image, manifest["image_size"], args.device)
        update_confusion_matrix(matrix, prediction, target)
        save_visualization(args.output, path.name, image, prediction, target)
        if index % 20 == 0:
            print(f"Evaluated {index}/{len(paths)}")

    metrics = compute_metrics(matrix)
    print(f"Test images: {len(paths)}")
    print(f"Foreground mIoU: {metrics['foreground_miou']:.4f}")
    print(f"Foreground Dice: {metrics['foreground_dice']:.4f}")
    print("\nClass          IoU      Dice")
    for i, name in enumerate(cfg.CLASS_NAMES):
        print(f"{name:12s} {metrics['iou'][i]:.4f}   {metrics['dice'][i]:.4f}")
    # 정답과 예측 모두에 없는 클래스의 NaN은 JSON null로 저장.
    for key in ("iou", "dice"):
        metrics[key] = [v if np.isfinite(v) else None for v in metrics[key]]
    report = {"weights": str(args.weights.resolve()), "test_images": len(paths),
              "class_names": cfg.CLASS_NAMES, "dataset": manifest,
              "confusion_matrix": matrix.tolist(), **metrics}
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Original | GT | Prediction: {args.output / 'visualizations'}")


if __name__ == "__main__":
    main()
