"""정답 마스크 없이 이미지 한 장 추론. 기본 출력은 256x256 비교 이미지."""

import argparse
from pathlib import Path

from PIL import Image

import config as cfg
from inference import load_dataset, load_model, predict_mask, save_visualization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=cfg.CHECKPOINT_PATH)
    parser.add_argument("--data", type=Path, default=cfg.DATA_YAML)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=cfg.OUTPUT_DIR / "single")
    args = parser.parse_args()
    _, manifest = load_dataset(args.data)
    size = manifest["image_size"]
    with Image.open(args.source) as src:
        # 학습 데이터와 같은 grayscale -> 정사각형 resize -> RGB 복제.
        image = src.convert("L").resize((size, size), Image.Resampling.BILINEAR).convert("RGB")
    prediction = predict_mask(load_model(args.weights), image, size, args.device)
    path = save_visualization(args.output, f"{args.source.stem}.png", image, prediction)
    print(f"Original (resized) | Prediction: {path}")


if __name__ == "__main__":
    main()
