"""기존 split CSV를 YOLO semantic용 images/masks + YAML로 변환한다."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

import config as cfg


def read_split(csv_path):
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty split: {csv_path}")
    return rows


def sample_name(row):
    path = Path(row["image_path"])
    return f"{path.parent.parent.name}_{path.stem}.png"


def convert_sample(row, image_size):
    """U-Net Dataset과 동일한 grayscale/resize/마스크 임계값을 사용한다."""
    defect_type = row["defect_type"]
    if defect_type not in {*cfg.DEFECT_TO_ID, "Free"}:
        raise ValueError(f"Unknown defect type: {defect_type}")
    is_defect = int(row["is_defect"])
    if is_defect != int(defect_type != "Free"):
        raise ValueError(f"Inconsistent label: {row['image_path']}")

    with Image.open(cfg.REPO_ROOT / row["image_path"]) as src:
        image = src.convert("L")
    if is_defect:
        if not row["mask_path"]:
            raise ValueError(f"Missing defect mask: {row['image_path']}")
        with Image.open(cfg.REPO_ROOT / row["mask_path"]) as src:
            mask = src.convert("L")
        if mask.size != image.size:
            raise ValueError(f"Image/mask size mismatch: {row['image_path']}")
    else:
        mask = Image.new("L", image.size, 0)

    size = (image_size, image_size)
    image = image.resize(size, Image.Resampling.BILINEAR)
    mask = mask.resize(size, Image.Resampling.NEAREST)
    target = np.zeros((image_size, image_size), dtype=np.uint8)
    if is_defect:
        target[np.asarray(mask) > 127] = cfg.DEFECT_TO_ID[defect_type]
    # 사전학습 모델의 3채널 입력: grayscale을 R/G/B에 동일하게 복제.
    return image.convert("RGB"), Image.fromarray(target)


def prepare_dataset(output=cfg.DATASET_DIR, image_size=cfg.IMAGE_SIZE):
    output = Path(output).resolve()
    if image_size < 32 or image_size % 32:
        raise ValueError("Image size must be a positive multiple of 32.")

    splits = {name: read_split(cfg.SPLIT_DIR / f"{name}.csv")
              for name in ("train", "val", "test")}
    # 잘못된 CSV로 동일 원본이 여러 split에 들어가는 것을 방지.
    seen = set()
    for split, rows in splits.items():
        names = set()
        for row in rows:
            path = (cfg.REPO_ROOT / row["image_path"]).resolve()
            if path in seen:
                raise ValueError(f"Duplicate image across/within splits: {path}")
            seen.add(path)
            name = sample_name(row)
            if name in names:
                raise ValueError(f"Duplicate output name in {split}: {name}")
            names.add(name)

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Dataset already exists: {output}\n"
            "Use the existing dataset.yaml, or pass --output with a new directory."
        )

    manifest = {"image_size": image_size, "class_names": cfg.CLASS_NAMES,
                "split_sha256": {}, "counts": {}}
    for split, rows in splits.items():
        image_dir = output / "images" / split
        mask_dir = output / "masks" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            image, target = convert_sample(row, image_size)
            name = sample_name(row)
            image.save(image_dir / name)
            target.save(mask_dir / name)
        manifest["counts"][split] = len(rows)
        manifest["split_sha256"][split] = hashlib.sha256(
            (cfg.SPLIT_DIR / f"{split}.csv").read_bytes()
        ).hexdigest()
        print(f"{split}: {len(rows)} images")

    data = {"path": output.as_posix(), "train": "images/train",
            "val": "images/val", "test": "images/test", "masks_dir": "masks",
            "names": dict(enumerate(cfg.CLASS_NAMES))}
    # 변환이 모두 성공한 뒤에만 학습용 YAML을 만든다.
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Dataset YAML: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=cfg.DATASET_DIR)
    parser.add_argument("--size", type=int, default=cfg.IMAGE_SIZE)
    args = parser.parse_args()
    prepare_dataset(args.output, args.size)


if __name__ == "__main__":
    main()
