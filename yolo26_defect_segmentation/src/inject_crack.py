"""Crack 결함을 train의 Free 이미지에 붙여넣어 합성 학습 데이터를 추가한다.

Dynamic Label Injection(arXiv 2408.10031)의 아이디어를 오프라인으로 구현:
Crack 영역을 회전/뒤집기 후 Poisson blending 또는 cut-paste로 삽입.
원본 데이터셋은 그대로 두고 새 폴더에 복사본 + 합성 이미지를 만든다.
val/test에는 합성 이미지를 넣지 않는다.
"""

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
import yaml

import config as cfg
from inference import load_dataset

CRACK_ID = cfg.CLASS_NAMES.index("Crack")
BLEND_MARGIN = 5  # Poisson blending에 함께 가져갈 Crack 주변 배경 폭(px)


def load_pair(root, name):
    image = np.asarray(Image.open(root / "images/train" / name).convert("L"))
    mask = np.asarray(Image.open(root / "masks/train" / name))
    return image, mask


def crop_crack(image, mask, rng):
    """Crack 주변을 잘라내고 90도 회전/뒤집기를 랜덤 적용한다."""
    crack = (mask == CRACK_ID).astype(np.uint8)
    kernel = np.ones((2 * BLEND_MARGIN + 1,) * 2, np.uint8)
    region = cv2.dilate(crack, kernel)
    x, y, w, h = cv2.boundingRect(region)
    patch = image[y:y + h, x:x + w]
    crack, region = crack[y:y + h, x:x + w], region[y:y + h, x:x + w]
    k = int(rng.integers(4))
    patch, crack, region = (np.rot90(a, k) for a in (patch, crack, region))
    if rng.random() < 0.5:
        patch, crack, region = (np.fliplr(a) for a in (patch, crack, region))
    return (np.ascontiguousarray(a) for a in (patch, crack, region))


def paste(background, patch, crack, region, rng):
    """patch를 background의 랜덤 위치에 붙이고 (이미지, 라벨)을 반환한다."""
    size = background.shape[0]
    h, w = patch.shape
    # seamlessClone은 마스크가 이미지 경계에 닿으면 실패하므로 2px 여유를 둔다.
    if h > size - 4 or w > size - 4:
        return None
    y = int(rng.integers(2, size - h - 1))
    x = int(rng.integers(2, size - w - 1))

    label = np.zeros_like(background)
    label[y:y + h, x:x + w][crack > 0] = CRACK_ID
    if rng.random() < 0.5:
        # cut-paste: Crack 픽셀만 그대로 옮긴다.
        out = background.copy()
        out[y:y + h, x:x + w][crack > 0] = patch[crack > 0]
        return out, label, "cutpaste"

    # Poisson blending: 배경과 같은 크기의 캔버스에 놓고 붙여서 좌표를 1:1로 맞춘다.
    src = background.copy()
    src[y:y + h, x:x + w] = patch
    blend_mask = np.zeros_like(background)
    blend_mask[y:y + h, x:x + w] = region * 255
    bx, by, bw, bh = cv2.boundingRect(blend_mask)
    center = (bx + bw // 2, by + bh // 2)
    to_bgr = lambda a: cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    out = cv2.seamlessClone(to_bgr(src), to_bgr(background), to_bgr(blend_mask),
                            center, cv2.NORMAL_CLONE)
    return cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), label, "poisson"


def inject(source, output, count, seed):
    root, manifest = load_dataset(source)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    names = sorted(p.name for p in (root / "masks/train").glob("*.png"))
    crack_names, free_names = [], []
    for name in names:
        mask = np.asarray(Image.open(root / "masks/train" / name))
        if (mask == CRACK_ID).any():
            crack_names.append(name)
        # 마스크가 비어 있는 결함 이미지도 있으므로 파일명으로 Free를 확인한다.
        elif "_Free_" in name and not mask.any():
            free_names.append(name)
    if not crack_names or not free_names:
        raise ValueError("Need both Crack and Free images in the train split.")
    print(f"Crack sources: {len(crack_names)}, Free backgrounds: {len(free_names)}")

    shutil.copytree(root, output)
    rng = np.random.default_rng(seed)
    methods = {"cutpaste": 0, "poisson": 0}
    made = 0
    while made < count:
        patch, crack, region = crop_crack(*load_pair(root, rng.choice(crack_names)), rng)
        background, _ = load_pair(root, rng.choice(free_names))
        result = paste(background, patch, crack, region, rng)
        if result is None:
            continue
        image, label, method = result
        methods[method] += 1
        name = f"synth_crack_{made:04d}.png"
        # 원본과 같게 grayscale을 3채널로 복제해서 저장.
        Image.fromarray(image).convert("RGB").save(output / "images/train" / name)
        Image.fromarray(label).save(output / "masks/train" / name)
        made += 1

    manifest["counts"]["train"] += count
    manifest["crack_injection"] = {"source": str(root), "count": count,
                                   "seed": seed, "methods": methods}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                          encoding="utf-8")
    yaml_path = output / "dataset.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["path"] = output.as_posix()
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Added {count} synthetic Crack images {methods}")
    print(f"Dataset YAML: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="dataset.yaml of the dataset to copy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args()
    inject(args.source, args.output, args.count, args.seed)


if __name__ == "__main__":
    main()
