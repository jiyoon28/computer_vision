"""결함 mask의 덩어리(connected component)를 찾아 원본 이미지에서 크롭한다.

Seg 기반 Clf(cascade) 실험용. 학습 때는 정답 mask, 추론 때는 YOLO mask에서 같은 함수로 크롭한다.
"""

import csv
import random
from pathlib import Path

# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = REPO_ROOT / "data" / "processed" / "segmentation" / "splits"
# 기존 ConvNeXt와 같은 순서(알파벳). Free = 결함이 아닌 크롭
CLASS_NAMES = ["Blowhole", "Break", "Crack", "Fray", "Free", "Uneven"]
NORMAL_CLASS = "Free"

# 아래 값은 train 정답 mask의 덩어리 분포(bbox 긴 변 중앙값 58px)를 보고 잡은 경험값
DILATION = 3        # 3px 이내 조각은 한 덩어리로 (1~3px 파편 13개가 사라짐)
MARGIN_RATIO = 0.25  # bbox 각 변을 25%씩 넓혀 주변 맥락 포함
MIN_SIDE = 64        # 작은 결함도 최소 64px 크롭


def find_boxes(mask):
    """mask(>0 = 결함) -> 덩어리별 bbox 목록 [(x0, y0, x1, y1), ...] (mask 좌표)."""
    binary = ndimage.binary_dilation(np.asarray(mask) > 0, iterations=DILATION)
    labeled, _ = ndimage.label(binary, structure=np.ones((3, 3)))
    return [(s[1].start, s[0].start, s[1].stop, s[0].stop) for s in ndimage.find_objects(labeled)]


def expand_box(box, mask_size, image_size):
    """mask 좌표 bbox -> 여유를 둔 원본 이미지 좌표 bbox.

    YOLO mask는 384x384로 찌그러진 좌표라 가로/세로 배율을 따로 적용해 원본으로 되돌린다.
    """
    (mw, mh), (iw, ih) = mask_size, image_size
    x0, y0, x1, y1 = box[0] * iw / mw, box[1] * ih / mh, box[2] * iw / mw, box[3] * ih / mh
    w = max((x1 - x0) * (1 + 2 * MARGIN_RATIO), MIN_SIDE)
    h = max((y1 - y0) * (1 + 2 * MARGIN_RATIO), MIN_SIDE)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = max(0, round(cx - w / 2)), max(0, round(cy - h / 2))
    return left, top, min(iw, round(cx + w / 2)), min(ih, round(cy + h / 2))


def read_split(split):
    with (SPLIT_DIR / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build_crops(split, seed=42):
    """정답 mask로 학습/검증용 크롭 목록을 만든다. [(image_path, box, label), ...]

    결함 이미지: 결함 덩어리마다 1개 (결함 종류 라벨)
    정상 이미지: 결함 크롭과 비슷한 크기로 무작위 1개 (Free 라벨)
      -> YOLO가 잘못 칠한 영역을 분류기가 '정상'으로 걸러낼 수 있게 하려는 목적
    mask가 빈 결함 이미지(train Uneven 3장)는 결함 위치를 몰라서 제외한다.
    """
    rng = random.Random(seed)
    rows = read_split(split)
    crops, defect_sizes = [], []
    for row in rows:
        if row["defect_type"] == NORMAL_CLASS:
            continue
        with Image.open(REPO_ROOT / row["mask_path"]) as src:
            mask = np.asarray(src.convert("L")) > 127
        size = (mask.shape[1], mask.shape[0])
        for box in find_boxes(mask):
            crop = expand_box(box, size, size)
            crops.append((row["image_path"], crop, CLASS_NAMES.index(row["defect_type"])))
            defect_sizes.append((crop[2] - crop[0], crop[3] - crop[1]))

    for row in rows:
        if row["defect_type"] != NORMAL_CLASS:
            continue
        with Image.open(REPO_ROOT / row["image_path"]) as src:
            iw, ih = src.size
        w, h = rng.choice(defect_sizes)
        w, h = min(w, iw), min(h, ih)
        left, top = rng.randint(0, iw - w), rng.randint(0, ih - h)
        crops.append((row["image_path"], (left, top, left + w, top + h), CLASS_NAMES.index(NORMAL_CLASS)))
    return crops


class CropDataset(Dataset):
    """(크롭 텐서, 라벨). ConvNeXt train.py의 evaluate()와 같은 형식."""

    def __init__(self, crops, transform):
        self.crops = crops
        self.transform = transform
        self.classes = CLASS_NAMES
        self.targets = [label for _, _, label in crops]

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index):
        image_path, box, label = self.crops[index]
        with Image.open(REPO_ROOT / image_path) as src:
            crop = src.convert("L").crop(box)
        return self.transform(crop), label
