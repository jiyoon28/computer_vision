import csv
import random
from pathlib import Path

# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from config import DATASET_DIR, NUM_CLASSES, SPLIT_DIR


def sample_name(image_path):
    # yolo26 prepare_data.py와 같은 파일명 규칙: MT_Crack/Imgs/exp1_num_3191.jpg -> MT_Crack_exp1_num_3191.png
    path = Path(image_path)
    return f"{path.parent.parent.name}_{path.stem}.png"


def list_samples(split, dataset_dir=DATASET_DIR):
    """split에 쓸 파일명 목록.

    train: 폴더 전체 (원본 548 + 합성 Crack 60). train CSV는 폴더를 만든 뒤 바뀌지 않음
    val/test: 현재 split CSV 기준. 폴더에는 있지만 CSV에서 빠진 이미지(excluded.csv)는 쓰지 않는다
    """
    image_dir = Path(dataset_dir) / "images" / split
    if split == "train":
        names = sorted(p.name for p in image_dir.glob("*.png"))
    else:
        with (SPLIT_DIR / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
            names = [sample_name(row["image_path"]) for row in csv.DictReader(stream)]
        missing = [n for n in names if not (image_dir / n).is_file()]
        if missing:
            raise FileNotFoundError(f"{split} CSV에는 있지만 {image_dir}에 없는 이미지: {missing[:3]}")
    if not names:
        raise ValueError(f"빈 split입니다: {image_dir}")
    return names


def load_pair(dataset_dir, split, name):
    root = Path(dataset_dir)
    with Image.open(root / "images" / split / name) as src:
        image = src.convert("RGB")  # grayscale이 R/G/B에 복제돼 저장돼 있음
    with Image.open(root / "masks" / split / name) as src:
        mask = np.asarray(src.convert("L"))  # 픽셀 값 = 클래스 번호 0~5
    if mask.max() >= NUM_CLASSES:
        raise ValueError(f"mask에 알 수 없는 클래스 번호: {name}")
    return image, mask


class SegDataset(Dataset):
    def __init__(self, split, train=False, dataset_dir=DATASET_DIR):
        self.split = split
        self.train = train
        self.dataset_dir = dataset_dir
        self.names = list_samples(split, dataset_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        image, mask = load_pair(self.dataset_dir, self.split, name)
        image = TF.to_tensor(image)  # [3, H, W], 0~1. ImageNet 정규화는 모델 안에서 함
        mask = torch.from_numpy(mask.astype(np.int64))

        # UNet / YOLO와 같은 상하좌우 뒤집기. 이미지와 mask를 같은 방향으로
        if self.train:
            if random.random() < 0.5:
                image, mask = image.flip(-1), mask.flip(-1)
            if random.random() < 0.5:
                image, mask = image.flip(-2), mask.flip(-2)

        return {"image": image, "mask": mask, "name": name}
