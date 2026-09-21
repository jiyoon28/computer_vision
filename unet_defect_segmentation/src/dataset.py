import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from config import IMAGE_SIZE, DEFECT_TO_ID
from splits import read_csv


class MagneticTileDataset(Dataset):
    def __init__(self, csv_path, train=False, image_size=IMAGE_SIZE):
        # 기존 splits.py가 CSV의 상대경로를 절대경로로 복원
        self.df = read_csv(csv_path)
        self.train = train
        self.image_size = image_size

        if self.df.empty:
            raise ValueError(f"빈 데이터셋입니다: {csv_path}")

        # 오타나 예상하지 않은 결함 종류를 조기에 확인
        allowed_types = set(DEFECT_TO_ID) | {"Free"}
        unknown = set(self.df["defect_type"]) - allowed_types

        if unknown:
            raise ValueError(f"알 수 없는 결함 종류: {unknown}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        mask_path = row["mask_path"]
        defect_type = row["defect_type"]
        is_defect = int(row["is_defect"])

        # 영상은 grayscale 1채널로 읽기
        with Image.open(image_path) as src:
            image = src.convert("L")

        # 정상 여부와 결함 종류 정보가 일치하는지 확인
        if is_defect != int(defect_type != "Free"):
            raise ValueError(f"라벨 정보가 일치하지 않습니다: {image_path}")

        if is_defect:
            if not mask_path:
                raise ValueError(f"결함 마스크가 없습니다: {image_path}")

            with Image.open(mask_path) as src:
                mask = src.convert("L")

            if mask.size != image.size:
                raise ValueError(f"이미지와 마스크 크기가 다릅니다: {image_path}")
        else:
            # 정상 이미지는 모든 픽셀이 배경
            mask = Image.new("L", image.size, color=0)

        size = [self.image_size, self.image_size]

        # 이미지에는 bilinear 보간
        image = TF.resize(
            image,
            size,
            interpolation=InterpolationMode.BILINEAR,
        )

        # 마스크에는 nearest 보간: 클래스 경계를 섞지 않음
        mask = TF.resize(
            mask,
            size,
            interpolation=InterpolationMode.NEAREST,
        )

        # 학습 때만 증강
        # 이미지와 정답 마스크를 반드시 같은 방향으로 변환
        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # [1, H, W], float32
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.5], std=[0.5])

        # 원본 마스크: 배경 0 / 결함 255
        binary_mask = np.asarray(mask) > 127

        # 정답: 배경 0 / 해당 결함의 클래스 번호 1~5
        target = np.zeros(binary_mask.shape, dtype=np.int64)

        if is_defect:
            class_id = DEFECT_TO_ID[defect_type]
            target[binary_mask] = class_id

        # CrossEntropyLoss의 정답은 [H, W], torch.long
        # 기존 이진 코드와 달리 unsqueeze(0)를 하지 않음!
        target = torch.from_numpy(target)

        return {
            "image": image,
            "mask": target,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "defect_type": defect_type,
            "is_defect": is_defect,
        }