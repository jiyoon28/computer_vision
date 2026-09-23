import random

import numpy as np
import torch
from PIL import Image # 이미지와 mask 파일 열 때 사용
from torch.utils.data import Dataset # 사용자 정의 Dataset 만들기 위해 사용
import torchvision.transforms.functional as TF # resize, flip, tensor 변환, nomalization 수행
from torchvision.transforms import InterpolationMode # Resize 할 때 어떤 보간법 사용할지 지정

from config import IMAGE_SIZE, DEFECT_TO_ID
from splits import read_csv

# 한 장의 이미지와 그 이미지에 대응하는 결함 mask를 읽고
# resize/augmentation 을 한 다음 최종적으로 모델이 사용할 수 있는 tensor 형태로 반환

class MagneticTileDataset(Dataset): 
    def __init__(self, csv_path, train=False, image_size=IMAGE_SIZE):
        # 기존 splits.py가 CSV의 상대경로를 절대경로로 복원
        self.df = read_csv(csv_path)
        self.train = train  # train=True: 학습데이터이면 augmentation 사용
        self.image_size = image_size

        if self.df.empty:
            raise ValueError(f"빈 데이터셋입니다: {csv_path}")

        # 오타나 예상하지 않은 결함 종류를 조기에 확인
        allowed_types = set(DEFECT_TO_ID) | {"Free"} # 허용되는 defect type을 만듦
        unknown = set(self.df["defect_type"]) - allowed_types # 허용되지 않는 defect type 있는지 검사

        if unknown:
            raise ValueError(f"알 수 없는 결함 종류: {unknown}")

    def __len__(self):
        return len(self.df) # 이미지 몇 장 있는지 반환

    def __getitem__(self, idx): # idx 번째 이미지를 가져오는 함수
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        mask_path = row["mask_path"]
        defect_type = row["defect_type"]
        is_defect = int(row["is_defect"])

        # 영상은 grayscale 1채널로 읽기
        with Image.open(image_path) as src:
            image = src.convert("L") # L은 PIL에서 grayscale 이미지 의미 => 최종 이미지 tensor shape = [1, H, W]

        # 정상 여부와 결함 종류 정보가 일치하는지 확인
        if is_defect != int(defect_type != "Free"):
            raise ValueError(f"라벨 정보가 일치하지 않습니다: {image_path}")

        # 결함 이미지라면 mask가 있어야 함
        if is_defect:
            if not mask_path:
                raise ValueError(f"결함 마스크가 없습니다: {image_path}")

            with Image.open(mask_path) as src:
                mask = src.convert("L") # mask도 grayscale로 읽음

            if mask.size != image.size: # 원본 이미지와 mask의 크기가 같은지 확인
                raise ValueError(f"이미지와 마스크 크기가 다릅니다: {image_path}")
        else:
            # 정상 이미지는 모든 픽셀이 배경
            # 정상 이미지는 결함이 없으므로 별도의 mask 이미지가 필요하지 않음
            # 모든 pixel = 0 인 mask를 만들어줌
            mask = Image.new("L", image.size, color=0)

        size = [self.image_size, self.image_size]

        # 이미지에는 bilinear 보간
        # ex) 100 120 => 중간 pixel -> 110 정도
        image = TF.resize(
            image,
            size,
            interpolation=InterpolationMode.BILINEAR,
        )

        # 마스크에는 nearest 보간: 클래스 경계를 섞지 않음 (클래스 의미를 갖기 때문)
        mask = TF.resize(
            mask,
            size,
            interpolation=InterpolationMode.NEAREST,
        )

        # 학습 때만 증강
        # 이미지와 정답 마스크를 반드시 같은 방향으로 변환
        if self.train: # 학습 데이터일 때만 augmentation 적용
            # random.random()은 0~1사이 값 만들기 때문에 약 50% 확률로 flip
            if random.random() < 0.5:
                # image와 mask를 둘다 flip
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
            class_id = DEFECT_TO_ID[defect_type] # 결함 영역에 class ID 넣기
            # mask에서 결함이 있는 pixel만 해당 classID로 바꿈
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