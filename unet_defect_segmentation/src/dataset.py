import random

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from config import IMAGE_SIZE
from splits import read_csv

class MagneticTileDataset(Dataset):
    def __init__(self, csv_path, train=False):
        """
        csv_path:
            prepare_data.py에서 만든 train.csv / val.csv / test.csv 경로

        train:
            True  -> 학습 데이터, augmentation 적용
            False -> validation/test, augmentation 적용 안 함
        """
        # CSV에는 REPO_ROOT 기준 상대경로가 저장되어 있다.
        # read_csv()가 절대경로로 복원해 주므로
        # 어느 위치에서 실행해도 파일을 찾을 수 있다.
        self.df = read_csv(csv_path)
        self.train = train

    def __len__(self):
        """Dataset 전체 이미지 개수를 반환."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        idx번째 이미지와 Ground Truth Mask를 읽어서 반환한다.
        """

        # 1. CSV에서 idx번째 이미지 정보 가져오기
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        mask_path = row["mask_path"]
        defect_type = row["defect_type"]
        is_defect = int(row["is_defect"])
        
        # 2. 원본 이미지 읽기
        # Magnetic Tile은 grayscale이므로 1채널("L")로 읽음
        image = Image.open(image_path).convert("L")
        
        # 3. Ground Truth Mask 읽기
        if is_defect == 1 and not pd.isna(mask_path) and str(mask_path) != "":
            # Defect 이미지라면 실제 mask 파일을 읽음
            mask = Image.open(str(mask_path)).convert("L")
        else:
            #  Normal(free) 이미지라면 defect가 없으므로 모든 픽셀이 0인 mask를 만든다
            mask = Image.new("L", image.size, color=0)
            
        # 4. Image와 Mask 크기를 동일하게 256x256으로 맞춤
        # Image는 일반적인 이미지이므로 bilinear interpolation 사용 
        image = TF.resize(
            image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR
        )
        
        # Mask는 class label 이므로 반드시 nearest interpolation 사용
        # Bilinear를 쓰면 0과 255 사이의 애매한 값 생길수도 있음
        mask = TF.resize(
            mask, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.NEAREST
        )
        
        # 5. Data Augmentation
        # 학습데이터에만 적용함
        if self.train: 
            # 50% 확률로 좌우 반전
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                
            # 50% 확률로 상하 반전
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                
        # 6. Image를 Tensor로 변환 
        # PIL Image: HxW 0~255
        # Tensor: 1xHxW, 0~1
        image = TF.to_tensor(image)
        
        # Image normalization
        # 0~1 범위를 대략 -1~1 범위로 변경
        image = TF.normalize(
            image, mean=[0.5], std=[0.5]
        )
        
        # 7. Mask를 Binary Tensor로 변환
        mask = np.array(mask)
        
        # mask가 0/255 형태라고 가정 127보다 크면 defect=1 아니면 background=0
        mask = (mask > 127).astype(np.float32)
        
        # numpy -> PyTorch Tensor
        mask = torch.from_numpy(mask)
        
        # HxW -> 1xHxW
        # U-Net output과 shape을 맞추기 위해 channel 차원 추가
        mask = mask.unsqueeze(0)
        
        # 8. 필요한 데이터 반환
        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": "" if pd.isna(mask_path) else str(mask_path),
            "defect_type": defect_type,
            "is_defect": is_defect
        }