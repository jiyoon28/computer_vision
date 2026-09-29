from pathlib import Path
# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# convnext_defect_classification/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# computer_vision/  (classification / segmentation 공용 데이터 루트)
REPO_ROOT = Path(__file__).resolve().parents[2]

# ResNet, DINOv2와 같은 분할을 사용해야 결과를 공정하게 비교할 수 있음
DATA_DIR = REPO_ROOT / "data" / "processed" / "classification_by_type"

# ConvNeXt-V2 사전학습 가중치(ImageNet)의 정규화 값
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomRot90:
    """0/90/180/270도 중 하나로 회전. 결함은 방향과 상관없이 같은 종류이므로 라벨이 바뀌지 않음"""

    def __call__(self, image):
        k = int(torch.randint(0, 4, (1,)))
        return torch.rot90(image, k, dims=(1, 2))


def build_transforms(image_size):
    # 원본 크기가 약 100~630 x 220~400 이라 384면 대부분 원본 해상도에 가깝게 유지됨
    # ResNet과 같은 방식(정사각형 resize)으로 맞춰서 입력 처리 차이를 줄임
    base = [
        transforms.Grayscale(num_output_channels=3),  # 흑백 -> RGB 3채널 복제
        transforms.Resize((image_size, image_size)),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    # 작은 결함이 잘려 나갈 수 있어서 RandomResizedCrop 같은 crop 증강은 쓰지 않음
    train_transform = transforms.Compose(
        base
        + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # 촬영 조건 차이 정도만 흉내 내도록 약하게 설정
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        + normalize
        + [RandomRot90()]
    )
    eval_transform = transforms.Compose(base + normalize)
    return train_transform, eval_transform


def get_datasets(image_size=384):
    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)

    if not (
        train_dataset.class_to_idx
        == val_dataset.class_to_idx
        == test_dataset.class_to_idx
    ):
        raise ValueError("train / val / test의 클래스 구성이 다릅니다.")

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(image_size=384, batch_size=16, num_workers=4, seed=42):
    train_dataset, val_dataset, test_dataset = get_datasets(image_size)

    # seed를 고정한 generator로 섞는 순서를 재현 가능하게 만듦
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_args
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = get_datasets()
    print("Class mapping:", train_dataset.class_to_idx)
    print("Train / Val / Test:", len(train_dataset), len(val_dataset), len(test_dataset))
    image, label = train_dataset[0]
    print("Image shape:", image.shape, "Label:", label)
