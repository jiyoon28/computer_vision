from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

# Iamge preprocessing

# Compose : 여러 개의 이미지 전처리를 순서대로 묶는 도구
transform = transforms.Compose([
    
    # 원본 grayscale -> RGB 3채널
    # 원본 이미지는 grayscale 이미지, 보통 HxW, 1xHxW형태인데
    # ResNet-18은 ImageNet에서 RGB로 학습되었기 때문에 입력 channel이 3개여야함
    # 즉 한 픽셀이 128이였다면 [128, 128, 128]로 복제한다고 생각
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)), # 이미지 크기를 모두 224x224로 바꾸기
    transforms.ToTensor(), # PIL Image나 numpy image를 PyTorch Tensor로 바꿔줌 channel x height x width
    
    # ImageNet normalization Tensor 값을 한번 더 정규화
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # 각 채널의 ImageNet 데이터에서 계산된 평균값 [R,G,B]
        std=[0.229, 0.224, 0.225] 
    )
])

def get_dataloaders(batch_size=32):
    # ImageFolder가 자동으로 폴더 이름을 class로 인식함
    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=transform # 이미지를 꺼낼 때 아까 만든 전처리를 자동으로 적용하라는 뜻
    )
    
    val_dataset = datasets.ImageFolder(
        DATA_DIR / "val",
        transform=transform
    )
    
    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=transform
    )
    
    # train_dataset을 Data Loader로 감쌈
    train_loader = DataLoader(
        train_dataset, # 전체 train이미지와 label이 들어있는 dataset
        batch_size=batch_size, # 이미지 한번에 몇 장씩 가져올지 결정
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True 
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

if __name__ == "__main__":

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    ) = get_dataloaders()

    print("Class mapping:",train_dataset.class_to_idx)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    images, labels = next(iter(train_loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Labels:", labels)