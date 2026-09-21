# 전체 흐름
# TEST_DIR 이미지 -> ImageFolderWithPaths로드 -> DataLoader로 batch 생성
# -> DINOv2 pretrained model 로드 -> 이미지 -> feature vector 추츨
# -> feature/label/path 저장 -> dinov2_features.npz 생성
import numpy as np
import torch

# DataLoader: dataset을 한 장씩이 아니라 batch 단위로 모델에 넣게 해주는 도구
from torch.utils.data import DataLoader
from config import (
    TEST_DIR, # normal / defect 이미지가 있는 위치
    OUTPUT_DIR, # 결과 저장 위치
    MODEL_NAME, # 사용할 DINOv2 모델
    BATCH_SIZE, # 한 번에 처리할 이미지 수
    DEVICE, # CPU or GPU 
)

# ImageFolderWithPaths: image + label + path 반환 
from dataset import ImageFolderWithPaths, transform

# DINOv2 모델 불러오기
def load_model():
    print ("Loading pretrained DINOv2...")
    # Facebook Research 가 공개한 DINOv2 모델을 torch.hub을 통해 불러옴
    model = torch.hub.load(
        "facebookresearch/dinov2",
        MODEL_NAME # dinov2_vits14 모델 볼러옴
    )
    model = model.to(DEVICE)
    # 모델을 평가/Inference 모드로 바꿈
    # 지금 목적은 feature extraction이므로 eval()을 사용 
    model.eval()
    
    # DINOv2 weight 고정
    for param in model.parameters():
        # 지금 DINOv2는 feature extractor로만 사용
        # 입력 이미지 -> DINOv2 -> 384차원 feature
        param.requires_grad = False # 모든 모델 파라미터를 학습하지 않도록 고정
    return model

# 실제 이미지를 읽고 feature를 만드는 핵심 함수
def extract_features():
    dataset = ImageFolderWithPaths(
        TEST_DIR, #  TEST_DIR 안의 이미지를 읽음
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # Feature, label, path가 정확히 대응되어야 함
        num_workers=0 # 이미지를 로드하는 별도 프로세스르 만들지 않음
    )
    
    print("Class mapping:")
    print(dataset.class_to_idx) # ex) {'defect':0, 'normal':1}
    
    model = load_model() # 위에서 만든 load_model이용해서 DINOv2 준비
    
    all_features = [] # 각 batch에서 나온 결과를 계속 모으기 위한 리스트
    all_labels = []
    all_paths = []
    
    with torch.no_grad(): # gradient 계산 안함 모델 학습 안하니까 
        # DataLoader에서 batch 가져오기
        for images, labels, paths in dataloader:
            images = images.to(DEVICE)
            
            # 핵심
            # images -> (batch, 3, 244, 244) -> DINOv2 -> features => (batch, 384) dinov2_vits14를 사용하면 이미지 하나를 대표하는 feature가 384차원 벡터로 나옴
            features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_paths.extend(paths)
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print()
    print("Feature extraction complete")
    print("Feature shape:", features.shape)
    print("Labels shape:", labels.shape)
    
    save_path = OUTPUT_DIR / "dinov2_features.npz"
    
    # .npz: 여러 numpy 배열을 하나의 파일에 저장할 수 있는 형식
    np.savez(
        save_path,
        features=features,
        labels=labels,
        paths=np.array(all_paths),
        class_names=np.array(dataset.classes)
    )

    print()
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    extract_features()