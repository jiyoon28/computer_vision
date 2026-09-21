from pathlib import Path
# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import argparse
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import get_dataloaders # dataset.py 에서 DataLoader 생성 함수 가져오기
from model import create_model # model.py 에서 ResNet 모델 생성 함수 가져오기

parser = argparse.ArgumentParser()
parser.add_argument("--unfreeze-layer4", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# 같은 seed로 비교하면 초기 분류층과 데이터 섞는 순서의 차이를 줄일 수 있음
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 현재 train.py 기준으로 프로젝트 루트 경로 찾기

MODEL_DIR = PROJECT_ROOT / "models" # 학습된 모델을 저장할 폴더 경로
MODEL_DIR.mkdir(exist_ok=True) # models 폴더가 없으면 생성

experiment = (
    "layer4_fc"
    if args.unfreeze_layer4
    else "fc_only"
)

# 기존 결과인 resnet18_6class_best.pth 덮어쓰지 않고 별도 파일로 저장
checkpoint_path = (
    MODEL_DIR
    / f"resnet18_6class_{experiment}_seed{args.seed}.pth"
)

print("Experiment:", experiment)
print("Checkpoint:", checkpoint_path)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using device: {device}")

# Dataset
train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(batch_size=32)
print(train_dataset.class_to_idx) # 클래스 이름과 숫자 label 매핑 확인, 예: {'defect': 0, 'normal': 1}

# Model
class_names = train_dataset.classes

model = create_model(
    num_classes=len(class_names),
    freeze_backbone=True,
    unfreeze_layer4=args.unfreeze_layer4,
).to(device)

print("Classes:", class_names)

print("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(" ", name)

# Loss
criterion = nn.CrossEntropyLoss() # 다중 분류용 Cross Entropy Loss 사용

# Optimizer
# requires_grad=True로 바꿔도 optimizer에 넣지 않으면 업데이트되지 않음
if args.unfreeze_layer4:
    optimizer = Adam(
        [
            {
                "params": model.layer4.parameters(),
                "lr": 1e-4, # 이미 사전학습된 layer4는 작은 학습률로 조정
            },
            {
                "params": model.fc.parameters(),
                "lr": 1e-3, # 새로 초기화한 fc는 더 큰 학습률로 학습하도록 시작하는 설정
            },
        ]
    )
else:
    optimizer = Adam(
        model.fc.parameters(),
        lr=1e-3,
    )

# Training settings
num_epochs = 50 # 전체 training dataset을 10번 반복해서 학습
best_val_accuracy = -1.0 # 현재까지 가장 높은 validation accuracy 저장용

patience = 5 # 5 epoch 동안 성능이 개선되지 않으면 학습 종료
epochs_without_improvement = 0 # 성능이 개선되지 않은 epoch 수

for epoch in range(num_epochs): 
    
    # training mode
    model.train() # 모델을 학습 모드로 전환
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    train_loss = 0.0 # 한 epoch 동안의 loss 누적값 초기화
    train_correct = 0 # 맞게 예측한 이미지 개수 초기화
    train_total = 0 # 전체 학습 이미지 개수 초기화 
    
    for images, labels in train_loader: # train_loader 에서 batch 단위로 이미지와 정답 가져오기
        images = images.to(device) # 이미지 Tensor을 GPU 또는 CPU로 이동
        labels = labels.to(device) # 정답 label을 GPU 또는 CPU로 이동
        
        optimizer.zero_grad() # 이전 batch에서 계산된 gradient 초기화
        outputs = model(images) # Forward propagation: 이미지 입력 -> 모델 예측값 출력

        loss = criterion(outputs, labels) #  모델 예측값과 실제 label을 비교해서 Loss 계산
        loss.backward() # Backpropagation: Loss를 기준으로 gradient 계산
        
        optimizer.step() # 계산된 gradient를 이용해서 FC head의 weight 업데이트
        train_loss += loss.item() # 현재 batch의 loss 값을 epoch loss에 누적
        _, predicted = torch.max(outputs, dim=1) # 두 class score중 가장 큰 값의 index를 예측 class로 선택
        train_total += labels.size(0) # 현재 batch의 이미지 개수를 전체 이미지 수에 추가
        train_correct += (predicted == labels).sum().item() # 예측값과 실제값이 같은 이미지 개수 누적
        
    train_accuracy = train_correct / train_total # epoch 전체 training accuracy 계산
    
    # validation
    model.eval() # 모델을 평가 모드로 전환
    val_correct = 0 # validation에서 맞게 예측한 이미지 수 초기화
    val_total = 0 # validation전체 이미지 수 초기화
    
    with torch.no_grad(): 
        for images, labels in val_loader: # validation 데이터를 batch단위로 가져오기
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images) # Forward propagation만 수행
            
            _, predicted = torch.max(outputs, dim=1) # 가장 큰 class score의 index 선택
            val_total += labels.size(0) # validation 이미지 수 누적
            val_correct += (predicted == labels).sum().item() # 맞게 예측한 validation 이미지 수 누적
        
    val_accuracy = val_correct / val_total # validation accuracy 계산
    
    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {train_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )  # 현재 epoch의 평균 loss, training accuracy, validation accuracy 출력

    # Best Model 저장
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_accuracy": val_accuracy,
                "epoch": epoch + 1,
                "experiment": experiment,
                "seed": args.seed,
                "unfreeze_layer4": args.unfreeze_layer4,
            },
            checkpoint_path,
        )
        
        print("Best model saved.")

    else:
        epochs_without_improvement += 1
        print(
            f"No improvement: "
            f"{epochs_without_improvement}/{patience}"
        )

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break
