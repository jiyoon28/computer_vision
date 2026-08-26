from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import get_dataloaders # dataset.py 에서 DataLoader 생성 함수 가져오기
from model import create_model # model.py 에서 ResNet 모델 생성 함수 가져오기

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 현재 train.py 기준으로 프로젝트 루트 경로 찾기

MODEL_DIR = PROJECT_ROOT / "models" # 학습된 모델을 저장할 폴더 경로
MODEL_DIR.mkdir(exist_ok=True) # models 폴더가 없으면 생성

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using device: {device}")

# Dataset
train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(batch_size=32)
print(train_dataset.class_to_idx) # 클래스 이름과 숫자 label 매핑 확인, 예: {'defect': 0, 'normal': 1}

# Model
model = create_model(num_classes=2, freeze_backbone=True) # 출력 class는 2개, pretrained backbone은 freeze
model = model.to(device) # 모델의 parameter을 GPU 또는 CPU로 이동

# Loss
criterion = nn.CrossEntropyLoss() # 다중 분류용 Cross Entropy Loss 사용

# Optimizer
optimizer = Adam(model.fc.parameters(), lr=1e-3) # freeze된 backbone은 제외하고 마지막 FC head만 학습

# Training settings
num_epochs = 50 # 전체 training dataset을 10번 반복해서 학습
best_val_accuracy = 0.0 # 현재까지 가장 높은 validation accuracy 저장용

patience = 5 # 5 epoch 동안 성능이 개선되지 않으면 학습 종료
epochs_without_improvement = 0 # 성능이 개선되지 않은 epoch 수

for epoch in range(num_epochs): 
    
    # training mode
    model.train() # 모델을 학습 모드로 전환
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
        torch.save(
            model.state_dict(),
            MODEL_DIR / "resnet18_best.pth"
        ) # 현재 모델의 weight를 파일로 저장
        print("Best model saved.")
        
    else: # validation 성능이 좋아지지 않았다면
        epochs_without_improvement += 1 # 개선 없는 epoch 수 + 1
        print(
            f"No improvement: "
            f"{epochs_without_improvement}/{patience}"
        )
    if epochs_without_improvement >= patience: # patience 만큼 개선이 없으면
        print("Early stopping triggered.")
        break # 전체 epoch 반복문 종료