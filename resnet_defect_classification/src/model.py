# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def create_model(num_classes=2, freeze_backbone=True, unfreeze_layer4=False):
    # ImageNet pretrained ResNet
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Backbone freeze 기존 파라미터 전체 고정
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # 옵션을 켰으면 마지막 특징 추출 블록만 학습 허용
    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True
    
    # 기존 IamgeNet head 제거
    # model.fc = ResNet-18의 마지막 Fully Connected layer => Linear(in_features=512, out_features=1000)
    # 즉 512개의 입력 feature -> Linear Layer -> 1000개의 출력 Class
    # in_features는 그 Linear Layer가 받는 입력 feature 개수를 가져오는 속성
    # ResNet-18에서는 model.fc.in_features 결과가 512임
    # 모델에게 직접 물어보는 방식 => 나중에 ResNet-50으로 바꾼다면 입력 feature 수가 달라짐
    input_features = model.fc.in_features
    
    # ResNet의 마지막 분류층을 정상/결함 분류용으로 교체
    # 새로운 Normal / Defect head 
    # 새로운 Linear layer를 만들어서 기존 model.fc 자리에 넣음
    # model.fc: ResNet의 마지막 분류층. fc는 Fully Connected의 약자
    # input_features: 앞쪽 ResNet이 이미지에서 추출하는 특징 개수. ResNet-18에서는 512개
    # num_classes: 구분할 클래스 개수. 현재는 defect/normal 2개
    # 현재 코드에서는 nn.Linear(512,2)
    # 결과: [결함 점수, 정상 점수] = [2.3, 0.7]
    # 이 점수는 아직 확률이 아닌 logits임 
    model.fc = nn.Linear(input_features, num_classes)
    
    return model

if __name__ == "__main__":
    model = create_model()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
