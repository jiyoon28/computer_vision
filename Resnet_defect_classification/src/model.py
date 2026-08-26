import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def create_model(num_classes=2, freeze_backbone=True):
    # ImageNet pretrained ResNet
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Backbone freeze
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # 기존 IamgeNet head 제거
    # model.fc = ResNet-18의 마지막 Fully Connected layer => Linear(in_features=512, out_features=1000)
    # 즉 512개의 입력 feature -> Linear Layer -> 1000개의 출력 Class
    # in_features는 그 Linear Layer가 받는 입력 feature 개수를 가져오는 속성
    # ResNet-18에서는 model.fc.in_features 결과가 512임
    # 모델에게 직접 물어보는 방식 => 나중에 ResNet-50으로 바꾼다면 입력 feature 수가 달라짐
    input_features = model.fc.in_features
    
    # 새로운 Normal / Defect head
    # 새로운 Linear layer를 만들어서 기존 model.fc 자리에 넣음
    model.fc = nn.Linear(input_features, num_classes)
    
    return model

if __name__ == "__main__":
    model = create_model()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)