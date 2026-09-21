import torch
import torch.nn as nn

#  기존 파일의 모델 로딩 함수를 재사용
from extract_features import load_model

class DINOClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 기존 load_model()에서 가중치를 이미 고정함
        self.backbone = load_model()
        # 현재 사용중인 dinov2_vits14는 384차원 특징 출력
        self.head = nn.Linear(384, num_classes)
        
    def forward(self, images):
        # 분류기 학습 중에도 DINOv2는 평가 모드 유지
        self.backbone.eval()
        
        # DINOv2의 gradient는 계산하지 않음
        with torch.no_grad():
            features = self.backbone(images) # [B, 384]
            
        # 이 층은 gradient를 계산하고 학습함
        logits = self.head(features) # [B, 6]
        return logits
