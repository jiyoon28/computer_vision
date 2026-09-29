import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 멀티클래스 segmentation용 손실함수
# CE(Cross Entropy) + Dice(Dice Loss)
class CEDiceLoss(nn.Module):
    def __init__(self, dice_weight=1.0): # Dice Loss 중요도 반영
        super().__init__()

        # 각 pixel을 하나의 클래스에 배정하는 역할
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        """
        logits:  [B, C, H, W] — 모델의 클래스별 점수 (이미지개수, 클래스개수, 픽셀크기) U-Net의 출력
        targets: [B, H, W]    — 정답 클래스 번호, torch.long (정답 mask)
        """

        # 1. 픽셀별 클래스 분류 손실(첫번째 loss)
        ce_loss = self.ce(logits, targets)

        # 2. 클래스별 확률
        # 학습용 Dice에서는 argmax를 사용하지 않음!
        # softmax 확률을 사용해야 gradient가 전달됨.
        # 모델이 출력한 logits를 클래스 확률로 바꿈 -> 0~1사이로 바꿈
        # dim=1인 이유: 모델 출력이 [Batch,Class,Height,Width] => class 방향으로 softmax 해야함
        probabilities = torch.softmax(logits, dim=1)

        num_classes = logits.shape[1]

        # 정답을 클래스별 0/1 마스크로 변환
        # [B, H, W] → [B, H, W, C] → [B, C, H, W]
        # 한 pixel의 정답이 class 3 => [0,0,0,1,0,0] => [B,H,W] -> [B,H,W,C]
        target_one_hot = F.one_hot(
            targets,
            num_classes=num_classes,
        ).permute(0, 3, 1, 2).to(probabilities.dtype) # one-hot 결과는 integer => probabilities는 floate -> Dice 계산 위해 데이터 타입 맞춰줌

        # 배경 채널 0을 제외하고 결함 5개 채널만 사용 (결함 5종류만 Dice 계산에 사용)
        # 배경 제외 안하면 실제 결함을 못잡아도 평균 Dice가 좋아보이는 문제 생김
        probabilities = probabilities[:, 1:]
        target_one_hot = target_one_hot[:, 1:]

        # 배치·높이·너비 방향으로 합산 → 클래스별 값
        dims = (0, 2, 3)

        # 실제 정답 영역에 대해 모델이 얼마나 높은 확률을 줬는지 계산
        intersection = (
            probabilities * target_one_hot
        ).sum(dim=dims)

        predicted_area = probabilities.sum(dim=dims)
        target_area = target_one_hot.sum(dim=dims)

        smooth = 1e-6 # Dice공식에서 0으로 나누기 방지

        # Dice 같이 쓰는 이유: 배경이 결함보다 압도적으로 많음
        # 모델이 거의 다 배경이라고 예측해도 전체 정확도 높게 나옴
        # 결함 영역 자체를 얼마나 제대로 잡았는지 평가하는 Dice Loss 같이 사용
        # dice 높을수록 segmentatio 잘된거
        dice = (
            2 * intersection + smooth
        ) / (
            predicted_area + target_area + smooth
        )

        # 현재 배치에 정답 영역이 있는 결함만 Dice 평균에 포함
        # 없는 클래스의 잘못된 예측은 Cross Entropy가 벌점을 줌
        # ex) present = [True, False, True, False, True]
        present = target_area > 0

        # present 중 하나라도 true가 있는지 확인
        if present.any():
            # dice = 0.9 diceloss = 0.1 -> 잘했으니까 loss가 작음
            # dice = 0.2 diceloss = 0.8 -> 못했으니까 loss가 큼
            # dice[present] 써서 실제로 존재하는 결함 종류만 평균 냄
            # ex) Crack, Break 만 있으면 이 두 class Dice만 계산함
            dice_loss = (1 - dice[present]).mean()
        else:
            # 정상 이미지만 있는 배치는 CE로 학습
            dice_loss = probabilities.sum() * 0.0

        return ce_loss + self.dice_weight * dice_loss

# Loss 객체를 만들어 반환하는 helper 함수
def build_loss():
    return CEDiceLoss(dice_weight=1.0)