import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CEDiceLoss(nn.Module):
    def __init__(self, dice_weight=1.0):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        """
        logits:  [B, C, H, W] — 모델의 클래스별 점수
        targets: [B, H, W]    — 정답 클래스 번호, torch.long
        """

        # 1. 픽셀별 클래스 분류 손실
        ce_loss = self.ce(logits, targets)

        # 2. 클래스별 확률
        # 학습용 Dice에서는 argmax를 사용하지 않음!
        # softmax 확률을 사용해야 gradient가 전달됨.
        probabilities = torch.softmax(logits, dim=1)

        num_classes = logits.shape[1]

        # 정답을 클래스별 0/1 마스크로 변환
        # [B, H, W] → [B, H, W, C] → [B, C, H, W]
        target_one_hot = F.one_hot(
            targets,
            num_classes=num_classes,
        ).permute(0, 3, 1, 2).to(probabilities.dtype)

        # 배경 채널 0을 제외하고 결함 5개 채널만 사용
        probabilities = probabilities[:, 1:]
        target_one_hot = target_one_hot[:, 1:]

        # 배치·높이·너비 방향으로 합산 → 클래스별 값
        dims = (0, 2, 3)

        intersection = (
            probabilities * target_one_hot
        ).sum(dim=dims)

        predicted_area = probabilities.sum(dim=dims)
        target_area = target_one_hot.sum(dim=dims)

        smooth = 1e-6

        dice = (
            2 * intersection + smooth
        ) / (
            predicted_area + target_area + smooth
        )

        # 현재 배치에 정답 영역이 있는 결함만 Dice 평균에 포함
        # 없는 클래스의 잘못된 예측은 Cross Entropy가 벌점을 줌
        present = target_area > 0

        if present.any():
            dice_loss = (1 - dice[present]).mean()
        else:
            # 정상 이미지만 있는 배치는 CE로 학습
            dice_loss = probabilities.sum() * 0.0

        return ce_loss + self.dice_weight * dice_loss


def build_loss():
    return CEDiceLoss(dice_weight=1.0)