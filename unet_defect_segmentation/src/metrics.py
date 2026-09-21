import numpy as np
import torch


@torch.no_grad()
def update_confusion_matrix(matrix, predictions, targets):
    """
    matrix:
        [C, C], 행=정답 / 열=예측

    predictions, targets:
        [B, H, W]

    배치별 점수를 평균하는 대신 전체 픽셀 수를 누적한다.
    """
    num_classes = matrix.shape[0]

    true = targets.reshape(-1).long()
    pred = predictions.reshape(-1).long()

    # 정답과 예측 조합을 하나의 정수로 표현
    indices = true * num_classes + pred

    counts = torch.bincount(
        indices,
        minlength=num_classes * num_classes,
    )

    matrix += counts.reshape(num_classes, num_classes)


def compute_metrics(matrix):
    # 작은 혼동행렬만 CPU로 이동해 계산
    matrix = matrix.detach().cpu().double()

    true_positive = matrix.diag()
    true_count = matrix.sum(dim=1)
    predicted_count = matrix.sum(dim=0)

    # IoU = 교집합 / 합집합
    union = true_count + predicted_count - true_positive
    iou = true_positive / union.clamp_min(1)

    # Dice = 2 * 교집합 / (정답 영역 + 예측 영역)
    total = true_count + predicted_count
    dice = 2 * true_positive / total.clamp_min(1)

    # 정답에도 예측에도 없는 클래스는 평균에서 제외
    # 정답에는 없지만 잘못 예측한 클래스는 0점으로 포함됨
    valid = union > 0
    iou[~valid] = float("nan")
    dice[~valid] = float("nan")

    # index 0인 배경을 제외한 평균
    foreground_valid = valid[1:]

    if foreground_valid.any():
        foreground_miou = iou[1:][foreground_valid].mean().item()
        foreground_dice = dice[1:][foreground_valid].mean().item()
    else:
        # 결함 정답·예측이 모두 없는 평가 집합의 처리
        foreground_miou = 0.0
        foreground_dice = 0.0

    return {
        "iou": iou.tolist(),
        "dice": dice.tolist(),
        "foreground_miou": foreground_miou,
        "foreground_dice": foreground_dice,
    }