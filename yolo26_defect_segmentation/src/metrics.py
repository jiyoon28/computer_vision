"""U-Net과 동일하게 전체 픽셀 누적 및 배경 제외 평균을 사용한다."""

import numpy as np


def update_confusion_matrix(matrix, predictions, targets):
    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes differ.")
    count = matrix.shape[0]
    pred = np.asarray(predictions, dtype=np.int64).ravel()
    true = np.asarray(targets, dtype=np.int64).ravel()
    if ((pred < 0) | (pred >= count) | (true < 0) | (true >= count)).any():
        raise ValueError("Mask contains an out-of-range class ID.")
    matrix += np.bincount(count * true + pred, minlength=count ** 2).reshape(count, count)


def compute_metrics(matrix):
    matrix = matrix.astype(np.float64)
    tp = matrix.diagonal()
    true = matrix.sum(axis=1)
    pred = matrix.sum(axis=0)
    union = true + pred - tp
    valid = union > 0
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=valid)
    dice = np.divide(2 * tp, true + pred, out=np.full_like(tp, np.nan), where=valid)
    return {
        "iou": iou.tolist(), "dice": dice.tolist(),
        "foreground_miou": float(iou[1:][valid[1:]].mean()) if valid[1:].any() else 0.0,
        "foreground_dice": float(dice[1:][valid[1:]].mean()) if valid[1:].any() else 0.0,
    }
