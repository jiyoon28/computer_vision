import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from dataset import MagneticTileDataset
from model import UNet
from losses import build_loss
from metrics import update_confusion_matrix, compute_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, optimizer=None):
    """
    optimizer가 있으면 학습, 없으면 검증·평가.
    evaluate.py에서도 이 함수를 재사용한다.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_images = 0

    # epoch 전체 픽셀의 정답·예측을 누적
    matrix = torch.zeros(
        cfg.NUM_CLASSES,
        cfg.NUM_CLASSES,
        dtype=torch.int64,
        device=cfg.DEVICE,
    )

    for batch in loader:
        images = batch["image"].to(cfg.DEVICE)
        targets = batch["mask"].to(cfg.DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        # 검증에서는 gradient를 계산하지 않음
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

        # 채널 방향으로 가장 높은 점수의 클래스 선택
        predictions = logits.detach().argmax(dim=1)

        update_confusion_matrix(
            matrix,
            predictions,
            targets,
        )

    metrics = compute_metrics(matrix)
    metrics["loss"] = total_loss / total_images

    return metrics


def main():
    set_seed(cfg.SEED)

    train_dataset = MagneticTileDataset(
        cfg.SPLIT_DIR / "train.csv",
        train=True,
    )
    val_dataset = MagneticTileDataset(
        cfg.SPLIT_DIR / "val.csv",
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # 自作 U-Net: 사전학습 가중치 없이 처음부터 학습
    model = UNet(
        in_channels=1,
        out_channels=cfg.NUM_CLASSES,
    ).to(cfg.DEVICE)

    criterion = build_loss()

    # U-Net 전체 파라미터를 학습
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
    )

    best_score = -1.0
    epochs_without_improvement = 0

    print("Device:", cfg.DEVICE)
    print("Classes:", cfg.CLASS_NAMES)
    print("Train / Val:", len(train_dataset), len(val_dataset))

    for epoch in range(cfg.NUM_EPOCHS):
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, criterion
        )

        score = val_metrics["foreground_miou"]

        print(
            f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}] "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Val FG mIoU: {score:.4f} "
            f"Val FG Dice: {val_metrics['foreground_dice']:.4f}"
        )

        # 배경 제외 평균 IoU가 가장 좋은 모델 저장
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": cfg.CLASS_NAMES,
                    "image_size": cfg.IMAGE_SIZE,
                    "epoch": epoch + 1,
                    "val_foreground_miou": score,
                    "seed": cfg.SEED,
                },
                cfg.CHECKPOINT_PATH,
            )

            print("Best model saved:", cfg.CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= cfg.EARLY_STOPPING_PATIENCE
        ):
            print("Early stopping.")
            break


if __name__ == "__main__":
    main()