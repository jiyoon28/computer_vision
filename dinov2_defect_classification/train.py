# 현재 Anaconda 환경에서는 NumPy를 먼저 로드해야 OpenMP 초기화 충돌을 피할 수 있음.
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classifier import DINOClassifier
from config import (
    TRAIN_DIR,
    VAL_DIR,
    OUTPUT_DIR,
    BATCH_SIZE,
    DEVICE,
)
from dataset import ImageFolderWithPaths, transform


def main():
    # 1. 데이터 준비
    train_dataset = ImageFolderWithPaths(
        TRAIN_DIR,
        transform=transform,
    )
    val_dataset = ImageFolderWithPaths(
        VAL_DIR,
        transform=transform,
    )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train과 val의 클래스 구성이 다릅니다.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Classes:", train_dataset.class_to_idx)

    # 2. 모델 준비
    model = DINOClassifier(
        num_classes=len(train_dataset.classes)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # DINOv2는 고정하고 분류층만 학습
    optimizer = torch.optim.Adam(
        model.head.parameters(),
        lr=1e-3,
    )

    num_epochs = 20
    best_val_accuracy = -1.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "dinov2_classifier_best.pth"

    # 3. 여러 epoch 반복
    for epoch in range(num_epochs):
        model.train()

        # 사전학습 특징 추출기는 평가 모드 유지
        model.backbone.eval()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 사용자가 질문한 코드는 이 배치 반복문 안에 들어감
        for images, labels, paths in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            train_total += labels.size(0)

        # 4. 매 epoch 검증
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels, paths in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(images)

                val_correct += (
                    logits.argmax(dim=1) == labels
                ).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {train_loss / train_total:.4f} "
            f"Train Acc: {train_correct / train_total:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # 5. 검증 정확도가 가장 높은 분류층 저장
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "head_state_dict": model.head.state_dict(),
                    "class_names": train_dataset.classes,
                    "backbone_name": "dinov2_vits14",
                    "val_accuracy": val_accuracy,
                    "epoch": epoch + 1,
                },
                save_path,
            )
            print(f"Best classifier saved: {save_path}")


if __name__ == "__main__":
    main()
