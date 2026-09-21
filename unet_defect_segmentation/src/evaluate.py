from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import config as cfg
from dataset import MagneticTileDataset
from model import UNet
from losses import build_loss
from train import run_epoch


def main():
    checkpoint = torch.load(
        cfg.CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    # 클래스 순서가 달라지면 같은 숫자도 의미가 달라짐
    if checkpoint["class_names"] != cfg.CLASS_NAMES:
        raise ValueError("체크포인트와 현재 클래스 순서가 다릅니다.")

    test_dataset = MagneticTileDataset(
        cfg.SPLIT_DIR / "test.csv",
        train=False,
        image_size=checkpoint["image_size"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(
        in_channels=1,
        out_channels=cfg.NUM_CLASSES,
    ).to(cfg.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    # 전체 test 픽셀에 대해 점수 계산
    metrics = run_epoch(
        model,
        test_loader,
        build_loss(),
    )

    print("Test images:", len(test_dataset))
    print("Best epoch:", checkpoint["epoch"])
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Foreground mIoU: {metrics['foreground_miou']:.4f}")
    print(f"Foreground Dice: {metrics['foreground_dice']:.4f}")

    print("\nClass          IoU      Dice")
    for index, name in enumerate(cfg.CLASS_NAMES):
        print(
            f"{name:12s} "
            f"{metrics['iou'][index]:.4f}   "
            f"{metrics['dice'][index]:.4f}"
        )

    # 숫자 라벨을 RGB 색으로 바꾸기 위한 표
    palette = np.asarray(cfg.PALETTE, dtype=np.uint8)

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(cfg.DEVICE)
            predictions = model(images).argmax(dim=1).cpu().numpy()
            targets = batch["mask"].numpy()

            # 입력 정규화를 되돌려 0~255 grayscale로 복원
            originals = (
                (batch["image"][:, 0].numpy() * 0.5 + 0.5)
                .clip(0, 1) * 255
            ).astype(np.uint8)

            for i, image_path in enumerate(batch["image_path"]):
                path = Path(image_path)

                # 서로 다른 원본 클래스 폴더의 파일명 충돌 방지
                name = f"{path.parent.parent.name}_{path.stem}"

                pred = predictions[i].astype(np.uint8)
                target = targets[i].astype(np.uint8)

                # 원시 마스크: 픽셀값 자체가 클래스 번호 0~5
                Image.fromarray(pred).save(
                    cfg.PREDICTION_DIR / f"{name}.png"
                )

                # 시각화: 원본 | 정답 | 예측을 가로로 붙임
                original_rgb = np.repeat(
                    originals[i, :, :, None], 3, axis=2
                )

                comparison = np.concatenate(
                    [
                        original_rgb,
                        palette[target],
                        palette[pred],
                    ],
                    axis=1,
                )

                Image.fromarray(comparison).save(
                    cfg.VIS_DIR / f"{name}.png"
                )

    print("\nRaw masks:", cfg.PREDICTION_DIR)
    print("Original | GT | Prediction:", cfg.VIS_DIR)


if __name__ == "__main__":
    main()