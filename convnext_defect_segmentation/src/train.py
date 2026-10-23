"""ConvNeXt encoder + 선택한 decoder를 Crack 증강 데이터로 학습한다.

python train.py --decoder unet      (unet / fpn / upernet)
val 결함 mIoU가 가장 높은 epoch를 저장한다. test는 루트의 compare_segmenters.py에서만 사용.
"""

# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import argparse
import json
import math
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

import config as cfg
from dataset import SegDataset
from model import create_model, split_param_groups

# UNet과 같은 loss(CE + Dice)와 지표(전체 픽셀 혼동행렬, 배경 제외 평균)를 써야 비교가 됨
sys.path.append(str(cfg.UNET_SRC))
from losses import build_loss  # noqa: E402
from metrics import compute_metrics, update_confusion_matrix  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", required=True, choices=cfg.DECODERS)
    # imagenet: timm 사전학습 가중치 / clf: 결함 분류로 fine-tune한 ConvNeXt
    parser.add_argument("--init", default="imagenet", choices=["imagenet", "clf"])
    # 100 / 15: YOLO(150 epoch)와 UNet(40 epoch) 사이로 잡은 경험값
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    # 8: YOLO와 같은 batch. RTX 3060 12GB에서 384 + bf16 기준
    parser.add_argument("--batch", type=int, default=8)
    # 학습률 / weight decay / warmup: 분류 학습에서 쓴 값 그대로
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    # 동작 확인용: epoch마다 앞쪽 몇 batch만 사용
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed):
    # 같은 seed로 비교하면 decoder 초기값과 데이터 섞는 순서의 차이를 줄일 수 있음
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_scheduler(optimizer, warmup_steps, total_steps):
    # warmup 동안 학습률을 0에서 천천히 올리고, 이후 cosine으로 0까지 줄임
    # 무작위 상태의 decoder가 만든 큰 gradient가 사전학습 encoder를 망가뜨리는 것을 막음
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, criterion, device, amp_dtype, optimizer=None, scheduler=None, max_batches=None):
    """optimizer가 있으면 학습, 없으면 검증. UNet train.py의 run_epoch에 AMP/scheduler를 더한 것."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_images = 0.0, 0
    matrix = torch.zeros(cfg.NUM_CLASSES, cfg.NUM_CLASSES, dtype=torch.int64, device=device)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(images)
            # Dice의 softmax/합산은 bf16에서 오차가 커서 loss는 float32로 계산
            loss = criterion(logits.float(), targets)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        total_images += images.size(0)
        update_confusion_matrix(matrix, logits.detach().argmax(dim=1), targets)

    metrics = compute_metrics(matrix)
    metrics["loss"] = total_loss / total_images
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # RTX 30xx(Ampere) 이상은 bf16 지원. fp16과 달리 GradScaler 없이도 안정적
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    print(f"Using device: {device}, AMP: {amp_dtype}")

    # SegFormer 전체 모델은 ConvNeXt를 쓰지 않으므로 이름에 convnext_를 붙이지 않음
    prefix = "" if args.decoder.startswith("segformer_") else "convnext_"
    run_name = f"{prefix}{args.decoder}_{args.init}_seed{args.seed}"
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_set = SegDataset("train", train=True)
    val_set = SegDataset("val")
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {"batch_size": args.batch, "num_workers": args.workers,
                     "pin_memory": device.type == "cuda", "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}")

    model = create_model(
        args.decoder, cfg.NUM_CLASSES, cfg.DEFAULT_MODEL_NAME,
        init_checkpoint=cfg.CLF_CHECKPOINT if args.init == "clf" else None,
    ).to(device)
    print(f"Decoder params: {sum(p.numel() for p in model.decoder.parameters()) / 1e6:.2f}M  "
          f"Total: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = build_loss()
    optimizer = torch.optim.AdamW(split_param_groups(model, args.encoder_lr, args.decoder_lr, args.weight_decay))
    steps_per_epoch = len(train_loader) if args.max_batches is None else min(len(train_loader), args.max_batches)
    scheduler = build_scheduler(optimizer, args.warmup_epochs * steps_per_epoch, args.epochs * steps_per_epoch)

    history, best_miou, best_epoch = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train = run_epoch(model, train_loader, criterion, device, amp_dtype, optimizer, scheduler, args.max_batches)
        with torch.no_grad():
            val = run_epoch(model, val_loader, criterion, device, amp_dtype, max_batches=args.max_batches)

        history.append({"epoch": epoch, "train_loss": train["loss"], "train_miou": train["foreground_miou"],
                        "val_loss": val["loss"], "val_miou": val["foreground_miou"],
                        "val_dice": val["foreground_dice"], "val_iou": val["iou"]})
        improved = val["foreground_miou"] > best_miou
        print(f"[{epoch:3d}/{args.epochs}] train loss {train['loss']:.4f} mIoU {train['foreground_miou']:.4f} | "
              f"val loss {val['loss']:.4f} mIoU {val['foreground_miou']:.4f} "
              f"Crack {val['iou'][3]:.3f} | {time.time() - start:.0f}s" + ("  *" if improved else ""))

        if improved:
            best_miou, best_epoch = val["foreground_miou"], epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "decoder": args.decoder,
                "init": args.init,
                "model_name": model.model_name,  # ConvNeXt(timm 이름) 또는 SegFormer(Hugging Face 이름)
                "class_names": cfg.CLASS_NAMES,
                "image_size": cfg.IMAGE_SIZE,
                "dataset": str(cfg.DATASET_DIR),
                "epoch": epoch,
                "val_miou": val["foreground_miou"],
                "args": vars(args),
            }, cfg.MODEL_DIR / f"{run_name}.pth")
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping: {args.patience} epoch 동안 val mIoU 개선 없음")
            break

        (cfg.OUTPUT_DIR / f"{run_name}_history.json").write_text(
            json.dumps(history, indent=2, allow_nan=True), encoding="utf-8")

    print(f"Best epoch {best_epoch}, val mIoU {best_miou:.4f} -> {cfg.MODEL_DIR / f'{run_name}.pth'}")


if __name__ == "__main__":
    main()
