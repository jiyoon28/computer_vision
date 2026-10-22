"""Seg 기반 Clf(cascade)용 크롭 분류기를 학습한다.

입력: 정답 mask의 결함 덩어리 크롭 + 정상 이미지 무작위 크롭
모델/하이퍼파라미터/학습률 스케줄은 convnext_defect_classification/src/train.py와 같다.
"""

# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crops import CLASS_NAMES, REPO_ROOT, CropDataset, build_crops

sys.path.insert(0, str(REPO_ROOT / "convnext_defect_classification" / "src"))
from dataset import build_transforms  # noqa: E402
from model import DEFAULT_MODEL_NAME, create_model, split_param_groups  # noqa: E402
from train import build_scheduler, compute_class_weights, evaluate, set_seed  # noqa: E402

MODEL_DIR = REPO_ROOT / "fusion_inspection" / "models"
OUTPUT_DIR = REPO_ROOT / "fusion_inspection" / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # 크롭은 대부분 원본 기준 100px 안팎이라 384까지 키울 필요가 적음
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="convnext_crop_224_seed42")
    parser.add_argument("--max-batches", type=int, default=None, help="파이프라인 점검용")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / f"{args.name}.pth"
    history_path = OUTPUT_DIR / f"{args.name}_history.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None

    train_transform, eval_transform = build_transforms(args.image_size)
    train_dataset = CropDataset(build_crops("train", args.seed), train_transform)
    val_dataset = CropDataset(build_crops("val", args.seed), eval_transform)
    for name, ds in (("Train", train_dataset), ("Val", val_dataset)):
        counts = Counter(CLASS_NAMES[t] for t in ds.targets)
        print(f"{name} crops: {len(ds)}", dict(sorted(counts.items())))

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_args = {"batch_size": args.batch, "num_workers": args.workers,
                   "pin_memory": device.type == "cuda", "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    _, class_weights = compute_class_weights(train_dataset, "sqrt")
    model = create_model(num_classes=len(CLASS_NAMES), model_name=DEFAULT_MODEL_NAME).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(split_param_groups(model, args.backbone_lr, args.head_lr, args.weight_decay))
    steps = len(train_loader) if args.max_batches is None else min(args.max_batches, len(train_loader))
    scheduler = build_scheduler(optimizer, args.warmup_epochs * steps, args.epochs * steps)

    best_f1, best_loss, waited, history = -1.0, float("inf"), 0, []
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        train_loss, train_total = 0.0, 0
        for i, (images, labels) in enumerate(train_loader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * labels.size(0)
            train_total += labels.size(0)

        val = evaluate(model, val_loader, criterion, device, amp_dtype, CLASS_NAMES, args.max_batches)
        history.append({"epoch": epoch + 1, "train_loss": train_loss / train_total,
                        **{f"val_{k}": v for k, v in val.items()}, "seconds": round(time.time() - start, 1)})
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss {train_loss / train_total:.4f} | "
              f"Val crop Acc {val['accuracy']:.4f} Macro F1 {val['macro_f1']:.4f} ({history[-1]['seconds']}s)")

        if val["macro_f1"] > best_f1 or (val["macro_f1"] == best_f1 and val["loss"] < best_loss):
            best_f1, best_loss, waited = val["macro_f1"], val["loss"], 0
            # convnext model_from_checkpoint()로 그대로 불러올 수 있는 형식
            torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES,
                        "model_name": DEFAULT_MODEL_NAME, "global_pool": "avg", "head_type": "linear",
                        "image_size": args.image_size, "epoch": epoch + 1,
                        "val_macro_f1": val["macro_f1"], "seed": args.seed}, checkpoint_path)
            print("Best model saved.")
        else:
            waited += 1
            if waited >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"Best val crop macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
