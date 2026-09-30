from pathlib import Path
# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import argparse
import json
import math
import random
import time
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from dataset import DATA_DIR, get_dataloaders
from model import DEFAULT_MODEL_NAME, create_model, split_param_groups

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

NORMAL_CLASS = "Free"  # 정상 클래스 이름. 결함 검출률(defect recall) 계산에 사용


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--class-weight", choices=["none", "sqrt", "inverse"], default="sqrt",
        help="클래스 불균형 보정. sqrt는 역빈도의 제곱근",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="ImageFolder 루트. YOLO와 같은 분할은 data/processed/classification_seg_split",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default=None, help="체크포인트/로그 이름. 기본값은 설정으로 자동 생성")
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="epoch마다 이 개수의 batch만 사용. 파이프라인 점검용",
    )
    return parser.parse_args()


def set_seed(seed):
    # 같은 seed로 비교하면 head 초기값과 데이터 섞는 순서의 차이를 줄일 수 있음
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(dataset, mode):
    # train: Free 666장 vs Fray 22장 (약 30배). 보정 없이 학습하면 Free 쪽으로 쏠림
    counts = np.bincount(dataset.targets, minlength=len(dataset.classes)).astype(np.float64)
    if mode == "none":
        weights = np.ones_like(counts)
    elif mode == "inverse":
        weights = counts.sum() / counts
    else:
        # 역빈도를 그대로 쓰면 Fray 가중치가 Free의 30배가 되어 정상 오탐이 늘기 쉬움
        # 제곱근으로 완화하면 약 5.5배. 경험적으로 많이 쓰는 절충값
        weights = np.sqrt(counts.sum() / counts)
    weights = weights / weights.mean()  # 평균 1로 맞춰 loss 크기를 기존과 비슷하게 유지
    return counts, torch.tensor(weights, dtype=torch.float32)


def build_scheduler(optimizer, warmup_steps, total_steps):
    # warmup 동안 학습률을 0에서 천천히 올리고, 이후 cosine으로 0까지 줄임
    # 새 head가 무작위 상태일 때 큰 gradient가 사전학습 backbone을 망가뜨리는 것을 막음
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype, class_names, max_batches=None):
    model.eval()
    total_loss, total = 0.0, 0
    all_labels, all_preds = [], []
    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        all_labels.append(labels.cpu())
        all_preds.append(outputs.argmax(dim=1).cpu())

    labels = torch.cat(all_labels).numpy()
    preds = torch.cat(all_preds).numpy()
    class_ids = list(range(len(class_names)))
    normal_id = class_names.index(NORMAL_CLASS)
    is_defect = labels != normal_id
    return {
        "loss": total_loss / total,
        "accuracy": float((labels == preds).mean()),
        # Free가 약 70%라 accuracy는 결함을 몇 개 놓쳐도 높게 나옴. 모델 선택은 macro F1로 함
        "macro_f1": float(f1_score(labels, preds, labels=class_ids, average="macro", zero_division=0)),
        # 결함 이미지를 (종류와 상관없이) 결함으로 판정한 비율
        "defect_recall": float((preds[is_defect] != normal_id).mean()) if is_defect.any() else float("nan"),
        "per_class_recall": dict(zip(
            class_names,
            recall_score(labels, preds, labels=class_ids, average=None, zero_division=0).round(3).tolist(),
        )),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_name = args.name or f"convnextv2_tiny_{args.image_size}_{args.class_weight}_seed{args.seed}"
    checkpoint_path = MODEL_DIR / f"{run_name}.pth"
    history_path = OUTPUT_DIR / f"{run_name}_history.json"
    print("Run:", run_name)
    print("Checkpoint:", checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # RTX 30xx(Ampere) 이상은 bf16 지원. fp16과 달리 GradScaler 없이도 안정적
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None
    print(f"Using device: {device}, AMP: {amp_dtype}")

    train_loader, val_loader, _, train_dataset, val_dataset, _ = get_dataloaders(
        image_size=args.image_size, batch_size=args.batch,
        num_workers=args.workers, seed=args.seed, data_dir=args.data_dir,
    )
    print("Data:", args.data_dir)
    class_names = train_dataset.classes
    print("Classes:", train_dataset.class_to_idx)
    print("Train / Val:", len(train_dataset), len(val_dataset))

    counts, class_weights = compute_class_weights(train_dataset, args.class_weight)
    print("Train counts:", dict(zip(class_names, counts.astype(int).tolist())))
    print("Class weights:", dict(zip(class_names, [round(w, 2) for w in class_weights.tolist()])))

    model = create_model(num_classes=len(class_names), model_name=args.model_name).to(device)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        split_param_groups(model, args.backbone_lr, args.head_lr, args.weight_decay),
    )

    steps_per_epoch = len(train_loader) if args.max_batches is None else min(args.max_batches, len(train_loader))
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=args.warmup_epochs * steps_per_epoch,
        total_steps=args.epochs * steps_per_epoch,
    )

    best_f1, best_loss = -1.0, float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        start = time.time()
        # ResNet 실험과 달리 전체 backbone을 학습하므로 BatchNorm 고정 처리가 필요 없음
        # (ConvNeXt는 BatchNorm 대신 LayerNorm을 사용)
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for i, (images, labels) in enumerate(train_loader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            # 전체 fine-tune 초반의 gradient 폭주를 막는 안전장치
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * labels.size(0)
            train_total += labels.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()

        val = evaluate(model, val_loader, criterion, device, amp_dtype, class_names, args.max_batches)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            **{f"val_{k}": v for k, v in val.items()},
            "seconds": round(time.time() - start, 1),
        }
        history.append(record)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Loss: {record['train_loss']:.4f} Train Acc: {record['train_accuracy']:.4f} | "
            f"Val Loss: {val['loss']:.4f} Acc: {val['accuracy']:.4f} "
            f"Macro F1: {val['macro_f1']:.4f} Defect Recall: {val['defect_recall']:.4f} "
            f"({record['seconds']}s)"
        )
        print("  Val recall:", val["per_class_recall"])

        # val이 202장이라 macro F1이 같은 epoch가 자주 나옴. 같으면 val loss가 낮은 쪽을 선택
        improved = val["macro_f1"] > best_f1 or (
            val["macro_f1"] == best_f1 and val["loss"] < best_loss
        )
        if improved:
            best_f1, best_loss = val["macro_f1"], val["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "model_name": args.model_name,
                    "image_size": args.image_size,
                    "epoch": epoch + 1,
                    "val_accuracy": val["accuracy"],
                    "val_macro_f1": val["macro_f1"],
                    "val_defect_recall": val["defect_recall"],
                    "seed": args.seed,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print("Best model saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement: {epochs_without_improvement}/{args.patience}")

        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break

    print(f"Best val macro F1: {best_f1:.4f}")
    print("History:", history_path)


# Windows에서 DataLoader worker(multiprocessing)를 쓰려면 main 가드가 필요함
if __name__ == "__main__":
    main()
