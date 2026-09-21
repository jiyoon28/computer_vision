코드는 안 건드리고, 네가 직접 작성할 수 있게 단계별로 정리해줄게.

먼저 데이터 규모를 확인했는데, 이게 실험 설계에 결정적이야:

split	defect	normal	합계
train	274	666	940
val	59	143	202
test	59	143	202
⚠️ val이 202장뿐이야. 1장 = 0.5%p. 그래서 "옵션 A가 baseline보다 val acc 1%p 높다"는 건 이미지 2장 차이고, 이건 성능 개선이 아니라 seed 노이즈일 확률이 높아. 이걸 대비하지 않으면 4단계 ablation 전체가 무의미해져. Step 0에서 이걸 먼저 잡고 간다.

Step 0. 비교 가능한 실험 기반 만들기 (제일 중요)
head를 바꾸기 전에, **"차이가 진짜인지 판별할 수 있는 상태"**를 먼저 만들어야 해. 여기서 할 일 4가지:

0-1. seed 고정

import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
0-2. early stopping 버그 수정
train.py:100 부근에서, 성능이 개선됐을 때 epochs_without_improvement = 0으로 리셋하는 줄이 빠져 있어. 지금은 개선이 있어도 카운터가 계속 쌓여서, 누적 5번 정체되면 학습이 끊겨. 실험마다 학습 길이가 제멋대로 달라지니 비교가 불가능해져. 반드시 먼저 고쳐.

0-3. val_loader의 shuffle=True → False
dataset.py:44. 결과값은 같지만 재현성 확보용.

0-4. 선택 지표를 accuracy → defect F1로
normal이 71%라서, "전부 normal로 찍기"만 해도 accuracy 0.708이 나와. accuracy로 best model을 고르면 defect를 못 잡는 모델이 선택될 수 있어. 4단계 내내 동일한 지표로 골라야 비교가 성립해.


from sklearn.metrics import precision_score, recall_score, f1_score

@torch.no_grad()
def evaluate(model, loader, device, defect_idx):
    model.eval()
    true_labels, pred_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        true_labels.extend(labels.tolist())
        pred_labels.extend(preds.cpu().tolist())

    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    kw = dict(pos_label=defect_idx, zero_division=0)
    return {
        "accuracy":         correct / len(true_labels),
        "defect_precision": precision_score(true_labels, pred_labels, **kw),
        "defect_recall":    recall_score(true_labels, pred_labels, **kw),
        "defect_f1":        f1_score(true_labels, pred_labels, **kw),
    }
defect_idx는 하드코딩하지 말고 train_dataset.class_to_idx["defect"]로 가져와. (ImageFolder가 알파벳순 정렬해서 defect=0, normal=1인데, 폴더명이 바뀌면 뒤집혀.)

0-5. 각 실험을 3 seed로 돌리고 평균 ± 표준편차로 비교
이게 val 202장 문제의 유일한 해결책이야. seed 42/43/44로 돌려서, 차이가 표준편차보다 작으면 "개선 없음"으로 판정해.

Step 1. model.py — head를 갈아끼울 수 있게 리팩터링
4단계를 한 파일로 다 커버하는 구조. 이걸 먼저 만들어두면 이후 단계는 인자만 바꾸면 돼.


import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ConcatPool2d(nn.Module):
    """GAP(평균)와 GMP(최댓값)를 채널 방향으로 이어붙여 출력 채널을 2배로 만든다."""

    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # (B, 512, 7, 7) -> (B, 1024, 1, 1)
        return torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)


def build_head(in_features, num_classes, head_type):
    # Step 2의 baseline
    if head_type == "linear":
        return nn.Linear(in_features, num_classes)

    # 옵션 A
    if head_type == "dropout_linear":
        return nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    # 옵션 B
    if head_type == "mlp":
        return nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256, bias=False),  # 뒤에 BN이 있으므로 bias 불필요
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    raise ValueError(f"Unknown head_type: {head_type}")


def create_model(
    num_classes=2,
    freeze_backbone=True,
    head_type="linear",
    concat_pool=False,
    unfreeze_layer4=False,
):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # 주의: freeze 루프는 반드시 새 head를 붙이기 "전"에 돌아야 한다.
    # 순서가 바뀌면 새로 만든 head까지 얼어버려서 아무것도 학습되지 않는다.
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features  # ResNet-18 -> 512

    if concat_pool:
        model.avgpool = ConcatPool2d()
        in_features = in_features * 2   # 512 -> 1024

    model.fc = build_head(in_features, num_classes, head_type)

    return model


def freeze_backbone_bn(model):
    """backbone을 얼렸을 때 BatchNorm의 running stats까지 고정한다.
    model.train() 호출 "뒤"에 매 epoch 불러야 한다."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
freeze_backbone_bn이 왜 필요한지 — 이건 놓치기 쉬운 함정이야. requires_grad=False는 gradient만 막지, model.train() 상태에서 BatchNorm의 running_mean/running_var가 자성 타일 데이터 분포로 갱신되는 건 못 막아. 즉 지금 네 baseline은 "완전히 freeze된 backbone"이 아니라 "BN 통계만 슬금슬금 움직이는 backbone"이야. 이게 seed마다 결과를 흔드는 원인 중 하나고, 4단계 비교를 오염시켜. 켜든 끄든 모든 실험에서 동일하게 유지하는 게 핵심이야.

Step 2. train.py — 인자로 실험을 바꾸는 스크립트

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import get_dataloaders
from model import create_model, freeze_backbone_bn
# set_seed, evaluate 는 Step 0에서 만든 것 (utils.py로 빼도 좋음)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "outputs" / "experiments"
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="baseline")           # 실험 이름 (파일명에 사용)
    p.add_argument("--head", default="linear",
                   choices=["linear", "dropout_linear", "mlp"])
    p.add_argument("--concat-pool", action="store_true")
    p.add_argument("--unfreeze-layer4", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)      # head의 lr
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=7)
    return p.parse_args()


def build_optimizer(model, args):
    if args.unfreeze_layer4:
        # discriminative LR: 사전학습된 backbone은 낮게, 새 head는 높게
        return Adam(
            [
                {"params": model.layer4.parameters(), "lr": args.backbone_lr},
                {"params": model.fc.parameters(),     "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )

    # concat_pool 등으로 학습 대상이 바뀌어도 안전하게 잡아준다.
    # (model.fc.parameters() 로 하드코딩하면 나중에 놓치는 파라미터가 생긴다)
    trainable = [p for p in model.parameters() if p.requires_grad]
    return Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.exp} / seed {args.seed}] device: {device}")

    train_loader, val_loader, _, train_dataset, _, _ = get_dataloaders(batch_size=32)
    defect_idx = train_dataset.class_to_idx["defect"]

    model = create_model(
        num_classes=2,
        freeze_backbone=True,
        head_type=args.head,
        concat_pool=args.concat_pool,
        unfreeze_layer4=args.unfreeze_layer4,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args)

    best_score = 0.0
    best_metrics = None
    epochs_without_improvement = 0
    ckpt_path = MODEL_DIR / f"{args.exp}_seed{args.seed}.pth"

    for epoch in range(args.epochs):
        model.train()
        if not args.unfreeze_layer4:
            freeze_backbone_bn(model)   # train() 뒤에 호출해야 효과가 있다

        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val = evaluate(model, val_loader, device, defect_idx)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Loss: {train_loss / len(train_loader):.4f} "
            f"Val Acc: {val['accuracy']:.4f} "
            f"Defect P/R/F1: {val['defect_precision']:.4f}/"
            f"{val['defect_recall']:.4f}/{val['defect_f1']:.4f}"
        )

        # best model 기준 = defect_f1 (4단계 내내 동일하게 유지할 것)
        if val["defect_f1"] > best_score:
            best_score = val["defect_f1"]
            best_metrics = {**val, "epoch": epoch + 1}
            epochs_without_improvement = 0      # <- Step 0-2에서 고친 부분
            torch.save(model.state_dict(), ckpt_path)
            print("  Best model saved.")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement: {epochs_without_improvement}/{args.patience}")

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break

    result = {"exp": args.exp, "seed": args.seed, "config": vars(args), **best_metrics}
    with open(RESULT_DIR / f"{args.exp}_seed{args.seed}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("Best:", best_metrics)


if __name__ == "__main__":
    main()
Step 3. 4단계 실험 실행
각 실험을 seed 3개씩 돌려. PowerShell 기준:


# 1) baseline — 현재 코드와 동일한 1-layer linear head
foreach ($s in 42,43,44) { python src/train.py --exp baseline --head linear --seed $s }

# 2) 옵션 A — Dropout(0.3) + Linear
foreach ($s in 42,43,44) { python src/train.py --exp dropout --head dropout_linear --seed $s }

# 3) 옵션 C — GAP+GMP concat (A의 head를 그대로 쓰고 pooling만 교체)
foreach ($s in 42,43,44) { python src/train.py --exp concat --head dropout_linear --concat-pool --seed $s }

# 4) 옵션 B — MLP head (여기서 떨어지면 데이터 부족 확정)
foreach ($s in 42,43,44) { python src/train.py --exp mlp --head mlp --lr 3e-4 --weight-decay 1e-4 --seed $s }
한 번에 한 가지만 바꾸는 게 원칙이야. 위에서 3)이 --head dropout_linear를 유지하는 이유가 그거 — pooling 효과만 분리해서 보려는 것. 4)에서 lr을 3e-4로 낮추는 건 layer가 깊어졌기 때문이고, 이건 변수 2개를 동시에 바꾸는 셈이니 결과가 애매하면 --lr 1e-3으로도 한 번 돌려봐.

결과 취합

import json, statistics
from pathlib import Path

RESULT_DIR = Path("outputs/experiments")

for exp in ["baseline", "dropout", "concat", "mlp"]:
    files = sorted(RESULT_DIR.glob(f"{exp}_seed*.json"))
    if not files:
        continue
    runs = [json.load(open(f)) for f in files]
    for key in ["accuracy", "defect_recall", "defect_f1"]:
        vals = [r[key] for r in runs]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{exp:10s} {key:16s} {mean:.4f} ± {std:.4f}")
    print()
판정 기준: baseline 대비 평균이 올랐어도, 그 차이가 **표준편차보다 작으면 "개선 없음"**으로 처리하고 다음 단계로 넘어가. 202장짜리 val에서 이 규칙 없이 판단하면 노이즈를 쫓게 돼.

Step 4. layer4 unfreeze (여기서 진짜 성능이 나온다)
3단계까지에서 가장 좋았던 head 구성을 골라서 그대로 쓰고, backbone만 푼다.


# 예: concat 구성이 이겼다고 가정
foreach ($s in 42,43,44) {
  python src/train.py --exp concat_ft --head dropout_linear --concat-pool `
    --unfreeze-layer4 --lr 1e-3 --backbone-lr 1e-4 --weight-decay 1e-4 --seed $s
}
주의할 점:

backbone-lr은 head lr의 1/10 수준으로. 같게 두면 사전학습 feature가 초반 큰 gradient에 망가져서 오히려 baseline보다 떨어져.
train 940장에 layer4(약 800만 파라미터)를 푸는 거라 과적합이 빠르게 온다. patience를 5 정도로 줄이고 epoch별 val 곡선을 꼭 봐.
더 안정적인 방법은 2단계 학습: head만 10 epoch 학습 → 그 weight를 로드해서 layer4를 풀고 낮은 lr로 이어서 학습. 처음부터 같이 푸는 것보다 거의 항상 나아.
freeze_backbone_bn을 끄게 되니(위 코드에서 --unfreeze-layer4면 자동으로 안 부름), BN 통계가 이제 자성 타일 분포로 학습돼. 이건 fine-tuning에서는 의도한 동작이 맞아.
마지막 두 가지
1. test set은 4단계 내내 열지 마. 지금 evaluate.py가 test로 평가하는데, 실험할 때마다 test를 보면 202장에 과적합돼서 최종 숫자가 부풀려져. 모든 비교는 val로만 하고, 최종 1개 구성을 정한 뒤 test는 딱 한 번 돌려.

2. Step 4까지 다 해도 기대만큼 안 오르면, 원인은 head가 아니라 augmentation일 가능성이 높아. dataset.py:19-35에서 train과 val이 같은 transform을 공유하고 있어서 augmentation이 0인 상태야. train 940장에 backbone까지 풀면 augmentation 없이는 반드시 과적합해. RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(15), ColorJitter(brightness=0.2, contrast=0.2) 정도가 자성 타일 결함에 안전한 조합이야(결함의 방향성이 의미 없는 도메인이라 flip/rotation을 써도 label이 안 깨져). 이걸 Step 0에 넣을지 Step 5로 뺄지는 네 선택인데, 넣는다면 Step 0에 넣어서 4단계 전체에 동일하게 적용해야 비교가 유지돼.