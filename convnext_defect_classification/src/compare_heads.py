"""Linear / MLP / Cosine head를 같은 이미지에서 비교한다.

세 체크포인트 모두 seg split, 384 강제 리사이즈, seed 42, 같은 하이퍼파라미터로 학습했고 head만 다르다.
"""

# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import f1_score, recall_score

from dataset import build_transforms
from model import model_from_checkpoint
from train import MODEL_DIR, NORMAL_CLASS, OUTPUT_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = REPO_ROOT / "data" / "processed" / "segmentation" / "splits"
# ImageFolder 순서(알파벳). 체크포인트의 class_names와 같은지 load()에서 확인
CLASS_NAMES = ["Blowhole", "Break", "Crack", "Fray", "Free", "Uneven"]

RUNS = {
    "linear": MODEL_DIR / "convnextv2_tiny_384_sqrt_seed42_segsplit.pth",
    "mlp": MODEL_DIR / "convnextv2_tiny_384_sqrt_seed42_segsplit_mlp.pth",
    "cosine": MODEL_DIR / "convnextv2_tiny_384_sqrt_seed42_segsplit_cosine.pth",
}


def load(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError(f"Class order differs: {checkpoint['class_names']}")
    _, transform = build_transforms(checkpoint["image_size"])
    return model_from_checkpoint(checkpoint).to(device).eval(), transform


@torch.no_grad()
def predict(model, transform, image, device):
    """(예측 클래스 번호, 결함 확률 = 1 - P(Free))"""
    probs = model(transform(image).unsqueeze(0).to(device)).softmax(dim=1)[0].cpu().numpy()
    return int(probs.argmax()), float(1 - probs[CLASS_NAMES.index(NORMAL_CLASS)])


def metrics(labels, preds):
    labels, preds = np.asarray(labels), np.asarray(preds)
    normal_id = CLASS_NAMES.index(NORMAL_CLASS)
    ids = list(range(len(CLASS_NAMES)))
    is_defect = labels != normal_id
    return {
        "accuracy": float((labels == preds).mean()),
        "macro_f1": float(f1_score(labels, preds, labels=ids, average="macro", zero_division=0)),
        "defect_recall": float((preds[is_defect] != normal_id).mean()),
        "missed_defects": int((preds[is_defect] == normal_id).sum()),
        "false_alarms": int((preds[~is_defect] != normal_id).sum()),
        "per_class_recall": dict(zip(CLASS_NAMES, recall_score(
            labels, preds, labels=ids, average=None, zero_division=0).round(3).tolist())),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {name: load(path, device) for name, path in RUNS.items()}
    for name, path in RUNS.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        print(f"{name:7s} head_type={checkpoint.get('head_type', 'linear'):7s} best epoch={checkpoint['epoch']}")

    normal_id = CLASS_NAMES.index(NORMAL_CLASS)
    report = {}
    for split in ("val", "test"):
        with (SPLIT_DIR / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        labels = [CLASS_NAMES.index(r["defect_type"]) for r in rows]
        results = {name: [] for name in models}
        for row in rows:
            with Image.open(REPO_ROOT / row["image_path"]) as src:
                image = src.convert("L")
            for name, (model, transform) in models.items():
                results[name].append(predict(model, transform, image, device))

        report[split] = {name: metrics(labels, [p for p, _ in out]) for name, out in results.items()}
        print(f"\n===== {split} ({len(rows)} images) =====")
        print(f"{'':16s}" + "".join(f"{n:>10s}" for n in models))
        for key in ("accuracy", "macro_f1", "defect_recall", "missed_defects", "false_alarms"):
            values = [report[split][n][key] for n in models]
            fmt = "{:10.4f}" if isinstance(values[0], float) else "{:10d}"
            print(f"{key:16s}" + "".join(fmt.format(v) for v in values))
        print("class recall:")
        for cls in CLASS_NAMES:
            print(f"  {cls:9s}" + "".join(f"{report[split][n]['per_class_recall'][cls]:10.3f}" for n in models))

        # 정상/불량 사이 여유: 정상 이미지의 결함확률이 높을수록, 불량 이미지의 결함확률이 낮을수록 위험
        print("margin (정상 최고 결함확률 2·3위 / 맞춘 불량 최저 결함확률):")
        for name, out in results.items():
            free = sorted((p for (_, p), y in zip(out, labels) if y == normal_id), reverse=True)
            caught = sorted(p for (pred, p), y in zip(out, labels) if y != normal_id and pred != normal_id)
            print(f"  {name:7s} {free[1]:.3f}, {free[2]:.3f} / {caught[0]:.3f}")

        wrong = {}
        for row, y, *outs in zip(rows, labels, *results.values()):
            names = [n for n, (pred, _) in zip(models, outs) if pred != y]
            if names:
                wrong[f"{Path(row['image_path']).parent.parent.name}/{Path(row['image_path']).name}"] = names
        report[split]["wrong"] = wrong
        print("틀린 이미지 (어느 head가 틀렸나):")
        for image, names in wrong.items():
            print(f"  {image}: {', '.join(names)}")

    output = OUTPUT_DIR / "compare_heads.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
