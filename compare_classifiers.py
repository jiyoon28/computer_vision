"""Compare the saved six-class ResNet and DINOv2 models on the same test set."""

import csv
import importlib.util
import json
from pathlib import Path
import sys

# NumPy must load before torch in this Anaconda environment (OpenMP conflict).
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "data/processed/classification_by_type/test"
OUTPUT_DIR = ROOT / "outputs/model_comparison"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(model, dataset, device):
    model.to(device).eval()
    predictions = []
    with torch.inference_mode():
        for images, _ in DataLoader(dataset, batch_size=16, shuffle=False):
            logits = model(images.to(device))
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite outputs")
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    indices = list(range(len(dataset.classes)))
    report = classification_report(
        dataset.targets, predictions, labels=indices,
        target_names=dataset.classes, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(dataset.targets, predictions, labels=indices)
    normal = dataset.class_to_idx["Free"]
    true_defect = np.asarray(dataset.targets) != normal
    predicted_defect = np.asarray(predictions) != normal
    return {
        "accuracy": float(np.mean(np.asarray(dataset.targets) == predictions)),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "defect_detection_recall": float(predicted_defect[true_defect].mean()),
        "normal_false_alarm_rate": float(predicted_defect[~true_defect].mean()),
        "report": report,
        "confusion_matrix": matrix.tolist(),
    }, predictions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_dir = ROOT / "resnet_defect_classification"
    dino_dir = ROOT / "dinov2_defect_classification"
    resnet_path = (
        resnet_dir
        / "models"
        / "resnet18_6class_layer4_fc_seed42.pth"
    )
    dino_path = dino_dir / "outputs/dinov2_classifier_best.pth"
    resnet_checkpoint = torch.load(resnet_path, map_location="cpu", weights_only=True)
    dino_checkpoint = torch.load(dino_path, map_location="cpu", weights_only=True)

    resnet_model_module = load_module("resnet_model", resnet_dir / "src/model.py")
    resnet_data_module = load_module("resnet_dataset", resnet_dir / "src/dataset.py")
    # ResNet modules have unique names, so DINOv2's local imports remain separate.
    sys.path.insert(0, str(dino_dir))
    from classifier import DINOClassifier
    from dataset import transform as dino_transform
    from config import MODEL_NAME

    resnet_data = ImageFolder(TEST_DIR, transform=resnet_data_module.transform)
    dino_data = ImageFolder(TEST_DIR, transform=dino_transform)
    classes = resnet_data.classes
    if not (classes == resnet_checkpoint["class_names"] == dino_checkpoint["class_names"]):
        raise ValueError("Checkpoint class order differs from the test dataset")
    if resnet_data.samples != dino_data.samples:
        raise ValueError("Models must use the same test images and labels")
    if dino_checkpoint["backbone_name"] != MODEL_NAME:
        raise ValueError("DINOv2 checkpoint backbone differs from config")

    result = {
        "test_dir": str(TEST_DIR.relative_to(ROOT)),
        "test_size": len(resnet_data),
        "class_names": classes,
        "device": str(device),
        "note": "Each model uses its existing training preprocessing. No retraining.",
        "models": {},
    }
    predictions = {}
    for name, checkpoint, path, dataset in (
        ("ResNet-18", resnet_checkpoint, resnet_path, resnet_data),
        ("DINOv2", dino_checkpoint, dino_path, dino_data),
    ):
        print(f"Evaluating {name} on {len(dataset)} test images...", flush=True)
        if name == "ResNet-18":
            model = resnet_model_module.create_model(num_classes=len(classes))
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = DINOClassifier(num_classes=len(classes))
            model.head.load_state_dict(checkpoint["head_state_dict"])
        metrics, predictions[name] = evaluate(model, dataset, device)
        metrics["checkpoint"] = str(path.relative_to(ROOT))
        metrics["selected_epoch"] = checkpoint.get("epoch")
        metrics["saved_val_accuracy"] = checkpoint.get("val_accuracy")
        result["models"][name] = metrics
        print(json.dumps({k: v for k, v in metrics.items() if k not in ("report", "confusion_matrix")}, indent=2))
        del model

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    with (OUTPUT_DIR / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "true_class", "resnet_prediction", "dinov2_prediction"])
        for i, (path, label) in enumerate(resnet_data.samples):
            writer.writerow([
                str(Path(path).relative_to(ROOT)), classes[label],
                classes[predictions["ResNet-18"][i]], classes[predictions["DINOv2"][i]],
            ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, metrics) in zip(axes, result["models"].items()):
        ConfusionMatrixDisplay(
            np.asarray(metrics["confusion_matrix"]), display_labels=classes,
        ).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
        ax.set_title(f"{name}: accuracy {metrics['accuracy']:.1%}, macro F1 {metrics['macro_f1']:.1%}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    print("Class | count | ResNet F1 / recall | DINOv2 F1 / recall")
    for cls in classes:
        r = result["models"]["ResNet-18"]["report"][cls]
        d = result["models"]["DINOv2"]["report"][cls]
        print(f"{cls}: {int(r['support'])} | {r['f1-score']:.4f} / {r['recall']:.4f} | {d['f1-score']:.4f} / {d['recall']:.4f}")
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
