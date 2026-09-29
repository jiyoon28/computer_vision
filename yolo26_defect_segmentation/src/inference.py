"""평가/단일 이미지 추론에서 공통으로 사용하는 함수."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

import config as cfg


def yolo_class():
    try:
        from ultralytics import YOLO
        from ultralytics.models.yolo import semantic  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Install a semantic-capable Ultralytics version:\n"
            "python -m pip install -U -r yolo26_defect_segmentation/requirements.txt"
        ) from exc
    return YOLO


def load_dataset(data_yaml):
    path = Path(data_yaml).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Run prepare_data.py first: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    if not root.is_absolute():
        root = path.parent / root
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if data["names"] != dict(enumerate(cfg.CLASS_NAMES)) or manifest["class_names"] != cfg.CLASS_NAMES:
        raise ValueError("Dataset class order does not match config.py.")
    return root, manifest


def load_model(weights):
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Train the model first, or pass --weights: {weights}")
    model = yolo_class()(str(weights), task="semantic")
    if model.task != "semantic":
        raise ValueError("Expected a YOLO semantic segmentation checkpoint (-sem).")
    if model.names != dict(enumerate(cfg.CLASS_NAMES)):
        raise ValueError("Checkpoint classes differ; use the fine-tuned defect best.pt.")
    return model


def predict_mask(model, image, image_size, device=None):
    kwargs = {} if device is None else {"device": device}
    result = model.predict(source=image, imgsz=image_size, verbose=False,
                           save=False, **kwargs)[0]
    semantic_mask = getattr(result, "semantic_mask", None)
    if semantic_mask is None:
        raise RuntimeError("No semantic_mask output. Check the model and Ultralytics version.")
    values = semantic_mask.data
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if values.shape != (image.height, image.width):
        raise ValueError(f"Unexpected mask shape: {values.shape}")
    if not np.isin(values, np.arange(len(cfg.CLASS_NAMES))).all():
        raise ValueError("Prediction contains unexpected class IDs.")
    return values.astype(np.uint8)


def save_visualization(output_dir, name, image, prediction, target=None):
    output_dir = Path(output_dir)
    for subdir in ("predictions", "visualizations"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    palette = np.asarray(cfg.PALETTE, dtype=np.uint8)
    panels = [np.asarray(image.convert("RGB"))]
    if target is not None:
        panels.append(palette[target])
    panels.append(palette[prediction])
    Image.fromarray(prediction).save(output_dir / "predictions" / name)
    path = output_dir / "visualizations" / name
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)
    return path
