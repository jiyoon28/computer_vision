"""Compare ConvNeXt-UNet / FPN / UPerNet and YOLO26-seg on the same test images.

All models were trained on data/processed/yolo26_semantic_384_crackaug (same train images).
Test images = current segmentation test split (excluded.csv applied), read from that folder at 384x384.
Pixel metrics use the YOLO/U-Net formula: one confusion matrix over all pixels, background excluded from the mean.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

# NumPy must load before torch in this Anaconda environment (OpenMP conflict).
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data/processed/yolo26_semantic_384_crackaug"
SPLIT_DIR = ROOT / "data/processed/segmentation/splits"
CONVNEXT_DIR = ROOT / "convnext_defect_segmentation"
YOLO_WEIGHTS = ROOT / "yolo26_defect_segmentation/models/best.pt"
OUTPUT_DIR = ROOT / "outputs/segmentation_comparison"

# YOLO code imports its own `config`; ConvNeXt model.py is loaded by path so the names do not clash.
sys.path.insert(0, str(ROOT / "yolo26_defect_segmentation/src"))
import config as yolo_cfg  # noqa: E402
from inference import load_model, predict_mask  # noqa: E402
from metrics import compute_metrics, update_confusion_matrix  # noqa: E402
from prepare_data import read_split, sample_name  # noqa: E402

CLASS_NAMES = yolo_cfg.CLASS_NAMES
IMAGE_SIZE = 384


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_samples(split):
    samples = []
    for row in read_split(SPLIT_DIR / f"{split}.csv"):
        name = sample_name(row)
        with Image.open(DATASET_DIR / "images" / split / name) as src:
            image = src.convert("RGB")
        with Image.open(DATASET_DIR / "masks" / split / name) as src:
            target = np.asarray(src.convert("L"))
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}: {name}")
        # Image-level label comes from the CSV, not the mask: some defect masks are empty at 384x384
        samples.append((name, image, target, row["is_defect"] == "1"))
    return samples


def convnext_predictor(path, device, seg_model):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError(f"Class order differs: {path}")
    model = seg_model.model_from_checkpoint(checkpoint).to(device).eval()

    @torch.inference_mode()
    def predict(image):
        logits = model(TF.to_tensor(image).unsqueeze(0).to(device))
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    info = {"checkpoint": str(Path(path).resolve().relative_to(ROOT)), "decoder": checkpoint["decoder"],
            "init": checkpoint["init"], "selected_epoch": checkpoint["epoch"],
            "saved_val_miou": checkpoint["val_miou"],
            "params_m": sum(p.numel() for p in model.parameters()) / 1e6}
    return predict, info


def yolo_predictor(path, device):
    model = load_model(path)
    device_arg = 0 if device.type == "cuda" else "cpu"

    def predict(image):
        return predict_mask(model, image, IMAGE_SIZE, device_arg)

    info = {"checkpoint": str(Path(path).resolve().relative_to(ROOT)),
            "params_m": sum(p.numel() for p in model.model.parameters()) / 1e6}
    return predict, info


def evaluate(predict, samples, min_area, cuda):
    count = len(CLASS_NAMES)
    matrix = np.zeros((count, count), dtype=np.int64)
    missed, false_alarms, times = [], [], []
    predict(samples[0][1])  # warm-up (CUDA init / first-call overhead) so it is not timed
    for name, image, target, is_defect in samples:
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = predict(image)
        if cuda:  # GPU runs asynchronously; wait so the timer measures the real work
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        update_confusion_matrix(matrix, prediction, target)
        # Image-level decision as in fusion_inspection: any predicted defect pixels >= min_area -> defect
        flagged = int((prediction > 0).sum()) >= min_area
        if is_defect and not flagged:
            missed.append(name)
        if not is_defect and flagged:
            false_alarms.append(name)

    metrics = compute_metrics(matrix)
    for key in ("iou", "dice"):
        metrics[key] = [v if np.isfinite(v) else None for v in metrics[key]]
    defects = sum(s[3] for s in samples)
    metrics.update({
        "confusion_matrix": matrix.tolist(),
        "defect_images": defects, "normal_images": len(samples) - defects,
        "missed_defects": missed, "false_alarms": false_alarms,
        # Batch 1, model forward + argmax (+ YOLO's own pre/post-processing). Rough, for relative comparison only.
        "ms_per_image": 1000 * float(np.mean(times)),
    })
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convnext", type=Path, nargs="*",
                        default=sorted((CONVNEXT_DIR / "models").glob("convnext_*_seed42.pth")))
    parser.add_argument("--yolo", type=Path, default=YOLO_WEIGHTS)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples(args.split)
    print(f"{args.split}: {len(samples)} images from {DATASET_DIR.relative_to(ROOT)}")

    seg_model = load_module("convnext_seg_model", CONVNEXT_DIR / "src/model.py")
    predictors = {}
    for path in args.convnext:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # seed in the name so runs that differ only by seed do not overwrite each other
        predictors[f"ConvNeXt-{checkpoint['decoder']}({checkpoint['init']}, s{checkpoint['args']['seed']})"] = (
            lambda p=path: convnext_predictor(p, device, seg_model))
    predictors["YOLO26-seg"] = lambda: yolo_predictor(args.yolo, device)

    result = {"split": args.split, "images": len(samples), "dataset": str(DATASET_DIR.relative_to(ROOT)),
              "class_names": CLASS_NAMES, "min_area": args.min_area, "models": {}}
    for name, build in predictors.items():
        predict, info = build()
        result["models"][name] = {**info, **evaluate(predict, samples, args.min_area, device.type == "cuda")}
        del predict
        torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"{args.split}_metrics.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
    print("\nModel                              | fg mIoU  fg Dice | " + " ".join(f"{c[:6]:>6s}" for c in CLASS_NAMES[1:])
          + " | missed  FA | params   ms/img")
    for name, m in result["models"].items():
        print(f"{name:34s} | {m['foreground_miou']:.4f}   {m['foreground_dice']:.4f} | "
              + " ".join(f"{fmt(v):>6s}" for v in m["iou"][1:])
              + f" | {len(m['missed_defects']):4d} {len(m['false_alarms']):4d} | {m['params_m']:5.1f}M {m['ms_per_image']:6.1f}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
