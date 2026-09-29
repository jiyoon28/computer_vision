"""YOLO26-sem 사전학습 가중치를 결함 6개 클래스로 미세조정한다."""

import argparse
import json
from pathlib import Path
import shutil

import config as cfg
from inference import load_dataset, yolo_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=cfg.DATA_YAML)
    parser.add_argument("--model", default=cfg.PRETRAINED_MODEL)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--patience", type=int, default=cfg.PATIENCE)
    parser.add_argument("--batch", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--device", default=None, help="0 for CUDA, cpu for CPU")
    parser.add_argument("--name", default="yolo26n_sem")
    args = parser.parse_args()
    _, manifest = load_dataset(args.data)
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 공식 파일명인 경우 다운로드도 models 폴더에 저장한다.
    source = (cfg.MODEL_DIR / args.model
              if Path(args.model).name == args.model and args.model.endswith(".pt")
              and not Path(args.model).exists() else Path(args.model))
    model = yolo_class()(str(source), task="semantic")
    if model.task != "semantic":
        raise ValueError("Use a semantic model such as yolo26n-sem.pt.")
    kwargs = {} if args.device is None else {"device": args.device}
    model.train(
        data=str(args.data.resolve()), imgsz=manifest["image_size"],
        epochs=args.epochs, batch=args.batch, patience=args.patience,
        seed=cfg.SEED, workers=0, project=str(cfg.RUNS_DIR), name=args.name,
        exist_ok=False, optimizer="Adam", lr0=1e-3,
        # U-Net과 비슷한 증강: 수평/수직 뒤집기만 사용.
        fliplr=0.5, flipud=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        **kwargs,
    )
    best = Path(model.trainer.best)
    if not best.is_file():
        raise FileNotFoundError(f"No best checkpoint was produced: {best}")
    # 각 실험 원본은 runs/에 보존하고, 기본 추론 경로에 최신 성공 결과를 복사.
    shutil.copy2(best, cfg.CHECKPOINT_PATH)
    metadata = {"source_checkpoint": str(best.resolve()),
                "data_yaml": str(args.data.resolve()), "dataset": manifest}
    (cfg.MODEL_DIR / "best_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Best model: {cfg.CHECKPOINT_PATH}")
    print(f"Training run: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
