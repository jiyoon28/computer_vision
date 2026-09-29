"""YOLO26 semantic segmentation 실험 설정 (U-Net과 같은 클래스 순서)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SPLIT_DIR = REPO_ROOT / "data/processed/segmentation/splits"
DATASET_DIR = REPO_ROOT / "data/processed/yolo26_semantic"
DATA_YAML = DATASET_DIR / "dataset.yaml"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_PATH = MODEL_DIR / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = PROJECT_ROOT / "runs"

CLASS_NAMES = ["Background", "Blowhole", "Break", "Crack", "Fray", "Uneven"]
DEFECT_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES) if i}
PALETTE = [[0, 0, 0], [255, 80, 80], [80, 200, 80],
           [80, 130, 255], [255, 200, 50], [200, 80, 220]]

# sem = semantic segmentation, seg = instance segmentation.
PRETRAINED_MODEL = "yolo26n-sem.pt"
IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 40
PATIENCE = 8
SEED = 42
