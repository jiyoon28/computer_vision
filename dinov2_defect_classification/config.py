from pathlib import Path
import torch

# dinov2_defect_classification/
PROJECT_ROOT = Path(__file__).resolve().parent

# computer_vision/  (classification / segmentation 공용 데이터 루트)
REPO_ROOT = PROJECT_ROOT.parent

DATA_DIR = REPO_ROOT / "data" / "processed" / "classification_by_type"
TEST_DIR = DATA_DIR / "test"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DINOv2 설정
MODEL_NAME = "dinov2_vits14"
IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(F"Test data: {TEST_DIR}")

# 폴더 역할 : 데이터 위치, 모델 이름, 이미지 크기, batch size, GPU/CPU 한 곳에서 관리
