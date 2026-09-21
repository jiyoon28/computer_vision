from pathlib import Path

# 현재 Anaconda 환경의 OpenMP 충돌을 피하기 위한 import 순서
import numpy as np
import torch


# 파일 위치 기준으로 경로 계산
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "magnetic_tile_dataset"

PROCESSED_DIR = DATA_DIR / "processed" / "segmentation"
SPLIT_DIR = PROCESSED_DIR / "splits"

MODEL_DIR = PROJECT_ROOT / "models"

# 기존 이진 segmentation 결과와 구분
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "multiclass"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
VIS_DIR = OUTPUT_DIR / "visualizations"

for directory in (
    SPLIT_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    PREDICTION_DIR,
    VIS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# 픽셀 클래스 순서: 학습·평가·추론에서 동일해야 함
CLASS_NAMES = [
    "Background",
    "Blowhole",
    "Break",
    "Crack",
    "Fray",
    "Uneven",
]

NUM_CLASSES = len(CLASS_NAMES)

# CSV의 defect_type을 픽셀 클래스 번호로 바꾸는 표
DEFECT_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
    if index != 0
}


# 기존 U-Net은 4번 downsampling하므로 16의 배수 사용
IMAGE_SIZE = 256
BATCH_SIZE = 8

NUM_EPOCHS = 40
LEARNING_RATE = 1e-3

# validation의 foreground mIoU가 개선되지 않은 epoch 수
EARLY_STOPPING_PATIENCE = 8
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = MODEL_DIR / "unet_multiclass_best.pth"

# 시각화용 RGB 색상: 클래스 번호 순서와 대응
PALETTE = [
    [0, 0, 0],        # Background
    [255, 80, 80],    # Blowhole
    [80, 200, 80],    # Break
    [80, 130, 255],   # Crack
    [255, 200, 50],   # Fray
    [200, 80, 220],   # Uneven
]