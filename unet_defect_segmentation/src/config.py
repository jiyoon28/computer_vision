from pathlib import Path
import torch


# ============================================================
# 1. 프로젝트 경로 설정
# ============================================================

# 현재 파일:
# unet_defect_segmentation/src/config.py
#
# parents[0] = src/
# parents[1] = unet_defect_segmentation/
# parents[2] = computer_vision/
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# repo root
# classification / segmentation이 data/를 공유한다
REPO_ROOT = Path(__file__).resolve().parents[2]


# 공용 data/ 폴더
DATA_DIR = REPO_ROOT / "data"


# 원본 Magnetic Tile Dataset 위치 (두 프로젝트 공용)
RAW_DATA_DIR = (
    DATA_DIR
    / "raw"
    / "magnetic_tile_dataset"
)


# segmentation 전용 가공 데이터 폴더
PROCESSED_DIR = (
    DATA_DIR
    / "processed"
    / "segmentation"
)


# train.csv / val.csv / test.csv가 저장될 폴더
SPLIT_DIR = PROCESSED_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# 학습된 U-Net weight를 저장할 폴더
MODEL_DIR = PROJECT_ROOT / "models"


# 결과를 저장할 폴더
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# U-Net이 예측한 mask를 저장할 폴더
PREDICTION_DIR = OUTPUT_DIR / "predictions"


# GT vs Prediction 그림을 저장할 폴더
VIS_DIR = OUTPUT_DIR / "visualizations"


# ============================================================
# 2. 필요한 폴더 자동 생성
# ============================================================

# exist_ok=True:
# 폴더가 이미 있어도 오류를 내지 않는다.
for directory in [
    SPLIT_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    PREDICTION_DIR,
    VIS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 3. 이미지 설정
# ============================================================

# 모든 이미지를 256 × 256 크기로 변경해서 학습
IMAGE_SIZE = 256


# ============================================================
# 4. Training Hyperparameters
# ============================================================

# 한 번에 GPU에 넣을 이미지 개수
BATCH_SIZE = 8


# 최대 학습 epoch
NUM_EPOCHS = 40


# Adam optimizer의 learning rate
LEARNING_RATE = 1e-3


# Validation loss가 8 epoch 동안 좋아지지 않으면
# 학습을 조기 종료
EARLY_STOPPING_PATIENCE = 8


# U-Net의 probability를 binary mask로 바꿀 기준
#
# probability >= 0.5 → defect
# probability < 0.5  → background
THRESHOLD = 0.5


# 재현성을 위한 random seed
SEED = 42


# ============================================================
# 5. Device 설정
# ============================================================

# NVIDIA GPU가 사용 가능하면 CUDA
# 그렇지 않으면 CPU 사용
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("====================================")
print("Project Root :", PROJECT_ROOT)
print("Raw Data     :", RAW_DATA_DIR)
print("Device       :", DEVICE)
print("====================================")