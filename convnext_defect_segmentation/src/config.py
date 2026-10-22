"""ConvNeXt encoder + (UNet / FPN / UPerNet) decoder segmentation 실험 설정."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

# YOLO26 최고 모델과 같은 학습 데이터: 384 리사이즈 + 합성 Crack 60장
# train은 이 폴더를 그대로 쓰고, val/test는 SPLIT_DIR의 현재 CSV에 있는 이미지만 고른다
# (이 폴더는 excluded.csv 반영 전에 만들어져 val/test에 제외 대상이 남아 있음)
DATASET_DIR = REPO_ROOT / "data/processed/yolo26_semantic_384_crackaug"
SPLIT_DIR = REPO_ROOT / "data/processed/segmentation/splits"
UNET_SRC = REPO_ROOT / "unet_defect_segmentation/src"  # CE+Dice loss, 혼동행렬 지표 재사용

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# UNet / YOLO와 같은 클래스 순서. mask PNG의 픽셀 값이 이 번호
CLASS_NAMES = ["Background", "Blowhole", "Break", "Crack", "Fray", "Uneven"]
NUM_CLASSES = len(CLASS_NAMES)

DECODERS = ["unet", "fpn", "upernet"]
DEFAULT_MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
# --init clf: 결함 분류로 fine-tune한 ConvNeXt의 stem/stages를 encoder 초기값으로 사용
CLF_CHECKPOINT = (REPO_ROOT / "convnext_defect_classification/models"
                  / "convnextv2_tiny_384_sqrt_seed42_segsplit.pth")

# 데이터셋이 이미 384로 리사이즈돼 있음. encoder가 1/32까지 줄이므로 32의 배수
IMAGE_SIZE = 384
SEED = 42
