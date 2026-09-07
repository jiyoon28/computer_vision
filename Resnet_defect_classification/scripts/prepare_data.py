from pathlib import Path
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# =========================
# 1. 경로 설정
# =========================

# Resnet_defect_classification/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# computer_vision/  (classification / segmentation 공용 데이터 루트)
REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    REPO_ROOT
    / "data"
    / "raw"
    / "magnetic_tile_dataset"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "classification"
)

# =========================
# 2. 클래스 정의
# =========================

NORMAL_FOLDER = "MT_Free"

DEFECT_FOLDERS = [
    "MT_Blowhole",
    "MT_Break",
    "MT_Crack",
    "MT_Fray",
    "MT_Uneven",
]


# =========================
# 3. 이미지 목록 수집
# =========================

samples = []

# Normal
# sorted(): glob()의 반환 순서는 파일시스템마다 달라서, 정렬하지 않으면
# random_state=42 를 줘도 OS가 바뀌면 train/val/test split이 달라진다.
for image_path in sorted(
    (
        RAW_DIR
        / NORMAL_FOLDER
        / "Imgs"
    ).glob("*.jpg")
):
    samples.append(
        (image_path, "normal")
    )
    
# Defect
for folder in DEFECT_FOLDERS:
    for image_path in sorted(
        (
            RAW_DIR
            / folder
            / "Imgs"
        ).glob("*.jpg")
    ):
        
        samples.append(
            (image_path, "defect")
        )
        
print(f"Total images: {len(samples)}")

# =========================
# 4. Train / Val / Test 분할
# =========================

labels = [
    label
    for _, label in samples
]
indices = np.arange(
    len(samples)
)

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    stratify=labels,
    random_state=42
)

temp_labels = [
    labels[i]
    for i in temp_idx
]

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=temp_labels,
    random_state=42
)

splits = {
    "train": train_idx,
    "val": val_idx,
    "test": test_idx,
}


# =========================
# 5. 파일 복사
# =========================

for split_name, split_indices in splits.items():

    for idx in split_indices:

        image_path, label = samples[idx]

        destination = (
            OUTPUT_DIR
            / split_name
            / label
        )

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        original_class = (
            image_path
            .parent
            .parent
            .name
        )

        new_filename = (
            f"{original_class}_{image_path.name}"
        )

        shutil.copy(
            image_path,
            destination / new_filename
        )


print("Dataset preparation complete.")