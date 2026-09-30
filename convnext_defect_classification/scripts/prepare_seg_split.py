"""segmentation split CSV와 같은 분할로 ConvNeXt용 ImageFolder 폴더를 만든다.

classification_by_type은 분할이 달라서, YOLO val 이미지 대부분이 ConvNeXt train에 들어가 있다.
두 모델을 합쳐서 평가하려면 같은 이미지를 둘 다 학습 때 보지 않았어야 하므로 분할을 맞춘다.
"""

import csv
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = REPO_ROOT / "data" / "processed" / "segmentation" / "splits"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "classification_seg_split"


def main():
    if OUTPUT_DIR.exists():
        # 이전 결과가 남아 있으면 split이 섞일 수 있어서 새로 만든다
        shutil.rmtree(OUTPUT_DIR)

    for split in ("train", "val", "test"):
        with (SPLIT_DIR / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))

        for row in rows:
            image_path = REPO_ROOT / row["image_path"]
            # classification_by_type과 같은 파일 이름 규칙: MT_Free_exp1_num_1.jpg
            original_class = image_path.parent.parent.name
            destination = OUTPUT_DIR / split / row["defect_type"]
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_path, destination / f"{original_class}_{image_path.name}")

        print(f"{split}: {len(rows)} images")

    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
