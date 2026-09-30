"""four_cell_analysis.py 결과에서 최종 판정이 틀린 이미지를 원본 | 정답 | YOLO 예측 PNG로 저장한다.

틀린 이미지 = 실제 양/불과 분류기 또는 seg 판정이 다른 이미지.
"""

# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "yolo26_defect_segmentation" / "src"))

from inference import load_model, predict_mask, save_visualization  # noqa: E402
from prepare_data import convert_sample  # noqa: E402
from four_cell_analysis import DEFAULT_YOLO, OUTPUT_DIR, SPLIT_DIR, yolo_image_size  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO)
    args = parser.parse_args()

    result_dir = OUTPUT_DIR / args.split
    with (result_dir / "per_image.csv").open(encoding="utf-8-sig", newline="") as stream:
        results = list(csv.DictReader(stream))
    wrong = [r for r in results
             if r["gt_defect"] != r["cls_ng"] or r["gt_defect"] != r["seg_ng"]]
    if not wrong:
        print("틀린 이미지가 없어요.")
        return

    # 정답 mask를 만들려면 split CSV의 mask_path가 필요하다
    with (SPLIT_DIR / f"{args.split}.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = {r["image_path"]: r for r in csv.DictReader(stream)}

    model = load_model(args.yolo_weights)
    size = yolo_image_size()
    output_dir = result_dir / "errors"
    for r in wrong:
        image, target = convert_sample(rows[r["image_path"]], size)
        prediction = predict_mask(model, image, size)
        # 파일 이름에 정답과 두 모델 판정을 넣어서 폴더만 봐도 알 수 있게 함
        path = Path(r["image_path"])
        name = (f"{path.parent.parent.name}_{path.stem}"
                f"__clf-{r['cls_pred']}_p{float(r['cls_p_defect']):.2f}"
                f"__seg-{r['seg_area']}px.png")
        saved = save_visualization(output_dir, name, image, prediction, np.asarray(target))
        print(saved)


if __name__ == "__main__":
    main()
