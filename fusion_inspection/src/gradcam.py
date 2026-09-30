"""ConvNeXt가 이미지의 어디를 보고 판단했는지 Grad-CAM으로 확인한다.

저장 그림: 원본 | 정답 mask | Grad-CAM 겹친 그림
숫자: CAM 값 중 정답 결함 영역 안에 들어간 비율 (결함 면적 비율보다 훨씬 크면 결함을 보고 판단한 것)
"""

# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from four_cell_analysis import DEFAULT_CONVNEXT, OUTPUT_DIR, REPO_ROOT, SPLIT_DIR, load_convnext


def grad_cam(model, x, target_class):
    """마지막 stage 출력(feature map)과 그 기울기로 CAM을 만든다."""
    store = {}
    layer = model.stages[-1]
    forward_hook = layer.register_forward_hook(lambda m, i, o: store.update(feature=o))
    backward_hook = layer.register_full_backward_hook(lambda m, gi, go: store.update(grad=go[0]))
    try:
        model.zero_grad()
        logits = model(x)
        logits[0, target_class].backward()
    finally:
        forward_hook.remove()
        backward_hook.remove()

    # 채널별 기울기 평균 = 그 채널이 이 클래스 판단에 얼마나 중요한지
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["feature"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def overlay(gray, cam):
    # 빨강 = 많이 봄, 파랑 = 안 봄. 원본 60% + 색 40%
    heat = np.stack([cam, 1 - np.abs(cam - 0.5) * 2, 1 - cam], axis=-1) * 255
    base = np.stack([gray] * 3, axis=-1).astype(np.float32)
    return (base * 0.6 + heat * 0.4).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--images", nargs="+", required=True,
                        help="파일 이름 일부 (예: exp1_num_339819)")
    parser.add_argument("--convnext-weights", type=Path, default=DEFAULT_CONVNEXT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, class_names = load_convnext(args.convnext_weights, device)
    size = 384

    with (SPLIT_DIR / f"{args.split}.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream)
                if any(key in r["image_path"] for key in args.images)]

    output_dir = OUTPUT_DIR / args.split / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        with Image.open(REPO_ROOT / row["image_path"]) as src:
            gray_image = src.convert("L")
        x = transform(gray_image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model(x).argmax(dim=1))
        cam = grad_cam(model, x, pred)

        # 정답 mask를 모델 입력과 같은 384x384로 맞춤
        gray = np.asarray(gray_image.resize((size, size), Image.Resampling.BILINEAR))
        target = np.zeros((size, size), dtype=bool)
        if row["mask_path"]:
            with Image.open(REPO_ROOT / row["mask_path"]) as src:
                target = np.asarray(src.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 127

        inside = cam[target].sum() / cam.sum() if target.any() else float("nan")
        area = target.mean()
        # 결함 면적 비율 대비 몇 배나 CAM이 결함에 몰렸는지. 1배면 무작위로 본 것과 같음
        print(f"{Path(row['image_path']).name:22s} 정답 {row['defect_type']:8s} 예측 {class_names[pred]:8s} "
              f"CAM 결함영역 비율 {inside:.2f} / 면적 비율 {area:.3f} (x{inside / area:.1f})")

        gt_panel = np.stack([target * 255] * 3, axis=-1).astype(np.uint8)
        panel = np.concatenate([np.stack([gray] * 3, axis=-1), gt_panel, overlay(gray, cam)], axis=1)
        name = f"{Path(row['image_path']).parent.parent.name}_{Path(row['image_path']).stem}.png"
        Image.fromarray(panel).save(output_dir / name)

    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
