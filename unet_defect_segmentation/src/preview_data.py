# dataset.py가 이미지와 ground truth mask 를 제대로 읽는지 학습 전에 눈으로 확인하는 파일
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import SPLIT_DIR
from dataset import MagneticTileDataset

# Train Dataset 생성
dataset=MagneticTileDataset(
    SPLIT_DIR/"train.csv",
    train=False
)

# DataLoader 생성
# batch_size=4 → 이미지 4장을 한 번에 가져옴
# shuffle=True → 랜덤한 이미지 확인
loader=DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# 첫 번째 batch 가져오기
batch=next(iter(loader))

images=batch["image"]
masks=batch["mask"]
defect_types=batch["defect_type"]

# 결과 시각화
plt.figure(figsize=(10,8))

for i in range(len(images)):
    # image shape:
    # 1×256×256
    # squeeze() → 256×256
    image=images[i].squeeze().numpy()

    # dataset.py에서 -1~1로 Normalize했기 때문에
    # 다시 0~1 범위로 복원해서 화면에 표시
    image=image*0.5+0.5

    # Ground Truth Mask
    mask=masks[i].squeeze().numpy()

    # 왼쪽: Original Image
    plt.subplot(len(images),2,i*2+1)
    plt.imshow(image,cmap="gray")
    plt.title(f"Image - {defect_types[i]}")
    plt.axis("off")

    # 오른쪽: Ground Truth Mask
    plt.subplot(len(images),2,i*2+2)
    plt.imshow(mask,cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

plt.tight_layout()
plt.show()