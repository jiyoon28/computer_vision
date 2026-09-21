import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config as cfg
from dataset import MagneticTileDataset


def main():
    dataset = MagneticTileDataset(
        cfg.SPLIT_DIR / "train.csv",
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))
    palette = np.asarray(cfg.PALETTE, dtype=np.uint8)

    print("Images:", batch["image"].shape)
    print("Masks:", batch["mask"].shape)
    print("Mask dtype:", batch["mask"].dtype)
    print("Mask labels:", batch["mask"].unique().tolist())

    fig, axes = plt.subplots(
        len(batch["image"]), 2, figsize=(9, 10)
    )

    for i in range(len(batch["image"])):
        image = batch["image"][i, 0].numpy() * 0.5 + 0.5
        mask = batch["mask"][i].numpy()

        axes[i, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(batch["defect_type"][i])

        axes[i, 1].imshow(palette[mask])
        axes[i, 1].set_title("Ground Truth")

        axes[i, 0].axis("off")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()