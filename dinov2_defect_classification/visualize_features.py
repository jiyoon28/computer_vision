import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from config import OUTPUT_DIR


data = np.load(
    OUTPUT_DIR / "dinov2_features.npz",
    allow_pickle=True
)

features = data["features"]
labels = data["labels"]


pca = PCA(n_components=2)

features_2d = pca.fit_transform(
    features
)


class_names = data["classs_names"]


plt.figure(figsize=(8, 6))

for class_idx, class_name in enumerate(class_names):
    mask = labels == class_idx
    
    plt.scatter(
        features_2d[mask, 0],
        features_2d[mask, 1],
        label=str(class_name),
        alpha=0.7,
    )


plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.title(
    "DINOv2 Feature Distribution"
)

plt.legend()

plt.tight_layout()


save_path = (
    OUTPUT_DIR /
    "dinov2_pca.png"
)

plt.savefig(
    save_path,
    dpi=150
)

plt.show()


print(
    f"Saved to: {save_path}"
)