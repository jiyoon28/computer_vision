import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import OUTPUT_DIR

FEATURE_PATH = OUTPUT_DIR / "dinov2_features.npz"


data = np.load(
    FEATURE_PATH,
    allow_pickle=True
)

features = data["features"]
labels = data["labels"]
paths = data["paths"]


print("Features:", features.shape)
print("Labels:", labels.shape)


# 추출할 때 저장한 클래스 이름을 읽기
class_names = data["class_names"].tolist()
print("Classes:", class_names)

# 비교할 결함 종류 — 이 두 줄을 바꾸면 다른 종류도 비교 가능
CLASS_A = "Crack"
CLASS_B = "Break"

# 이름을 label 번호로 변환
if CLASS_A not in class_names or CLASS_B not in class_names:
    raise ValueError(
        f"비교할 클래스가 없습니다. 사용 가능한 클래스: {class_names}"
    )

class_a_label = class_names.index(CLASS_A)
class_b_label = class_names.index(CLASS_B)

# 각 클래스에 속한 이미지의 인덱스 찾기
class_a_indices = np.where(labels == class_a_label)[0]
class_b_indices = np.where(labels == class_b_label)[0]

print(f"{CLASS_A} images:", len(class_a_indices))
print(f"{CLASS_B} images:", len(class_b_indices))

# A 종류 2장, B 종류 1장이 필요
if len(class_a_indices) < 2 or len(class_b_indices) < 1:
    raise ValueError(
        f"{CLASS_A} 이미지 최소 2장, {CLASS_B} 이미지 최소 1장이 필요합니다."
    )

# 비교할 이미지 3개 선택
a1_idx = class_a_indices[0]
a2_idx = class_a_indices[1]
b1_idx = class_b_indices[0]

# 이미지별 특징 벡터: (384,) → (1, 384)
a1_feature = features[a1_idx].reshape(1, -1)
a2_feature = features[a2_idx].reshape(1, -1)
b1_feature = features[b1_idx].reshape(1, -1)

# 같은 결함 종류끼리 비교
same_similarity = cosine_similarity(
    a1_feature,
    a2_feature,
)[0][0]

# 다른 결함 종류끼리 비교
different_similarity = cosine_similarity(
    a1_feature,
    b1_feature,
)[0][0]

# 결과 출력
print("\n====== Comparison ======")

print(f"\n{CLASS_A} 1:", paths[a1_idx])
print(f"{CLASS_A} 2:", paths[a2_idx])
print(f"{CLASS_B} 1:", paths[b1_idx])

print(
    f"\n{CLASS_A} ↔ {CLASS_A} similarity: "
    f"{same_similarity:.4f}"
)
print(
    f"{CLASS_A} ↔ {CLASS_B} similarity: "
    f"{different_similarity:.4f}"
)