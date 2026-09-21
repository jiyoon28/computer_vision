from pathlib import Path  # 파일/폴더 경로를 다루기 위한 클래스
import shutil  # 오분류 이미지를 misclassified 폴더로 복사하기 위한 모듈

# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import torch  # PyTorch 기본 라이브러리
import matplotlib.pyplot as plt  # Confusion Matrix 시각화 및 저장

from sklearn.metrics import (
    accuracy_score,  # Accuracy 계산
    precision_score,  # Precision 계산
    recall_score,  # Recall 계산
    f1_score,  # F1 Score 계산
    confusion_matrix,  # Confusion Matrix 계산
    ConfusionMatrixDisplay  # Confusion Matrix를 그림으로 표시
)

from dataset import get_dataloaders  # dataset.py에서 DataLoader 생성 함수 가져오기
from model import create_model  # model.py에서 ResNet-18 모델 생성 함수 가져오기

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 현재 evaluate.py 기준으로 프로젝트 루트 경로 찾기
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_6class_layer4_fc_seed42.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "layer4_fc_seed42"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def save_misclassified(model, test_dataset, device, output_dir):
    # 정답과 다르게 예측한 이미지를 output_dir/misclassified 폴더에 복사해서 저장
    misclassified_dir = output_dir / "misclassified"

    # 이전 실행에서 남은 이미지가 섞이지 않도록 폴더를 비우고 새로 생성
    if misclassified_dir.exists():
        shutil.rmtree(misclassified_dir)
    misclassified_dir.mkdir(exist_ok=True)

    model.eval()

    with torch.no_grad():
        for index in range(len(test_dataset)):
            image, true_label = test_dataset[index]

            input_tensor = image.unsqueeze(0).to(device)

            output = model(input_tensor)
            predicted = output.argmax(dim=1).item()

            if predicted != true_label:
                original_path = test_dataset.samples[index][0]
                filename = Path(original_path).name

                shutil.copy(
                    original_path,
                    misclassified_dir / filename
                )


(
    train_loader,
    val_loader,
    test_loader,
    train_dataset,
    val_dataset,
    test_dataset
) = get_dataloaders()


# model load
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)

class_names = checkpoint["class_names"]

if class_names != test_dataset.classes:
    raise ValueError("저장된 모델과 test 데이터의 클래스 순서가 다릅니다.")

model = create_model(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Prediction
true_labels = [] # 실제 정답 label을 저장할 리스트
pred_labels = [] # 모델이 예측한 label을 저장할 리스트

with torch.no_grad(): # 평가에서는 gradient가 필요하지 않으므로 계산하지 않음
    for images, labels in test_loader: # test 데이터를 batch 단위로 가져오기
        images = images.to(device) # 입력 이미지를 GPU 또는 CPU로 이동
        outputs = model(images) # Forward: test 이미지를 모델에 넣어 class score(logit) 계산
        _, predicted = torch.max(outputs, dim=1) # 각 이미지에서 가장 큰 class score의 index를 예측 label로 선택
        true_labels.extend(labels.numpy()) # 현재 batch의 실제 label들을 리스트에 추가
        pred_labels.extend(predicted.cpu().numpy()) # GPU의 예측값을 CPU로 옮긴 뒤 리스트에 추가
        
from sklearn.metrics import classification_report

class_indices = list(range(len(class_names)))

print(f"Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")

# 클래스별 Precision / Recall / F1과 평균 출력
print(
    classification_report(
        true_labels,
        pred_labels,
        labels=class_indices,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
)

# 6 × 6 혼동행렬
cm = confusion_matrix(
    true_labels,
    pred_labels,
    labels=class_indices,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

fig, ax = plt.subplots(figsize=(8, 7))
display.plot(ax=ax, xticks_rotation=45, cmap="Blues")

plt.title("ResNet-18: 6-class Classification")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.show()

# 기존 오분류 이미지 저장 함수 재사용
save_misclassified(model, test_dataset, device, OUTPUT_DIR)
