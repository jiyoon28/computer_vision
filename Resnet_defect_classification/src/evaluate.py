from pathlib import Path  # 파일/폴더 경로를 다루기 위한 클래스
import shutil  # 오분류 이미지를 misclassified 폴더로 복사하기 위한 모듈

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
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_best.pth"  # 학습에서 저장한 best model 경로
OUTPUT_DIR = PROJECT_ROOT / "outputs"  # 평가 결과를 저장할 폴더 경로
OUTPUT_DIR.mkdir(exist_ok=True)  # outputs 폴더가 없으면 생성
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
model = create_model() # 학습 때와 동일한 ResNet-18 구조 생성
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
) # 저장해둔 resnet18_best.pth의 weight를 모델에 불러오기

model = model.to(device) # 모델을 GPU 또는 CPU로 이동
model.eval() # 모델을 평가 모드로 전환

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
        
# Metrics
defect_idx = test_dataset.class_to_idx["defect"] # defect 클래스가 몇 번 label 인지 확인
accuracy = accuracy_score(true_labels, pred_labels) # 전체 데이터 중 맞게 예측한 비율 계산
precision = precision_score(
    true_labels,
    pred_labels,
    pos_label=defect_idx
) # 모델이 defect라고 예측한 것 중 실제 defect인 비율 계산

recall = recall_score(
    true_labels,
    pred_labels,
    pos_label=defect_idx
) # 실제 defect 중 모델이 defect라고 찾아낸 비율 계산

f1 = f1_score(
    true_labels,
    pred_labels,
    pos_label=defect_idx
) # Precision과 Recall의 조화평균

print(f"Accuracy : {accuracy:.4f}")  # Accuracy를 소수점 4자리까지 출력
print(f"Precision: {precision:.4f}")  # Precision 출력
print(f"Recall   : {recall:.4f}")  # Recall 출력
print(f"F1 Score : {f1:.4f}")  # F1 Score 출력

# Confusion Matrix
normal_idx = test_dataset.class_to_idx["normal"]  # normal 클래스가 몇 번 label인지 확인

cm = confusion_matrix(
    true_labels,
    pred_labels,
    labels=[normal_idx, defect_idx]
)  # Normal/Defect 기준으로 Confusion Matrix 생성

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Defect"]
)  # Confusion Matrix에 표시할 클래스 이름 설정

display.plot()  # Confusion Matrix 시각화

plt.title("ResNet-18 Confusion Matrix")  # 그래프 제목 설정

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png"
)  # Confusion Matrix 이미지를 outputs 폴더에 저장

plt.show()  # Confusion Matrix 화면에 출력

# Misclassified images
save_misclassified(model, test_dataset, device, OUTPUT_DIR)  # 오분류 이미지를 outputs/misclassified 폴더에 저장