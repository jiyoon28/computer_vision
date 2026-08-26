from pathlib import Path
import sys
import torch
import torch.nn.functional as F # PyTorch의 여러 neural network 함수를 F라는 짧은 이름으로 가져옴
from PIL import Image # image 파일을 python에서 다룸
from model import create_model
from dataset import transform

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_best.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = create_model() # .pth 파일은 모델 구조 자체가 아니라 weight만 저장되어 있기 때문에 모델 구조 먼저 만들어야 함
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
) # 저장된 weight 불러오기

model = model.to(device)
model.eval()

# Image
image_path = sys.argv[1] # 터미널에서 입력한 두번째 값을 Python 코드 안으로 전달
image = Image.open(image_path) # 아직 PyTorch Tensor 상태 아님
image_tensor = transform(image) # 학습할 때 사용했던 preprocessing을 그대로 적용
# [3, 244, 244] (channel, height, width) -> [batch, channel, height, width] 맨앞에 bathc 차원 하나 추가 -> [1, 3, 224, 224] (1은 한장이라는 뜻)
image_tensor = image_tensor.unsqueeze(0).to(device) 

# Inference
with torch.no_grad():
    # 모델에 이미지 입력: [1, 3, 244, 244] => 출력 : [1, 2] tensor([[2.3, 0.7]])
    output = model(image_tensor)
    # dimension 0 -> batch, dimension 1 -> classes, softmax(output, dim=1): class방향으로 softmax 계산한다는 뜻
    probabilities = F.softmax(output, dim=1)
    # 가장 큰 확률의 index 찾음  ex) [0.83, 0.17] -> index 0 -> 결과는 tensor([0]) -> .item(): tensoir가 일반 python 숫자로 바뀜
    predicted_class = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, predicted_class].item()
    class_names = ["defect", "normal"]
    
    print(
    f"Prediction: "
    f"{class_names[predicted_class]}"
    )
    print(
    f"Confidence: "
    f"{confidence:.4f}"
    )