# 기존 normal / defect 이미지 읽어오는 파일
# datasets : 이미지 데이터셋 읽어오는 기능
# transforms : 이미지 크기 조정, Tensor 변환, 정규화 같은 전처리 기능
from torchvision import datasets, transforms
from config import IMAGE_SIZE

# Compose: 여러 개의 이미지 전처리 작업을 순서대로 묶어줌
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # PyTorch가 처리할 수 있는 Tensor 형태로 바꿈
    # height x width x channel -> channel x height x width 로 바뀜
    # 픽셀값도 0 ~ 255에서 0.0 ~ 1.0으로 변환됨
    transforms.ToTensor(),
    transforms.Normalize(
        # ImageNet 데이터셋의 평균과 표준편차임
        # ResNet, EfficientNet 같은 ImageNet pretrained 모델 사용할때 흔히 사용됨
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# dataset.ImageFolder를 상속해서 새로운 클래스 만듦
# ImageFolder는 폴더 이름을 class로 자동 인식함
# 예를 들어 알파벳 순서때문에 보통 dataset.class_to_idx
# 결과가 {'defect':0, 'normal':1} 폴더 이름 기준으로 class index 정해짐
class ImageFolderWithPaths(datasets.ImageFolder):
    # 데이터셋에서 특정 이미지 하나 가져올 때 실행되는 함수
    # dataset[0]을 하면 내부적으로 dataset.__getitem__(0)이 실행됨
    def __getitem__(self, index):
        # 부모 크래스인 datasets.ImageFolder의 __getitem__()를 호출함
        # 즉 기본 ImageFolder가 이미지 하나를 읽고 image, label 가져옴
        image, label = super().__getitem__(index)
        # self.smaples 안에 ("dataset/defect/defect001.jpg", 0), .. 있음
        path, _ = self.samples[index]
    
        return image, label, path
