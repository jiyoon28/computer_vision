# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import timm # PyTorch 기반의 이미지 모델 라이브러리

# fcmae_fit_in22k_in1k_384 -> 가중치가 어떤 과정을 거쳤는지 나타냄
# FCMAE 자기지도 사전학습 -> ImageNet-22k -> ImageNet-1k(384 해상도) 순서로 fine-tune된 가중치
# 입력 384로 학습된 가중치라서 이 프로젝트의 입력 크기와 맞음
# 3 x 384 x 384 -> RGB 이미지를 입력으로 사용하는 설정 
DEFAULT_MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"


def create_model(num_classes, model_name=DEFAULT_MODEL_NAME, pretrained=True, drop_path_rate=0.1):
    # num_classes를 넘기면 timm이 기존 ImageNet head(1000개)를 새 Linear로 교체해줌
    # drop_path_rate: 학습 중 residual block을 확률적으로 건너뛰는 정규화. 데이터가 적어서 켜둠
    # 실제 모델 생성
    model = timm.create_model(
        model_name,
        # 대규모 이미지에서 이미 학습된 ConvNext -> Edge/Texture/Shape등 이미 다양한 특징 알고 있음 -> 현재 이미지로 fine-tuning
        pretrained=pretrained,
        num_classes=num_classes,
        # 과적합 줄이기 위한 정규화 설정 -> ConvNext에서는 Residual Connection이 많이 들어가는데 
        # 학습 중 일부 경로를 확률적으로 건너뛰게 만듦
        drop_path_rate=drop_path_rate,
    )
    return model

# Optimizer 설정을 위한 함수 (Fine-Tuning을 어떻게 할 것인지 정하는 함수)
def split_param_groups(model, backbone_lr, head_lr, weight_decay):
    """
    새로 만든 head는 큰 학습률, 사전학습된 backbone은 작은 학습률로 학습.
    bias와 norm 계열(1차원 파라미터)은 weight decay를 적용하지 않는 것이 일반적.
    """
    # model.get_classifier(): ConvNeXt의 마지막 classification head를 가져옴
    # 그리고 그 안의 parameter들을 찾아서 ID를 저장함
    # 즉 "이 parameter들은 head에 속해 있음"이라고 표시해놓는 과정
    head_params = set(id(p) for p in model.get_classifier().parameters())
    # 네 개 그룹 생성 => 두 가지 기준으로 나누기 때문 Backbone vs Head / Weight Decay 적용 vs 미적용
    groups = {
        ("backbone", True): [], ("backbone", False): [],
        ("head", True): [], ("head", False): [],
    }
    # ConvNeXt가 가지고 있는 모든 학습 가능한 parameter를 하나씩 확인함
    # ex) Conv weight, Norm weight, Norm bias, Classifier weight, Classifier bias
    for param in model.parameters():
        # requires_grad=False: 학습하지 않는 parameter
        if not param.requires_grad:
            continue
        # 위에서 저장해둔 head_params와 비교해서 classifier parameter -> yes-> head / no-> backbone으로 구분
        part = "head" if id(param) in head_params else "backbone"
        # conv / linear weight -> 2차원 이상 -> weight decay 적용
        # bias / norm parameter -> 주로 1차원 -> weight dedcay 적용 x
        # weight decay : 과적합 방지를 위한 regularization -> 모델의 weight가 지나치게 커지는 것을 억제함 
        use_decay = param.ndim > 1
        groups[(part, use_decay)].append(param) # 해당 그룹에 pamameter 추가

    lrs = {"backbone": backbone_lr, "head": head_lr}
    return [
        {"params": params, "lr": lrs[part], "weight_decay": weight_decay if use_decay else 0.0}
        for (part, use_decay), params in groups.items()
        if params
    ]


if __name__ == "__main__":
    model = create_model(num_classes=6)
    print(model.default_cfg["input_size"], model.get_classifier())
    print("Parameters:", sum(p.numel() for p in model.parameters()))
