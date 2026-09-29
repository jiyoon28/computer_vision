# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import timm

# FCMAE 자기지도 사전학습 -> ImageNet-22k -> ImageNet-1k(384 해상도) 순서로 fine-tune된 가중치
# 입력 384로 학습된 가중치라서 이 프로젝트의 입력 크기와 맞음
DEFAULT_MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"


def create_model(num_classes, model_name=DEFAULT_MODEL_NAME, pretrained=True, drop_path_rate=0.1):
    # num_classes를 넘기면 timm이 기존 ImageNet head(1000개)를 새 Linear로 교체해줌
    # drop_path_rate: 학습 중 residual block을 확률적으로 건너뛰는 정규화. 데이터가 적어서 켜둠
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    return model


def split_param_groups(model, backbone_lr, head_lr, weight_decay):
    """
    새로 만든 head는 큰 학습률, 사전학습된 backbone은 작은 학습률로 학습.
    bias와 norm 계열(1차원 파라미터)은 weight decay를 적용하지 않는 것이 일반적.
    """
    head_params = set(id(p) for p in model.get_classifier().parameters())
    groups = {
        ("backbone", True): [], ("backbone", False): [],
        ("head", True): [], ("head", False): [],
    }
    for param in model.parameters():
        if not param.requires_grad:
            continue
        part = "head" if id(param) in head_params else "backbone"
        use_decay = param.ndim > 1
        groups[(part, use_decay)].append(param)

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
