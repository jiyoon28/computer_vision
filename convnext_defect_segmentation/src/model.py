# ConvNeXt encoder + 교체 가능한 decoder (UNet / FPN / UPerNet)
# 입력 -> Encoder(1/4, 1/8, 1/16, 1/32 feature 4개) -> Decoder -> 입력 크기 logits
# compare_segmenters.py가 이 파일만 따로 불러오므로 다른 로컬 파일은 import하지 않는다

# 현재 Anaconda 환경의 OpenMP 초기화 충돌을 피하도록 NumPy를 먼저 로드.
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

DECODERS = ["unet", "unet4", "fpn", "upernet", "segformer"]
DEFAULT_MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
# ImageNet pretrained backbone이 기대하는 입력 분포로 이미지를 정규화하기 위한 평균과 표준편차
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 공통 Conv 블록 Conv -> BatchNorm -> ReLU
def conv_bn_relu(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

# X를 ref와 동일한 가로, 세로 크기로 키우는 함수
# 새로운 값을 학습해서 만드는게 아니라 주변 픽셀 값들을 이용해서 중간 값을 계산
# 2차원 이미지에서는 가로와 세로 방향을 동시에 고려하는게 bilinear interpolation
def upsample_like(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class ConvNeXtEncoder(nn.Module):
    """timm ConvNeXt에서 stem과 stages만 꺼내 단계별 feature를 반환한다.

    384 입력 기준 (tiny): 96x96x96(1/4), 192x48x48(1/8), 384x24x24(1/16), 768x12x12(1/32)
    stem이 처음부터 1/4로 줄이므로 1/1, 1/2 feature는 없다
    """

    def __init__(self, model_name=DEFAULT_MODEL_NAME, pretrained=True, drop_path_rate=0.1):
        super().__init__()
        # drop_path_rate 0.1: 분류 학습과 같은 값 (데이터가 적어서 켜둠)
        # timm에서 pre-trained weight 불러와서 convnext v2 전체 모델 생성
        backbone = timm.create_model(model_name, pretrained=pretrained, drop_path_rate=drop_path_rate)
        self.stem = backbone.stem # 입력을 처음부터 1/4 해상도로 줄이고 feature로 변환
        self.stages = backbone.stages # feature extraction stage 4개를 불러옴
        self.channels = [info["num_chs"] for info in backbone.feature_info] # 각 stage의 channel 수를 저장함 tiny면 대략 [96,192,384,768]

    def forward(self, x):
        features = [] # 각 stage feature를 저장할 리스트 => 최종적으로 [s4, s8, s16, s32]
        x = self.stem(x) # 입력을 stem에 통과
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features # 이 feature 4개를 decoder에 넘김


# ============================================================
# Decoder 1. UNet: skip connection으로 한 단계씩 키워 원본 해상도까지 복원
# ============================================================

class DoubleConv(nn.Sequential):
    # 기존 UNet(unet_defect_segmentation)의 DoubleConv와 같은 구성
    # 첫 번째 Conv에서 channel을 변경하고, 두 번째 Conv에서 그대로 유지함
    def __init__(self, in_channels, out_channels):
        super().__init__(conv_bn_relu(in_channels, out_channels), conv_bn_relu(out_channels, out_channels))


class UNetDecoder(nn.Module):
    """1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1.

    ConvNeXt에는 1/2, 1/1 feature가 없어서 원본 이미지에서 얕은 conv 가지를 따로 뽑아 skip으로 쓴다.
    얇은 Crack처럼 몇 픽셀짜리 경계를 1/4에서 4배로 키우면 뭉개지기 쉬워서 넣은 것.
    채널 수(256/128/64/48/32)는 기존 UNet처럼 단계마다 절반씩 줄인 경험값.

    full_res=False ("unet4"): 얕은 가지와 1/2, 1/1 단계를 빼고 1/4에서 분류한다 (SegModel이 4배로 키움).
    UNet의 미세한 오탐이 얕은 가지 때문인지 확인하는 비교 실험용. 나머지 구조는 같다.
    """

    def __init__(self, encoder_channels, num_classes, channels=(256, 128, 64, 48, 32), full_res=True):
        super().__init__()
        c0, c1, c2, c3 = encoder_channels # convnext의 c0=1/4, c1=1/8, c2=1/16, c3=1/32 channel수
        d16, d8, d4, d2, d1 = channels # decoder 각 단계에서 사용할 channel 수 
        self.full_res_enabled = full_res
        self.dec16 = DoubleConv(c3 + c2, d16) # 1/32 feature를 1/16으로 키운 뒤 encoder feature와 concatenate하니까 c3+c2 channel이 들어옴
        self.dec8 = DoubleConv(d16 + c1, d8) # 1/16 decoder feature를 1/8로 키우고 s8과 concatenate
        self.dec4 = DoubleConv(d8 + c0, d4) # 1/8 -> 1/4로 키우고 s4와 concatenate
        if full_res:
            self.full_res = DoubleConv(3, d1)                                # 1/1
            self.half_res = nn.Sequential(nn.MaxPool2d(2), DoubleConv(d1, d2))  # 1/2
            self.dec2 = DoubleConv(d4 + d2, d2)
            self.dec1 = DoubleConv(d2 + d1, d1)
        self.classifier = nn.Conv2d(d1 if full_res else d4, num_classes, 1)

    def forward(self, features, image):
        s4, s8, s16, s32 = features # convnext feature 받음
        x = self.dec16(torch.cat([upsample_like(s32, s16), s16], dim=1))
        x = self.dec8(torch.cat([upsample_like(x, s8), s8], dim=1))
        x = self.dec4(torch.cat([upsample_like(x, s4), s4], dim=1))
        if not self.full_res_enabled:
            return self.classifier(x)
        f1 = self.full_res(image) # 원본이미지 -> 1/1 shallow feature
        f2 = self.half_res(f1) # 1/2 feature
        # 1/4 -> 1/2 -> 1/1로 올라가면서 shallow feature 와 concatenate
        x = self.dec2(torch.cat([upsample_like(x, f2), f2], dim=1))
        x = self.dec1(torch.cat([upsample_like(x, f1), f1], dim=1))
        return self.classifier(x) # 원본 해상도 logits 반환


# ============================================================
# Decoder 2. FPN (Panoptic FPN semantic head, Kirillov et al. 2019)
# ============================================================

class FPNDecoder(nn.Module):
    """위에서 아래로 더해 내려오는 pyramid를 만든 뒤, 모든 level을 1/4로 키워 더하고 분류.

    출력은 1/4 해상도이고 SegModel이 4배 bilinear로 키운다.
    256/128 채널, GroupNorm(32)은 논문과 segmentation_models_pytorch FPN의 기본값.
    """

    def __init__(self, encoder_channels, num_classes, pyramid_channels=256, segmentation_channels=128):
        super().__init__()
        # 여러 ConvNeXt stage feature를 top-down 으로 합침
        # 각 stage channel이 96, 192, 384, 768로 다르니까 1x1 Conv로 전부 256 channel로 통일함 => lateral projection
        self.lateral = nn.ModuleList(nn.Conv2d(c, pyramid_channels, 1) for c in encoder_channels)
        # level i(1/4 * 2^i)는 3x3 conv + 2배 업샘플을 i번 거쳐 1/4로 (1/4 level도 conv 1번)
        self.heads = nn.ModuleList() # 각 pyramid level을 1/4 해상도로 올리기 위한 block들을 저장
        for level in range(len(encoder_channels)): # 4개 pyramid level 각각 처리
            blocks, in_channels = [], pyramid_channels
            for _ in range(max(level, 1)):
                blocks.append(nn.Sequential(
                    nn.Conv2d(in_channels, segmentation_channels, 3, padding=1, bias=False),
                    # GroupNorm은 batch size가 작을 때 BatchNorm보다 안정적인 경우가 있음
                    nn.GroupNorm(32, segmentation_channels),
                    nn.ReLU(inplace=True),
                ))
                in_channels = segmentation_channels
            self.heads.append(nn.ModuleList(blocks))
        self.classifier = nn.Conv2d(segmentation_channels, num_classes, 1)

    def forward(self, features, image):
        # top-down: 깊은 level을 키워 바로 위 level에 더함
        # 각 ConvNeXt feature를 256 channel로 변환
        pyramid = [lateral(f) for lateral, f in zip(self.lateral, features)]
        # 깊은 층부터 얕은 층 방향으로 이동
        for i in range(len(pyramid) - 2, -1, -1):
            pyramid[i] = pyramid[i] + F.interpolate(pyramid[i + 1], size=pyramid[i].shape[-2:], mode="nearest")

        target = pyramid[0] # 목표 해상도를 가장 큰 1/4로 설정
        merged = 0
        for x, blocks in zip(pyramid, self.heads):
            for block in blocks:
                x = block(x)
                if x.shape[-2:] != target.shape[-2:]:
                    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            merged = merged + x # 각 pyramid feature를 1/4까지 키워서 전부 합침
        return self.classifier(merged) # 최종적으로 1/4 segmentation logits반환


# ============================================================
# Decoder 3. UPerNet (Xiao et al. 2018). ConvNeXt 논문의 ADE20K 실험에서 쓴 decoder
# ============================================================

# UPerNet에 들어가는 모듈
# PPM = Pyramid Pooling Module
# 가장 깊은 feature에 여러 크기의 global context를 추가함
class PPM(nn.Module):
    """가장 깊은 feature를 1x1, 2x2, 3x3, 6x6으로 평균 풀링해 전역 맥락을 붙인다 (PSPNet)."""

    def __init__(self, in_channels, channels, scales=(1, 2, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList(
            # 입력 크기와 상관없이 지정된 크기로 pooling 
            nn.Sequential(nn.AdaptiveAvgPool2d(s), conv_bn_relu(in_channels, channels, 1)) for s in scales
        )
        # 각 pooled feature channel을 줄이고 가공
        self.bottleneck = conv_bn_relu(in_channels + len(scales) * channels, channels)

    def forward(self, x):
        # 각 scale pooled feature를 다시 원래 x크기로 확대
        pooled = [upsample_like(branch(x), x) for branch in self.branches]
        # 원래 deep feature + 4개 context feature를 channel 방향으로 붙임
        # bottleneck: 붙인 feature을 다시 압축/가공
        return self.bottleneck(torch.cat([x, *pooled], dim=1))

# PPM + FPM + multi-scale concatenation
class UPerNetDecoder(nn.Module):
    """PPM(1/32) + FPN top-down, 네 level을 1/4로 키워 이어붙인 뒤 분류.

    mmsegmentation UPerHead 구조. 원래 채널은 512인데 데이터가 600장 수준이고
    GPU가 12GB라 256으로 줄였다 (경험적 판단, 512와 비교해 본 값은 아님).
    출력은 1/4 해상도이고 SegModel이 4배로 키운다.
    """

    def __init__(self, encoder_channels, num_classes, channels=256, dropout=0.1):
        super().__init__()
        # 가장 깊은 1/32 feature에 PPM 적용
        self.ppm = PPM(encoder_channels[-1], channels) 
        # 1/4, 1/8, 1/16  feature들을 256 channel로 통일
        self.lateral = nn.ModuleList(conv_bn_relu(c, channels, 1) for c in encoder_channels[:-1])
        # 각 scale feature를 Conv로 정리
        self.fpn = nn.ModuleList(conv_bn_relu(channels, channels) for _ in encoder_channels[:-1])
        # 4개  scale feature를 concatenate한뒤 하나로 합칠 convolution
        self.fuse = conv_bn_relu(len(encoder_channels) * channels, channels)
        self.dropout = nn.Dropout2d(dropout)  # 0.1: mmsegmentation 기본값
        # 최종 segmentation head
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features, image):
        # 얕은 3개 Convnext feature를 channel 통일
        laterals = [lateral(f) for lateral, f in zip(self.lateral, features[:-1])]
        laterals.append(self.ppm(features[-1])) # 가장 깊은 feature는 PPM을 통과시킴
        # => 1/4, 1/8, 1/16, 1/32 + Global context를 얻음
        for i in range(len(laterals) - 2, -1, -1):
            # FPN top-down fusion
            laterals[i] = laterals[i] + upsample_like(laterals[i + 1], laterals[i])

        # 각 scale feature를 conv로 다시 정리
        outs = [fpn(x) for fpn, x in zip(self.fpn, laterals[:-1])] + [laterals[-1]] # 모두 1/4 해상도로 통일
        outs = [outs[0]] + [upsample_like(x, outs[0]) for x in outs[1:]]
        return self.classifier(self.dropout(self.fuse(torch.cat(outs, dim=1))))


# ============================================================
# Decoder 4. SegFormer All-MLP decoder (Xie et al. 2021)
# ============================================================

class SegFormerDecoder(nn.Module):
    """네 level을 각각 같은 채널로 맞추고(1x1 = 픽셀별 Linear), 1/4로 키워 이어붙인 뒤 1x1로 합쳐 분류.

    3x3 conv가 하나도 없는 가장 가벼운 decoder. 출력은 1/4 해상도이고 SegModel이 4배로 키운다.
    256 채널, dropout 0.1은 SegFormer B0/B1 설정 (B2 이상은 768). 가벼운 decoder로 충분한지 보려고 작은 쪽을 씀.
    """

    def __init__(self, encoder_channels, num_classes, channels=256, dropout=0.1):
        super().__init__()
        self.linear = nn.ModuleList(nn.Conv2d(c, channels, 1) for c in encoder_channels)
        self.fuse = conv_bn_relu(len(encoder_channels) * channels, channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features, image):
        target = features[0]
        outs = [upsample_like(linear(f), target) for linear, f in zip(self.linear, features)]
        return self.classifier(self.dropout(self.fuse(torch.cat(outs, dim=1))))


DECODER_CLASSES = {
    "unet": UNetDecoder,
    "unet4": lambda channels, num_classes: UNetDecoder(channels, num_classes, full_res=False),
    "fpn": FPNDecoder,
    "upernet": UPerNetDecoder,
    "segformer": SegFormerDecoder,
}


class SegModel(nn.Module):
    """입력: [B, 3, H, W] 0~1 RGB. 출력: [B, num_classes, H, W] logits.

    ImageNet 정규화를 모델 안에서 해서, 학습/평가/비교 스크립트가 to_tensor만 하면 되게 했다.
    """

    def __init__(self, decoder, num_classes, model_name=DEFAULT_MODEL_NAME, pretrained=True):
        super().__init__()
        self.encoder = ConvNeXtEncoder(model_name, pretrained=pretrained)
        self.decoder = DECODER_CLASSES[decoder](self.encoder.channels, num_classes)
        self.model_name = model_name
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        if x.shape[-1] % 32 or x.shape[-2] % 32:
            raise ValueError(f"입력 크기는 32의 배수여야 합니다: {tuple(x.shape[-2:])}")
        x = (x - self.mean) / self.std
        logits = self.decoder(self.encoder(x), x)
        # FPN / UPerNet은 1/4 해상도 출력 -> 입력 크기로
        if logits.shape[-2:] != x.shape[-2:]:
            logits = upsample_like(logits, x)
        return logits


# ============================================================
# SegFormer 전체 모델 (MiT encoder + All-MLP decoder). ConvNeXt encoder와 비교용
# ============================================================

# 이름 -> Hugging Face 사전학습 encoder. mit-b2(약 25M)가 ConvNeXt-T(약 28M)와 크기가 비슷함
SEGFORMER_MODELS = {"segformer_b2": "nvidia/mit-b2"}


class SegFormerModel(nn.Module):
    """transformers의 SegFormer를 SegModel과 같은 입출력으로 감싼다.

    입력: [B, 3, H, W] 0~1 RGB / 출력: [B, num_classes, H, W] logits.
    encoder/decoder 속성을 맞춰 두어 split_param_groups(학습률 분리)를 그대로 쓴다.
    decoder 768 채널: SegFormer 논문의 B2 설정 그대로.
    """

    def __init__(self, hub_name, num_classes, pretrained=True):
        super().__init__()
        # transformers는 이 모델을 쓸 때만 필요하므로 여기서 import
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        kwargs = {"num_labels": num_classes, "decoder_hidden_size": 768}
        if pretrained:
            # mit-b2는 ImageNet 분류 체크포인트라 encoder만 불러오고 decode head는 새로 만든다
            net = SegformerForSemanticSegmentation.from_pretrained(hub_name, **kwargs)
        else:
            net = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(hub_name, **kwargs))
        self.net = net
        self.encoder = net.segformer
        self.decoder = net.decode_head
        self.model_name = hub_name
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        if x.shape[-1] % 32 or x.shape[-2] % 32:
            raise ValueError(f"입력 크기는 32의 배수여야 합니다: {tuple(x.shape[-2:])}")
        logits = self.net(pixel_values=(x - self.mean) / self.std).logits  # 1/4 해상도
        return upsample_like(logits, x)


def create_model(decoder, num_classes, model_name=DEFAULT_MODEL_NAME, pretrained=True, init_checkpoint=None):
    """init_checkpoint: 결함 분류 ConvNeXt 체크포인트. 주면 encoder를 그 가중치로 덮어쓴다.

    decoder가 SEGFORMER_MODELS의 이름이면 ConvNeXt 대신 SegFormer 전체 모델을 만든다 (model_name 무시).
    """
    if decoder in SEGFORMER_MODELS:
        if init_checkpoint is not None:
            raise ValueError("분류 체크포인트 초기화(--init clf)는 ConvNeXt encoder에서만 가능합니다.")
        return SegFormerModel(SEGFORMER_MODELS[decoder], num_classes, pretrained=pretrained)
    if decoder not in DECODERS:
        raise ValueError(f"decoder must be one of {DECODERS + list(SEGFORMER_MODELS)}: {decoder}")
    model = SegModel(decoder, num_classes, model_name, pretrained=pretrained and init_checkpoint is None)
    if init_checkpoint is not None:
        checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["model_name"] != model_name:
            raise ValueError(f"분류 체크포인트 모델이 다릅니다: {checkpoint['model_name']}")
        # 분류 모델의 stem.* / stages.* 이름이 encoder 안의 이름과 같다. head는 버림
        state = {k: v for k, v in checkpoint["model_state_dict"].items() if k.startswith(("stem.", "stages."))}
        model.encoder.load_state_dict(state, strict=True)
    return model


def model_from_checkpoint(checkpoint):
    """학습 체크포인트에 저장된 설정으로 같은 구조를 만들고 가중치를 불러온다."""
    model = create_model(
        decoder=checkpoint["decoder"],
        num_classes=len(checkpoint["class_names"]),
        model_name=checkpoint["model_name"],
        pretrained=False,  # 가중치는 체크포인트에서 불러오므로 다운로드 불필요
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def split_param_groups(model, encoder_lr, decoder_lr, weight_decay):
    """사전학습 encoder는 작은 학습률, 새로 만든 decoder는 큰 학습률.

    bias와 norm 계열(1차원 파라미터)은 weight decay를 적용하지 않음 (분류 쪽과 같은 규칙).
    """
    groups = []
    for part, lr in (("encoder", encoder_lr), ("decoder", decoder_lr)):
        params = [p for p in getattr(model, part).parameters() if p.requires_grad]
        groups.append({"params": [p for p in params if p.ndim > 1], "lr": lr, "weight_decay": weight_decay})
        groups.append({"params": [p for p in params if p.ndim <= 1], "lr": lr, "weight_decay": 0.0})
    return [g for g in groups if g["params"]]


# 모델 Shape Test
if __name__ == "__main__":
    for name in DECODERS:
        model = create_model(name, num_classes=6, pretrained=False).eval()
        with torch.no_grad():
            out = model(torch.rand(2, 3, 384, 384))
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        print(f"{name:8s} out {tuple(out.shape)}  decoder params {decoder_params / 1e6:.2f}M")
