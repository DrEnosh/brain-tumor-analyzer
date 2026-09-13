import torch
from torch import nn
from torchvision import models as tvm

from ml.config import CLASSES


class Classifier(nn.Module):
    # Head must stay pool -> linear: that makes Grad-CAM exactly ReLU(sum_k w_ck * A_k), which the ONNX export relies on.

    def __init__(self, features: nn.Module, feature_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.pool(self.features(x)).flatten(1)))


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


def build_model(arch: str, pretrained: bool = True, num_classes: int = len(CLASSES)) -> Classifier:
    if arch == "baseline_cnn":
        channels = [3, 32, 64, 128, 256, 256]
        features = nn.Sequential(*[_conv_block(a, b) for a, b in zip(channels, channels[1:])])
        return Classifier(features, 256, num_classes, dropout=0.3)
    if arch == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.efficientnet_b0(weights=weights)
        return Classifier(net.features, 1280, num_classes, dropout=0.3)
    if arch == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = tvm.resnet50(weights=weights)
        features = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                 net.layer1, net.layer2, net.layer3, net.layer4)
        return Classifier(features, 2048, num_classes, dropout=0.3)
    raise ValueError(f"unknown architecture: {arch}")


def is_pretrained_backbone(arch: str) -> bool:
    return arch != "baseline_cnn"


def load_checkpoint(path, device="cpu") -> tuple[Classifier, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
