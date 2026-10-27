"""Model definitions. All models output ONE logit (PNEUMONIA vs NORMAL)."""
import logging

import torch
from torch import nn
from torchvision import models as tv_models

log = logging.getLogger("cxr")

MODEL_NAMES = ("baseline", "densenet121")


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class BaselineCNN(nn.Module):
    """Small 4-block CNN: conv-BN-ReLU x2 + pool per block, global average pooling, dropout."""

    def __init__(self, dropout=0.3, widths=(16, 32, 64, 128)):
        super().__init__()
        chans = (3,) + tuple(widths)
        self.features = nn.Sequential(*[_block(a, b) for a, b in zip(chans[:-1], chans[1:])])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[-1], 1))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_densenet121(dropout=0.3, pretrained=False):
    if pretrained:
        log.warning("pretrained=True: torchvision will DOWNLOAD DenseNet121 ImageNet weights "
                    "(~30 MB) into the torch cache if not already cached.")
        model = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.DEFAULT)
    else:
        model = tv_models.densenet121(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier.in_features, 1))
    return model


def freeze_backbone(model):
    """Train only the classifier head (DenseNet121). BN layers in the backbone stay in eval mode."""
    for p in model.features.parameters():
        p.requires_grad = False
    model._frozen_backbone = True


def set_train_mode(model):
    model.train()
    if getattr(model, "_frozen_backbone", False):
        model.features.eval()


def build_model(name, dropout=0.3, pretrained=False, freeze=False):
    if name == "baseline":
        if pretrained or freeze:
            raise ValueError("--pretrained/--freeze-backbone apply to densenet121 only")
        return BaselineCNN(dropout=dropout)
    if name == "densenet121":
        model = build_densenet121(dropout=dropout, pretrained=pretrained)
        if freeze:
            freeze_backbone(model)
        return model
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
