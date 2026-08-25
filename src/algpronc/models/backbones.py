"""Frozen pretrained feature extractors.

Every backbone returned here is `.eval()` with all parameters
`requires_grad_(False)` -- this codebase never fine-tunes the backbone, it
extracts features once (see `algpronc.data.features`) and does all learning
in closed-form ridge heads on top.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_SUPPORTED = ("vit_b16_in21k", "vit_b16_in1k", "resnet50")


def _build_vit(timm_name: str, *, pretrained: bool) -> tuple[nn.Module, int]:
    import timm  

    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
    feature_dim = model.num_features
    return model, feature_dim


def _build_resnet50(*, pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feature_dim


def build_backbone(name: str, *, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build a frozen, eval-mode feature extractor.

    Args:
        name: one of `'vit_b16_in21k'`, `'vit_b16_in1k'`, `'resnet50'`.
        pretrained: load pretrained weights (downloads on first use).

    Returns:
        `(module, feature_dim)`. `module` is `.eval()` and every parameter has
        `requires_grad_(False)`; `module(x)` returns the pooled embedding of
        width `feature_dim` (CLS token for ViT, global-average-pool for
        ResNet -- the classifier head is stripped in both cases).

    Raises:
        ValueError: on an unrecognised `name`, listing the supported names.
    """
    if name == "vit_b16_in21k":
        
        model, feature_dim = _build_vit("vit_base_patch16_224.augreg_in21k", pretrained=pretrained)
    elif name == "vit_b16_in1k":
        
        model, feature_dim = _build_vit("vit_base_patch16_224.augreg_in1k", pretrained=pretrained)
    elif name == "resnet50":
        model, feature_dim = _build_resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"unknown backbone '{name}'; supported backbones: {_SUPPORTED}")

    model.eval()
    model.requires_grad_(False)
    return model, feature_dim


@torch.no_grad()
def backbone_transform_meta(name: str) -> dict:
    """Image-size / normalisation metadata a backbone expects.

    All three supported backbones share the standard 224x224 ImageNet
    pipeline, but this is factored out so `data.datasets` / `data.features`
    can build the cache key and transform from one place instead of
    duplicating magic numbers.
    """
    if name not in _SUPPORTED:
        raise ValueError(f"unknown backbone '{name}'; supported backbones: {_SUPPORTED}")
    return {
        "image_size": 224,
        "resize": 256,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    }
