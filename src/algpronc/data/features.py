"""Frozen-backbone feature extraction + on-disk caching.

This is the cost lever for the whole project: every experiment after this
point is CPU linear algebra over cached `(N, d)` float32 tensors. The one
rule that matters is **a cache hit must never construct a backbone** --
importing `timm` and pulling pretrained weights on every experiment run
would defeat the entire point of caching.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from algpronc.models.backbones import backbone_transform_meta, build_backbone
from algpronc.utils.device import resolve_device
from algpronc.utils.logging import get_logger

_log = get_logger("algpronc.data.features")


@torch.no_grad()
def extract_features(
    dataset: Dataset,
    backbone: torch.nn.Module,
    *,
    device: torch.device | str,
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward pass over `dataset` through `backbone`, batched, no grad.

    Args:
        dataset: yields `(image, label)`.
        backbone: `.eval()`, frozen; moved to `device` here.
        device: torch device (or string) to run the forward pass on.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.

    Returns:
        `(features, labels)`: `features` is `(N, d)` float32 on CPU, `labels`
        is `(N,)` int64 on CPU, in dataset iteration order.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    backbone = backbone.to(device)
    backbone.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    feats_chunks: list[torch.Tensor] = []
    labels_chunks: list[torch.Tensor] = []
    for images, labels in tqdm(loader, desc="extract_features", unit="batch"):
        images = images.to(device, non_blocking=True)
        out = backbone(images)
        feats_chunks.append(out.detach().to("cpu", dtype=torch.float32))
        labels_t = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels)
        labels_chunks.append(labels_t.detach().to("cpu", dtype=torch.int64))

    features = torch.cat(feats_chunks, dim=0) if feats_chunks else torch.empty(0, 0, dtype=torch.float32)
    labels_out = torch.cat(labels_chunks, dim=0) if labels_chunks else torch.empty(0, dtype=torch.int64)
    return features, labels_out


def _cache_payload(dataset_name: str, split: str, backbone_name: str, pretrained: bool, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "split": split,
        "backbone": backbone_name,
        "pretrained": bool(pretrained),
        "image_size": meta["image_size"],
        "resize": meta["resize"],
        "mean": list(meta["mean"]),
        "std": list(meta["std"]),
    }


def cache_key(dataset_name: str, split: str, backbone_name: str, pretrained: bool) -> str:
    """Short, stable cache key: sha256 of (dataset, split, backbone, image size,
    normalisation, pretrained flag), truncated to 16 hex chars.

    Deliberately does NOT depend on batch_size/num_workers/device -- those
    change performance, not the resulting features.
    """
    meta = backbone_transform_meta(backbone_name)
    payload = _cache_payload(dataset_name, split, backbone_name, pretrained, meta)
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _feature_cache_path(data_dir: str, key: str) -> Path:
    return Path(data_dir) / "features" / f"{key}.pt"


def cached_features(
    cfg: Any,
    *,
    train: bool,
    device: torch.device | str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load cached `(features, labels)` for `cfg.data.dataset` / `cfg.model.backbone`,
    extracting (and caching) them on first use.

    `cfg` is an `ExperimentConfig` (or anything exposing `.data`, `.model`,
    `.device`, `.resolved_data_dir()`). On a cache **hit**, this function
    returns before `build_dataset`/`build_backbone` are ever called -- no
    dataset download, no timm import, no weight download.

    Returns:
        `(features, labels)`: `(N, d)` float32, `(N,)` int64.
    """
    data_dir = cfg.resolved_data_dir()
    dataset_name = cfg.data.dataset
    backbone_name = cfg.model.backbone
    pretrained = cfg.model.pretrained
    split = "train" if train else "test"

    key = cache_key(dataset_name, split, backbone_name, pretrained)
    cache_path = _feature_cache_path(data_dir, key)

    if cache_path.exists():
        _log.info(
            f"cached_features: HIT key={key} path={cache_path} "
            f"(dataset={dataset_name} split={split} backbone={backbone_name}) "
            "-- loading from disk, backbone will NOT be constructed"
        )
        blob = torch.load(cache_path, map_location="cpu")
        return blob["features"], blob["labels"]

    _log.info(
        f"cached_features: MISS key={key} path={cache_path} "
        f"(dataset={dataset_name} split={split} backbone={backbone_name}) "
        "-- building dataset + backbone for extraction"
    )

    
    from algpronc.data.datasets import build_dataset

    resolved_device = device if device is not None else resolve_device(getattr(cfg, "device", "auto"))
    resolved_batch_size = batch_size if batch_size is not None else (cfg.data.batch_size if train else cfg.data.eval_batch_size)
    resolved_num_workers = num_workers if num_workers is not None else cfg.data.num_workers

    dataset = build_dataset(dataset_name, data_dir, train, backbone=backbone_name)
    backbone, feature_dim = build_backbone(backbone_name, pretrained=pretrained)

    features, labels = extract_features(
        dataset,
        backbone,
        device=resolved_device,
        batch_size=resolved_batch_size,
        num_workers=resolved_num_workers,
    )

    meta = backbone_transform_meta(backbone_name)
    payload_meta = _cache_payload(dataset_name, split, backbone_name, pretrained, meta)
    payload_meta["feature_dim"] = feature_dim
    payload_meta["n_samples"] = int(features.shape[0])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "labels": labels, "meta": payload_meta}, cache_path)
    _log.info(f"cached_features: wrote {cache_path} shape={tuple(features.shape)}")

    return features, labels
