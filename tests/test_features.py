"""Tests for algpronc.data.features and algpronc.models.projection.

No test in this file downloads anything: the feature-cache round trip uses a
tiny in-memory fake dataset and a stub 2-layer `nn.Module` "backbone" instead
of a real timm/torchvision model, and `build_dataset`/`build_backbone` are
monkeypatched out on the (only) cache-miss path so nothing ever reaches the
network. A couple of genuinely download-requiring smoke tests are included
at the bottom, gated behind `@pytest.mark.skipif` on an environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from algpronc.data.features import cache_key, cached_features, extract_features
from algpronc.models.projection import RandomProjection, build_projection


class _FakeImageDataset(Dataset):
    """Tiny deterministic stand-in for a real image dataset: fixed-seed random
    "images" of shape (3, 8, 8) and cyclic int labels. No I/O whatsoever."""

    def __init__(self, n: int = 17, n_classes: int = 4, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, 8, 8, generator=g)
        self.labels = [i % n_classes for i in range(n)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


class _StubBackbone(nn.Module):
    """A stub 2-layer 'backbone': flatten -> linear -> relu -> linear."""

    def __init__(self, in_hw: int = 8, out_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * in_hw * in_hw, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class _StubDataCfg:
    dataset: str = "fake"
    batch_size: int = 4
    eval_batch_size: int = 4
    num_workers: int = 0


@dataclass
class _StubModelCfg:
    backbone: str = "resnet50"  
    pretrained: bool = False


@dataclass
class _StubCfg:
    data_dir: str
    device: str = "cpu"
    data: _StubDataCfg = field(default_factory=_StubDataCfg)
    model: _StubModelCfg = field(default_factory=_StubModelCfg)

    def resolved_data_dir(self) -> str:
        return self.data_dir


def test_extract_features_shapes_and_dtypes():
    dataset = _FakeImageDataset(n=17, n_classes=4)
    backbone = _StubBackbone(out_dim=6)
    features, labels = extract_features(dataset, backbone, device="cpu", batch_size=4, num_workers=0)
    assert features.shape == (17, 6)
    assert features.dtype == torch.float32
    assert labels.shape == (17,)
    assert labels.dtype == torch.int64
    assert labels.tolist() == [i % 4 for i in range(17)]


def test_extract_features_matches_direct_forward_pass():
    dataset = _FakeImageDataset(n=9, n_classes=3, seed=1)
    backbone = _StubBackbone(out_dim=5)
    features, _ = extract_features(dataset, backbone, device="cpu", batch_size=4, num_workers=0)
    with torch.no_grad():
        expected = backbone(dataset.images)
    assert torch.allclose(features, expected, atol=1e-6)


def test_extract_features_is_batch_size_invariant():
    dataset = _FakeImageDataset(n=13, n_classes=4, seed=2)
    backbone = _StubBackbone(out_dim=4)
    f1, l1 = extract_features(dataset, backbone, device="cpu", batch_size=1, num_workers=0)
    f2, l2 = extract_features(dataset, backbone, device="cpu", batch_size=5, num_workers=0)
    assert torch.allclose(f1, f2, atol=1e-6)
    assert torch.equal(l1, l2)


def test_cache_key_is_short_and_deterministic():
    k1 = cache_key("cifar100", "train", "resnet50", True)
    k2 = cache_key("cifar100", "train", "resnet50", True)
    assert k1 == k2
    assert isinstance(k1, str)
    assert len(k1) == 16


def test_cache_key_varies_with_each_component():
    base = cache_key("cifar100", "train", "resnet50", True)
    assert base != cache_key("cifar100", "test", "resnet50", True)
    assert base != cache_key("tinyimagenet", "train", "resnet50", True)
    assert base != cache_key("cifar100", "train", "vit_b16_in21k", True)
    assert base != cache_key("cifar100", "train", "resnet50", False)


def test_cache_key_unknown_backbone_raises():
    with pytest.raises(ValueError):
        cache_key("cifar100", "train", "not_a_real_backbone", True)


def test_cached_features_round_trip(tmp_path, monkeypatch):
    dataset = _FakeImageDataset(n=11, n_classes=3, seed=3)
    backbone = _StubBackbone(out_dim=7)

    def fake_build_dataset(name, root, train, **kwargs):
        return dataset

    def fake_build_backbone(name, *, pretrained=True):
        return backbone, backbone.out_dim

    monkeypatch.setattr("algpronc.data.datasets.build_dataset", fake_build_dataset)
    monkeypatch.setattr("algpronc.data.features.build_backbone", fake_build_backbone)

    cfg = _StubCfg(data_dir=str(tmp_path))

    
    features1, labels1 = cached_features(cfg, train=True)
    assert features1.shape == (11, 7)
    assert features1.dtype == torch.float32
    assert labels1.dtype == torch.int64

    key = cache_key(cfg.data.dataset, "train", cfg.model.backbone, cfg.model.pretrained)
    cache_path = tmp_path / "features" / f"{key}.pt"
    assert cache_path.exists()

    blob = torch.load(cache_path, map_location="cpu")
    assert set(blob.keys()) == {"features", "labels", "meta"}
    assert blob["features"].shape == (11, 7)
    assert blob["labels"].shape == (11,)
    assert blob["meta"]["dataset"] == "fake"
    assert blob["meta"]["split"] == "train"
    assert blob["meta"]["feature_dim"] == 7

    
    def boom_backbone(name, *, pretrained=True):
        raise AssertionError("build_backbone must not be called on a cache hit")

    def boom_dataset(name, root, train, **kwargs):
        raise AssertionError("build_dataset must not be called on a cache hit")

    monkeypatch.setattr("algpronc.data.features.build_backbone", boom_backbone)
    monkeypatch.setattr("algpronc.data.datasets.build_dataset", boom_dataset)

    features2, labels2 = cached_features(cfg, train=True)
    assert torch.equal(features1, features2)
    assert torch.equal(labels1, labels2)


def test_cached_features_train_and_test_splits_are_independent(tmp_path, monkeypatch):
    train_ds = _FakeImageDataset(n=6, n_classes=2, seed=10)
    test_ds = _FakeImageDataset(n=4, n_classes=2, seed=20)
    backbone = _StubBackbone(out_dim=3)

    def fake_build_dataset(name, root, train, **kwargs):
        return train_ds if train else test_ds

    def fake_build_backbone(name, *, pretrained=True):
        return backbone, backbone.out_dim

    monkeypatch.setattr("algpronc.data.datasets.build_dataset", fake_build_dataset)
    monkeypatch.setattr("algpronc.data.features.build_backbone", fake_build_backbone)

    cfg = _StubCfg(data_dir=str(tmp_path))
    train_features, _ = cached_features(cfg, train=True)
    test_features, _ = cached_features(cfg, train=False)

    assert train_features.shape == (6, 3)
    assert test_features.shape == (4, 3)
    assert not torch.allclose(train_features.mean(), test_features.mean())

    key_train = cache_key("fake", "train", "resnet50", False)
    key_test = cache_key("fake", "test", "resnet50", False)
    assert key_train != key_test
    assert (tmp_path / "features" / f"{key_train}.pt").exists()
    assert (tmp_path / "features" / f"{key_test}.pt").exists()


def test_build_projection_disabled_returns_none_and_input_dim():
    cfg_rp = SimpleNamespace(enabled=False, proj_dim=999, activation="relu", seed=0)
    proj, out_dim = build_projection(cfg_rp, d_in=768)
    assert proj is None
    assert out_dim == 768


def test_build_projection_enabled_returns_callable_and_proj_dim():
    cfg_rp = SimpleNamespace(enabled=True, proj_dim=100, activation="relu", seed=0)
    proj, out_dim = build_projection(cfg_rp, d_in=32)
    assert out_dim == 100
    x = torch.randn(5, 32, dtype=torch.float64)
    y = proj(x)
    assert y.shape == (5, 100)


def test_random_projection_is_pure_function_of_seed():
    p1 = RandomProjection(d_in=16, proj_dim=32, seed=42, activation="relu")
    p2 = RandomProjection(d_in=16, proj_dim=32, seed=42, activation="relu")
    assert torch.equal(p1.Wr, p2.Wr)
    x = torch.randn(3, 16, dtype=torch.float64)
    assert torch.equal(p1(x), p2(x))


def test_random_projection_different_seeds_differ():
    p1 = RandomProjection(d_in=16, proj_dim=32, seed=1, activation="relu")
    p2 = RandomProjection(d_in=16, proj_dim=32, seed=2, activation="relu")
    assert not torch.equal(p1.Wr, p2.Wr)


def test_random_projection_variance_matches_1_over_d_in():
    d_in = 2000
    p = RandomProjection(d_in=d_in, proj_dim=1, seed=0, activation="relu")
    
    
    empirical_var = p.Wr[:, 0].var().item()
    assert empirical_var == pytest.approx(1.0 / d_in, rel=0.25)


def test_random_projection_relu_is_nonnegative():
    p = RandomProjection(d_in=8, proj_dim=64, seed=0, activation="relu")
    x = torch.randn(10, 8, dtype=torch.float64)
    y = p(x)
    assert (y >= 0).all()


def test_random_projection_no_bias_zero_input_gives_zero_output():
    p = RandomProjection(d_in=8, proj_dim=64, seed=0, activation="relu")
    x = torch.zeros(2, 8, dtype=torch.float64)
    y = p(x)
    assert torch.equal(y, torch.zeros_like(y))


def test_random_projection_preserves_input_dtype():
    p = RandomProjection(d_in=8, proj_dim=16, seed=0, activation="gelu")
    x64 = torch.randn(4, 8, dtype=torch.float64)
    x32 = x64.to(torch.float32)
    y64 = p(x64)
    y32 = p(x32)
    assert y64.dtype == torch.float64
    assert y32.dtype == torch.float32
    
    assert torch.allclose(y64.to(torch.float32), y32, atol=1e-4)


def test_random_projection_unknown_activation_raises():
    with pytest.raises(ValueError):
        RandomProjection(d_in=8, proj_dim=16, seed=0, activation="tanh")


def test_random_projection_wrong_input_dim_raises():
    p = RandomProjection(d_in=8, proj_dim=16, seed=0, activation="relu")
    with pytest.raises(ValueError):
        p(torch.randn(2, 7, dtype=torch.float64))


_RUN_DOWNLOAD_TESTS = os.environ.get("ALGPRONC_RUN_DOWNLOAD_TESTS", "") in ("1", "true", "yes")


@pytest.mark.skipif(not _RUN_DOWNLOAD_TESTS, reason="set ALGPRONC_RUN_DOWNLOAD_TESTS=1 to run network-dependent tests")
def test_cifar100_end_to_end_feature_extraction_smoke(tmp_path):
    from algpronc.data.datasets import build_dataset
    from algpronc.models.backbones import build_backbone

    dataset = build_dataset("cifar100", str(tmp_path), train=False, backbone="resnet50")
    small = torch.utils.data.Subset(dataset, list(range(32)))
    backbone, feature_dim = build_backbone("resnet50", pretrained=True)
    features, labels = extract_features(small, backbone, device="cpu", batch_size=8, num_workers=0)
    assert features.shape == (32, feature_dim)
    assert labels.shape == (32,)
