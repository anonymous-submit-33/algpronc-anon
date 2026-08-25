"""Dataset loaders: `build_dataset(name, root, train) -> Dataset`.

Every dataset returns `(image, label)` pairs with `label` a plain Python int
(global class id) and `image` a transformed `torch.Tensor`. The transform is
fixed (resize 224 / center-crop / ImageNet mean-std, no augmentation) because
the backbone is frozen and features are extracted exactly once per split --
augmentation would only add noise to a feature cache we want to be
deterministic and reusable across every experiment.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100, ImageFolder

from algpronc.models.backbones import backbone_transform_meta
from algpronc.utils.logging import get_logger

_SUPPORTED = ("cifar100", "tinyimagenet", "domainnet")

_TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
_DOMAINNET_URL = "http://ai.bu.edu/M3SDA/#dataset"

_log = get_logger("algpronc.data.datasets")


def _to_rgb(img):
    return img.convert("RGB")


def default_eval_transform(*, image_size: int = 224, resize: int = 256, mean=None, std=None) -> T.Compose:
    """The one transform used everywhere: resize -> center-crop -> normalise.

    No augmentation (no random crop/flip/jitter): the backbone is frozen, so
    every extra bit of stochasticity here just adds noise to a feature cache
    we want to be exactly reusable across experiments, and would force
    re-extraction if we ever wanted it deterministic.

    Uses a module-level function (not a lambda) for the RGB-conversion step:
    `DataLoader(num_workers>0)` pickles the transform to hand it to worker
    processes, and a closure lambda is not picklable.
    """
    mean = mean or (0.485, 0.456, 0.406)
    std = std or (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(image_size),
            T.Lambda(_to_rgb),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def _transform_for(backbone: str | None) -> T.Compose:
    if backbone is None:
        return default_eval_transform()
    meta = backbone_transform_meta(backbone)
    return default_eval_transform(image_size=meta["image_size"], resize=meta["resize"], mean=meta["mean"], std=meta["std"])


def _build_cifar100(root: str | os.PathLike, train: bool, transform: T.Compose) -> Dataset:
    data_root = Path(root)
    data_root.mkdir(parents=True, exist_ok=True)
    return CIFAR100(root=str(data_root), train=train, download=True, transform=transform)


def _download_and_extract_tinyimagenet(dest_root: Path) -> Path:
    """Download+extract the standard tiny-imagenet-200.zip if not already present.

    Returns the path to the extracted `tiny-imagenet-200` directory.
    """
    extracted = dest_root / "tiny-imagenet-200"
    if extracted.is_dir():
        return extracted

    dest_root.mkdir(parents=True, exist_ok=True)
    zip_path = dest_root / "tiny-imagenet-200.zip"
    if not zip_path.exists():
        _log.info(f"tinyimagenet: downloading {_TINYIMAGENET_URL} -> {zip_path}")
        urllib.request.urlretrieve(_TINYIMAGENET_URL, zip_path)
    else:
        _log.info(f"tinyimagenet: reusing already-downloaded {zip_path}")

    _log.info(f"tinyimagenet: extracting {zip_path} -> {dest_root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_root)

    return extracted


def _reorganise_tinyimagenet_val(extracted: Path) -> None:
    """TinyImageNet's val split ships as a flat `val/images/*.JPEG` dir plus a
    `val/val_annotations.txt` (tab-separated: filename, wnid, x1,y1,x2,y2).
    ImageFolder needs `val/<wnid>/*.JPEG`, so move each image into its class
    folder once. Idempotent: no-op if already reorganised.
    """
    val_dir = extracted / "val"
    flat_images_dir = val_dir / "images"
    annotations_path = val_dir / "val_annotations.txt"

    if not flat_images_dir.is_dir():
        return  

    _log.info(f"tinyimagenet: reorganising {val_dir} into per-class folders")
    with open(annotations_path) as fh:
        rows = [line.strip().split("\t") for line in fh if line.strip()]

    for filename, wnid, *_ in rows:
        class_dir = val_dir / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        src = flat_images_dir / filename
        if src.exists():
            shutil.move(str(src), str(class_dir / filename))

    shutil.rmtree(flat_images_dir, ignore_errors=True)


def _build_tinyimagenet(root: str | os.PathLike, train: bool, transform: T.Compose) -> Dataset:
    dest_root = Path(root) / "tinyimagenet"
    extracted = _download_and_extract_tinyimagenet(dest_root)
    if train:
        split_dir = extracted / "train"
        
        
    else:
        split_dir = extracted / "val"
        _reorganise_tinyimagenet_val(extracted)
    return ImageFolder(root=str(split_dir), transform=transform)


def _build_domainnet(root: str | os.PathLike, train: bool, transform: T.Compose) -> Dataset:
    raise NotImplementedError(
        "DomainNet is not implemented (explicitly deprioritised for this pass). "
        f"Download from {_DOMAINNET_URL} -- six per-domain zips "
        "(clipart/infograph/painting/quickdraw/real/sketch.zip) plus official "
        "train/test split-list .txt files per domain. A loader would need a "
        "`domain` selector (not just train/test) and to parse the split lists "
        "into an ImageFolder-like index; left as future work rather than a "
        "rushed implementation."
    )


def build_dataset(
    name: str,
    root: str | os.PathLike,
    train: bool,
    *,
    backbone: str | None = None,
    transform: T.Compose | None = None,
) -> Dataset:
    """Build a `(image, label)` dataset, downloading/extracting on first use.

    Args:
        name: `'cifar100' | 'tinyimagenet' | 'domainnet'`.
        root: dataset root directory (typically `cfg.resolved_data_dir()`);
            each dataset gets its own subdirectory under it.
        train: train split if True, else the held-out eval split (for
            TinyImageNet this is the `val` split, since `test` ships unlabeled).
        backbone: if given, use that backbone's transform meta (image size /
            normalisation) instead of the default 224/ImageNet-mean-std.
        transform: full override of the transform; takes precedence over `backbone`.

    Returns:
        `torch.utils.data.Dataset` yielding `(image: Tensor, label: int)`.
    """
    if transform is None:
        transform = _transform_for(backbone)

    if name == "cifar100":
        return _build_cifar100(root, train, transform)
    if name == "tinyimagenet":
        return _build_tinyimagenet(root, train, transform)
    if name == "domainnet":
        return _build_domainnet(root, train, transform)
    raise ValueError(f"unknown dataset '{name}'; supported datasets: {_SUPPORTED}")
