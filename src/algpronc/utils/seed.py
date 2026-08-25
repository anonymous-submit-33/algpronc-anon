from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (all devices)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def child_seed(seed: int, *tags: int | str) -> int:
    """Derive a stable sub-seed (e.g. per client, per round)."""
    h = seed
    for tag in tags:
        t = tag if isinstance(tag, int) else int.from_bytes(str(tag).encode(), "little", signed=False)
        h = (h * 1000003 + t) % (2**31 - 1)
    return h
