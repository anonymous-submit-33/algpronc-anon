from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """cuda if available, else mps, else cpu. `preference` may force a backend."""
    if preference and preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
