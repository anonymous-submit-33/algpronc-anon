"""Frozen backbones and RanPAC-style random projection."""

from __future__ import annotations

from .backbones import build_backbone
from .projection import RandomProjection, build_projection

__all__ = ["build_backbone", "build_projection", "RandomProjection"]
