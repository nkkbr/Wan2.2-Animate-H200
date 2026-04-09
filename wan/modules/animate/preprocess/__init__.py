# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Lazy exports for preprocess modules.

Avoid importing the heavy preprocess pipeline just to access small helper modules.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["ProcessPipeline", "SAM2VideoPredictor"]


def __getattr__(name: str):
    if name == "ProcessPipeline":
        return import_module(".process_pipepline", __name__).ProcessPipeline
    if name == "SAM2VideoPredictor":
        return import_module(".video_predictor", __name__).SAM2VideoPredictor
    raise AttributeError(name)
