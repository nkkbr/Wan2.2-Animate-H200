# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from importlib import import_module

__all__ = [
    "configs",
    "distributed",
    "modules",
    "utils",
    "WanI2V",
    "WanS2V",
    "WanT2V",
    "WanTI2V",
    "WanAnimate",
]

_LAZY_ATTRS = {
    "configs": (".configs", None),
    "distributed": (".distributed", None),
    "modules": (".modules", None),
    "utils": (".utils", None),
    "WanI2V": (".image2video", "WanI2V"),
    "WanS2V": (".speech2video", "WanS2V"),
    "WanT2V": (".text2video", "WanT2V"),
    "WanTI2V": (".textimage2video", "WanTI2V"),
    "WanAnimate": (".animate", "WanAnimate"),
}


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name, __name__)
    if attr_name is None:
        return module
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
