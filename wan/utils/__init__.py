# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from importlib import import_module

__all__ = [
    "HuggingfaceTokenizer",
    "get_sampling_sigmas",
    "retrieve_timesteps",
    "FlowDPMSolverMultistepScheduler",
    "FlowUniPCMultistepScheduler",
]

_LAZY_ATTRS = {
    "HuggingfaceTokenizer": ("..modules.tokenizers", "HuggingfaceTokenizer"),
    "get_sampling_sigmas": (".fm_solvers", "get_sampling_sigmas"),
    "retrieve_timesteps": (".fm_solvers", "retrieve_timesteps"),
    "FlowDPMSolverMultistepScheduler": (".fm_solvers", "FlowDPMSolverMultistepScheduler"),
    "FlowUniPCMultistepScheduler": (".fm_solvers_unipc", "FlowUniPCMultistepScheduler"),
}


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
