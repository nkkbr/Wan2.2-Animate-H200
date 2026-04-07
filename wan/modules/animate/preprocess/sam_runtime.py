import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SAM_RUNTIME_PROFILES = {
    "legacy_safe": {
        "use_flash_attn": False,
        "math_kernel_on": True,
        "old_gpu": True,
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
    },
    "h200_safe": {
        "use_flash_attn": False,
        "math_kernel_on": True,
        "old_gpu": False,
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": False,
    },
    "h200_aggressive": {
        "use_flash_attn": False,
        "math_kernel_on": True,
        "old_gpu": False,
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
    },
}


PREPROCESS_RUNTIME_PROFILES = {
    "legacy_safe": {
        "preprocess_runtime_profile": "legacy_safe",
        "sam_runtime_profile": "legacy_safe",
        "analysis_resolution_area": None,
        "analysis_min_short_side": None,
        "sam_apply_postprocessing": True,
        "sam_chunk_len": 120,
        "sam_keyframes_per_chunk": 8,
        "sam_use_negative_points": True,
        "sam_reprompt_interval": 40,
        "sam_prompt_mode": "points",
    },
    "h200_safe": {
        "preprocess_runtime_profile": "h200_safe",
        "sam_runtime_profile": "h200_safe",
        "analysis_resolution_area": [832, 480],
        "analysis_min_short_side": 480,
        "sam_apply_postprocessing": False,
        "sam_chunk_len": 20,
        "sam_keyframes_per_chunk": 4,
        "sam_use_negative_points": False,
        "sam_reprompt_interval": 0,
        "sam_prompt_mode": "mask_seed",
    },
    "h200_aggressive": {
        "preprocess_runtime_profile": "h200_aggressive",
        "sam_runtime_profile": "h200_aggressive",
        "analysis_resolution_area": [1280, 720],
        "analysis_min_short_side": 720,
        "sam_apply_postprocessing": False,
        "sam_chunk_len": 60,
        "sam_keyframes_per_chunk": 6,
        "sam_use_negative_points": False,
        "sam_reprompt_interval": 0,
        "sam_prompt_mode": "mask_seed",
    },
}


def resolve_sam_runtime_profile(
    profile_name="legacy_safe",
    *,
    use_flash_attn=None,
    math_kernel_on=None,
    old_gpu=None,
    offload_video_to_cpu=None,
    offload_state_to_cpu=None,
):
    if profile_name not in SAM_RUNTIME_PROFILES:
        raise ValueError(
            f"Unsupported SAM runtime profile: {profile_name}. "
            f"Expected one of {sorted(SAM_RUNTIME_PROFILES)}."
        )
    config = deepcopy(SAM_RUNTIME_PROFILES[profile_name])
    config["profile_name"] = profile_name
    overrides = {
        "use_flash_attn": use_flash_attn,
        "math_kernel_on": math_kernel_on,
        "old_gpu": old_gpu,
        "offload_video_to_cpu": offload_video_to_cpu,
        "offload_state_to_cpu": offload_state_to_cpu,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = bool(value)
    return config


def resolve_preprocess_runtime_profile(profile_name="legacy_safe"):
    if profile_name not in PREPROCESS_RUNTIME_PROFILES:
        raise ValueError(
            f"Unsupported preprocess runtime profile: {profile_name}. "
            f"Expected one of {sorted(PREPROCESS_RUNTIME_PROFILES)}."
        )
    config = deepcopy(PREPROCESS_RUNTIME_PROFILES[profile_name])
    if config["analysis_resolution_area"] is not None:
        config["analysis_resolution_area"] = [int(config["analysis_resolution_area"][0]), int(config["analysis_resolution_area"][1])]
    if config["analysis_min_short_side"] is not None:
        config["analysis_min_short_side"] = int(config["analysis_min_short_side"])
    return config


def apply_sam_runtime_profile(runtime_config):
    import sam2.modeling.sam.transformer as transformer

    transformer.USE_FLASH_ATTN = bool(runtime_config["use_flash_attn"])
    transformer.MATH_KERNEL_ON = bool(runtime_config["math_kernel_on"])
    transformer.OLD_GPU = bool(runtime_config["old_gpu"])
    return {
        "profile_name": runtime_config["profile_name"],
        "use_flash_attn": bool(transformer.USE_FLASH_ATTN),
        "math_kernel_on": bool(transformer.MATH_KERNEL_ON),
        "old_gpu": bool(transformer.OLD_GPU),
        "offload_video_to_cpu": bool(runtime_config["offload_video_to_cpu"]),
        "offload_state_to_cpu": bool(runtime_config["offload_state_to_cpu"]),
    }


def ensure_trace_dir(output_path=None, trace_dir=None):
    if trace_dir is not None:
        target = Path(trace_dir)
    elif output_path is not None:
        target = Path(output_path) / "sam2_debug"
    else:
        raise ValueError("Either output_path or trace_dir must be provided.")
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_trace_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)
    return path


def _to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def write_chunk_trace(trace_dir, chunk_index, payload):
    trace_dir = ensure_trace_dir(trace_dir=trace_dir)
    payload = _to_builtin(payload)
    payload["chunk_index"] = int(chunk_index)
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    chunk_path = trace_dir / f"chunk_{int(chunk_index):04d}.json"
    latest_path = trace_dir / "latest_state.json"
    write_trace_json(chunk_path, payload)
    write_trace_json(latest_path, payload)
    return chunk_path


def prompt_entry_trace(prompt_entry):
    points = np.asarray(prompt_entry["points"], dtype=np.int32)
    labels = np.asarray(prompt_entry["labels"], dtype=np.int32)
    summary = {
        "frame_idx": int(prompt_entry["frame_idx"]),
        "tags": list(prompt_entry.get("tags", [])),
        "positive_count": int(prompt_entry.get("positive_count", int((labels == 1).sum()))),
        "negative_count": int(prompt_entry.get("negative_count", int((labels == 0).sum()))),
        "point_count": int(points.shape[0]),
        "label_count": int(labels.shape[0]),
        "positive_sources": _to_builtin(prompt_entry.get("positive_sources", {})),
        "person_bbox": _to_builtin(prompt_entry.get("person_bbox")),
    }
    if points.size > 0:
        summary["point_min"] = [int(points[:, 0].min()), int(points[:, 1].min())]
        summary["point_max"] = [int(points[:, 0].max()), int(points[:, 1].max())]
    else:
        summary["point_min"] = None
        summary["point_max"] = None
    return summary
