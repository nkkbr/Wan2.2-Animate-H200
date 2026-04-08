import json
import logging
from pathlib import Path

import cv2
import numpy as np


PREPROCESS_METADATA_VERSION = 1
PREPROCESS_STORAGE_FORMAT = "wan_animate_preprocess_v1"
PREPROCESS_COLOR_SPACE = "rgb"
PERSON_MASK_SEMANTICS = "person_foreground"
HARD_FOREGROUND_SEMANTICS = "hard_foreground"
BACKGROUND_KEEP_MASK_SEMANTICS = "1 - person_foreground"
BACKGROUND_KEEP_PRIOR_SEMANTICS = "background_keep_prior"
BACKGROUND_VISIBLE_SUPPORT_SEMANTICS = "background_visible_support"
UNRESOLVED_REGION_SEMANTICS = "background_unresolved_region"
BACKGROUND_CONFIDENCE_SEMANTICS = "background_confidence"
BACKGROUND_SOURCE_PROVENANCE_SEMANTICS = "background_source_provenance"
SOFT_ALPHA_SEMANTICS = "soft_alpha"
ALPHA_V2_SEMANTICS = "alpha_v2"
TRIMAP_V2_SEMANTICS = "trimap_v2"
ALPHA_UNCERTAINTY_V2_SEMANTICS = "alpha_uncertainty_v2"
FINE_BOUNDARY_MASK_SEMANTICS = "fine_boundary_mask"
HAIR_EDGE_MASK_SEMANTICS = "hair_edge_mask"
ALPHA_CONFIDENCE_SEMANTICS = "alpha_confidence_v2"
ALPHA_SOURCE_PROVENANCE_SEMANTICS = "alpha_source_provenance_v2"
SOFT_BAND_SEMANTICS = "boundary_transition_band"
BOUNDARY_BAND_SEMANTICS = SOFT_BAND_SEMANTICS
OCCLUSION_BAND_SEMANTICS = "occlusion_band"
UNCERTAINTY_MAP_SEMANTICS = "uncertainty_map"
FACE_ALPHA_SEMANTICS = "face_alpha"
FACE_UNCERTAINTY_SEMANTICS = "face_uncertainty"
FACE_PARSING_SEMANTICS = "face_parsing_v1"
POSE_UNCERTAINTY_SEMANTICS = "pose_uncertainty"
FACE_BOUNDARY_SEMANTICS = "face_boundary"
HAIR_BOUNDARY_SEMANTICS = "hair_boundary"
HAND_BOUNDARY_SEMANTICS = "hand_boundary"
CLOTH_BOUNDARY_SEMANTICS = "cloth_boundary"
OCCLUDED_BOUNDARY_SEMANTICS = "occluded_boundary"
DEFAULT_REFERT_NUM = 5


def validate_refert_num(refert_num: int, clip_len: int | None = None) -> int:
    if not isinstance(refert_num, int):
        raise TypeError(f"refert_num must be an integer. Got {type(refert_num)!r}.")
    if refert_num <= 0:
        raise ValueError(f"refert_num must be > 0. Got {refert_num}.")
    if clip_len is not None and refert_num >= clip_len:
        raise ValueError(
            f"refert_num must satisfy 0 < refert_num < clip_len. Got refert_num={refert_num}, clip_len={clip_len}."
        )
    return refert_num


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_image_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_video_rgb(path: str | Path) -> np.ndarray:
    from decord import VideoReader

    reader = VideoReader(str(path))
    indices = list(range(len(reader)))
    frames = reader.get_batch(indices).asnumpy()
    validate_rgb_video(name=str(path), frames=frames)
    return frames


def validate_rgb_video(name: str, frames: np.ndarray) -> None:
    _require(isinstance(frames, np.ndarray), f"{name} must be a numpy array.")
    _require(frames.ndim == 4, f"{name} must have shape [T, H, W, C]. Got {frames.shape}.")
    _require(frames.shape[-1] == 3, f"{name} must use 3 RGB channels. Got {frames.shape}.")
    _require(frames.shape[0] > 0, f"{name} must contain at least one frame.")


def validate_rgb_image(name: str, image: np.ndarray) -> None:
    _require(isinstance(image, np.ndarray), f"{name} must be a numpy array.")
    _require(image.ndim == 3, f"{name} must have shape [H, W, C]. Got {image.shape}.")
    _require(image.shape[-1] == 3, f"{name} must use 3 RGB channels. Got {image.shape}.")


def validate_person_mask_frames(name: str, mask_frames: np.ndarray) -> None:
    _require(isinstance(mask_frames, np.ndarray), f"{name} must be a numpy array.")
    _require(mask_frames.ndim == 3, f"{name} must have shape [T, H, W]. Got {mask_frames.shape}.")
    _require(mask_frames.shape[0] > 0, f"{name} must contain at least one frame.")
    _require(np.isfinite(mask_frames).all(), f"{name} contains non-finite values.")
    if mask_frames.size > 0:
        min_value = float(mask_frames.min())
        max_value = float(mask_frames.max())
        _require(min_value >= -1e-6 and max_value <= 1 + 1e-6,
                 f"{name} must stay within [0, 1]. Got [{min_value}, {max_value}].")


def build_preprocess_metadata(
    *,
    video_path: str,
    refer_image_path: str,
    output_path: str,
    replace_flag: bool,
    retarget_flag: bool,
    use_flux: bool,
    resolution_area: list[int] | tuple[int, int],
    analysis_settings: dict | None = None,
    fps_request: int,
    fps_output: float,
    frame_count: int,
    height: int,
    width: int,
    iterations: int,
    k: int,
    w_len: int,
    h_len: int,
    reference_height: int,
    reference_width: int,
    src_files: dict | None = None,
    intermediate_save_format: str = "mp4",
    lossless_intermediate: bool = False,
    control_stabilization: dict | None = None,
    mask_generation: dict | None = None,
    soft_mask_settings: dict | None = None,
    background_settings: dict | None = None,
    boundary_fusion_settings: dict | None = None,
    reference_settings: dict | None = None,
    runtime_stats: dict | None = None,
    qa_outputs: dict | None = None,
) -> dict:
    if src_files is None:
        src_files = {
            "pose": {
                "path": "src_pose.mp4",
                "type": "video",
                "format": "mp4",
                "frame_count": frame_count,
                "height": height,
                "width": width,
                "channels": 3,
                "color_space": PREPROCESS_COLOR_SPACE,
                "dtype": "uint8",
                "shape": [frame_count, height, width, 3],
                "fps": float(fps_output),
            },
            "face": {
                "path": "src_face.mp4",
                "type": "video",
                "format": "mp4",
                "frame_count": frame_count,
                "height": 512,
                "width": 512,
                "channels": 3,
                "color_space": PREPROCESS_COLOR_SPACE,
                "dtype": "uint8",
                "shape": [frame_count, 512, 512, 3],
                "fps": float(fps_output),
            },
            "reference": {
                "path": "src_ref.png",
                "type": "image",
                "format": "png",
                "height": reference_height,
                "width": reference_width,
                "channels": 3,
                "color_space": PREPROCESS_COLOR_SPACE,
                "dtype": "uint8",
                "shape": [reference_height, reference_width, 3],
                "resized_height": height,
                "resized_width": width,
            },
        }

        if replace_flag:
            src_files["background"] = {
                "path": "src_bg.mp4",
                "type": "video",
                "format": "mp4",
                "frame_count": frame_count,
                "height": height,
                "width": width,
                "channels": 3,
                "color_space": PREPROCESS_COLOR_SPACE,
                "dtype": "uint8",
                "shape": [frame_count, height, width, 3],
                "fps": float(fps_output),
                "background_mode": "hole",
            }
            src_files["person_mask"] = {
                "path": "src_mask.mp4",
                "type": "video",
                "format": "mp4",
                "frame_count": frame_count,
                "height": height,
                "width": width,
                "channels": 1,
                "stored_channels": 3,
                "color_space": PREPROCESS_COLOR_SPACE,
                "dtype": "float32",
                "shape": [frame_count, height, width],
                "fps": float(fps_output),
                "value_range": [0.0, 1.0],
                "stored_value_range": [0, 255],
                "mask_semantics": PERSON_MASK_SEMANTICS,
            }

    return {
        "version": PREPROCESS_METADATA_VERSION,
        "pipeline": "wan_animate_preprocess",
        "storage_format": PREPROCESS_STORAGE_FORMAT,
        "mode": "replacement" if replace_flag else "animation",
        "replace_flag": replace_flag,
        "retarget_flag": retarget_flag,
        "use_flux": use_flux,
        "frame_count": frame_count,
        "fps": float(fps_output),
        "height": height,
        "width": width,
        "channels": 3,
        "color_space": PREPROCESS_COLOR_SPACE,
        "mask_semantics": PERSON_MASK_SEMANTICS if replace_flag else None,
        "derived_mask_semantics": {
            "background_keep_mask": BACKGROUND_KEEP_MASK_SEMANTICS,
        } if replace_flag else {},
        "processing": {
            "resolution_area": list(resolution_area),
            "analysis": analysis_settings or {},
            "fps_request": fps_request,
            "save_format": intermediate_save_format,
            "lossless_intermediate": bool(lossless_intermediate),
            "mask_strategy": {
                "iterations": iterations,
                "k": k,
                "w_len": w_len,
                "h_len": h_len,
            },
            "control_stabilization": control_stabilization or {},
            "sam2_mask_generation": mask_generation or {},
            "soft_mask": soft_mask_settings or {},
            "background": background_settings or {},
            "boundary_fusion": boundary_fusion_settings or {},
            "reference_normalization": reference_settings or {},
        },
        "runtime": runtime_stats or {},
        "source_inputs": {
            "video_path": str(Path(video_path).resolve()),
            "refer_image_path": str(Path(refer_image_path).resolve()),
            "output_path": str(Path(output_path).resolve()),
        },
        "src_files": src_files,
        "qa_outputs": qa_outputs or {},
    }


def write_preprocess_metadata(output_path: str | Path, metadata: dict) -> Path:
    metadata_path = Path(output_path) / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return metadata_path


def load_preprocess_metadata(src_root_path: str | Path, logger: logging.Logger | None = None) -> dict | None:
    metadata_path = Path(src_root_path) / "metadata.json"
    if not metadata_path.exists():
        if logger is not None:
            logger.warning(
                "No preprocess metadata found at %s. Falling back to the legacy filename-only contract.",
                metadata_path,
            )
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    validate_preprocess_metadata(metadata, src_root_path=src_root_path)
    return metadata


def validate_preprocess_metadata(metadata: dict, src_root_path: str | Path) -> None:
    _require(isinstance(metadata, dict), "metadata must be a JSON object.")
    required_keys = [
        "version",
        "storage_format",
        "replace_flag",
        "frame_count",
        "fps",
        "height",
        "width",
        "channels",
        "color_space",
        "src_files",
    ]
    for key in required_keys:
        _require(key in metadata, f"metadata.json is missing required field: {key}")

    _require(
        metadata["version"] == PREPROCESS_METADATA_VERSION,
        f"Unsupported preprocess metadata version: {metadata['version']}",
    )
    _require(
        metadata["storage_format"] == PREPROCESS_STORAGE_FORMAT,
        f"Unsupported preprocess storage format: {metadata['storage_format']}",
    )
    _require(
        metadata["color_space"] == PREPROCESS_COLOR_SPACE,
        f"Unsupported preprocess color space: {metadata['color_space']}",
    )

    src_files = metadata["src_files"]
    _require(isinstance(src_files, dict), "metadata.src_files must be an object.")
    for key in ("pose", "face", "reference"):
        _require(key in src_files, f"metadata.src_files is missing required artifact: {key}")

    if metadata["replace_flag"]:
        _require(
            metadata.get("mask_semantics") == PERSON_MASK_SEMANTICS,
            "replacement metadata must declare person_foreground mask semantics.",
        )
        for key in ("background", "person_mask"):
            _require(key in src_files, f"replacement metadata missing required artifact: {key}")
        background_mode = src_files["background"].get("background_mode")
        if background_mode is not None:
            _require(
                background_mode in {"hole", "clean_plate_image", "clean_plate_video", "clean_plate_video_v1", "clean_plate_video_v2"},
                f"Unsupported background_mode in metadata: {background_mode}",
            )
        if "soft_band" in src_files:
            _require(
                src_files["soft_band"].get("mask_semantics") == SOFT_BAND_SEMANTICS,
                "soft_band artifact semantics mismatch.",
            )
        if "hard_foreground" in src_files:
            _require(
                src_files["hard_foreground"].get("mask_semantics") == HARD_FOREGROUND_SEMANTICS,
                "hard_foreground artifact semantics mismatch.",
            )
        if "soft_alpha" in src_files:
            _require(
                src_files["soft_alpha"].get("mask_semantics") == SOFT_ALPHA_SEMANTICS,
                "soft_alpha artifact semantics mismatch.",
            )
        if "alpha_v2" in src_files:
            _require(
                src_files["alpha_v2"].get("mask_semantics") == ALPHA_V2_SEMANTICS,
                "alpha_v2 artifact semantics mismatch.",
            )
        if "trimap_v2" in src_files:
            _require(
                src_files["trimap_v2"].get("mask_semantics") == TRIMAP_V2_SEMANTICS,
                "trimap_v2 artifact semantics mismatch.",
            )
        if "alpha_uncertainty_v2" in src_files:
            _require(
                src_files["alpha_uncertainty_v2"].get("mask_semantics") == ALPHA_UNCERTAINTY_V2_SEMANTICS,
                "alpha_uncertainty_v2 artifact semantics mismatch.",
            )
        if "fine_boundary_mask" in src_files:
            _require(
                src_files["fine_boundary_mask"].get("mask_semantics") == FINE_BOUNDARY_MASK_SEMANTICS,
                "fine_boundary_mask artifact semantics mismatch.",
            )
        if "hair_edge_mask" in src_files:
            _require(
                src_files["hair_edge_mask"].get("mask_semantics") == HAIR_EDGE_MASK_SEMANTICS,
                "hair_edge_mask artifact semantics mismatch.",
            )
        if "alpha_confidence_v2" in src_files:
            _require(
                src_files["alpha_confidence_v2"].get("mask_semantics") == ALPHA_CONFIDENCE_SEMANTICS,
                "alpha_confidence_v2 artifact semantics mismatch.",
            )
        if "alpha_source_provenance_v2" in src_files:
            _require(
                src_files["alpha_source_provenance_v2"].get("mask_semantics") == ALPHA_SOURCE_PROVENANCE_SEMANTICS,
                "alpha_source_provenance_v2 artifact semantics mismatch.",
            )
        if "boundary_band" in src_files:
            _require(
                src_files["boundary_band"].get("mask_semantics") == BOUNDARY_BAND_SEMANTICS,
                "boundary_band artifact semantics mismatch.",
            )
        if "background_keep_prior" in src_files:
            _require(
                src_files["background_keep_prior"].get("mask_semantics") == BACKGROUND_KEEP_PRIOR_SEMANTICS,
                "background_keep_prior artifact semantics mismatch.",
            )
        if "visible_support" in src_files:
            _require(
                src_files["visible_support"].get("mask_semantics") == BACKGROUND_VISIBLE_SUPPORT_SEMANTICS,
                "visible_support artifact semantics mismatch.",
            )
        if "unresolved_region" in src_files:
            _require(
                src_files["unresolved_region"].get("mask_semantics") == UNRESOLVED_REGION_SEMANTICS,
                "unresolved_region artifact semantics mismatch.",
            )
        if "background_confidence" in src_files:
            _require(
                src_files["background_confidence"].get("mask_semantics") == BACKGROUND_CONFIDENCE_SEMANTICS,
                "background_confidence artifact semantics mismatch.",
            )
        if "background_source_provenance" in src_files:
            _require(
                src_files["background_source_provenance"].get("mask_semantics") == BACKGROUND_SOURCE_PROVENANCE_SEMANTICS,
                "background_source_provenance artifact semantics mismatch.",
            )
        if "occlusion_band" in src_files:
            _require(
                src_files["occlusion_band"].get("mask_semantics") == OCCLUSION_BAND_SEMANTICS,
                "occlusion_band artifact semantics mismatch.",
            )
        if "uncertainty_map" in src_files:
            _require(
                src_files["uncertainty_map"].get("mask_semantics") == UNCERTAINTY_MAP_SEMANTICS,
                "uncertainty_map artifact semantics mismatch.",
            )
        if "face_alpha" in src_files:
            _require(
                src_files["face_alpha"].get("mask_semantics") == FACE_ALPHA_SEMANTICS,
                "face_alpha artifact semantics mismatch.",
            )
        if "face_uncertainty" in src_files:
            _require(
                src_files["face_uncertainty"].get("mask_semantics") == FACE_UNCERTAINTY_SEMANTICS,
                "face_uncertainty artifact semantics mismatch.",
            )
        if "face_parsing" in src_files:
            _require(
                src_files["face_parsing"].get("label_semantics") == FACE_PARSING_SEMANTICS,
                "face_parsing artifact semantics mismatch.",
            )
        if "pose_uncertainty" in src_files:
            _require(
                src_files["pose_uncertainty"].get("mask_semantics") == POSE_UNCERTAINTY_SEMANTICS,
                "pose_uncertainty artifact semantics mismatch.",
            )
        if "face_boundary" in src_files:
            _require(
                src_files["face_boundary"].get("mask_semantics") == FACE_BOUNDARY_SEMANTICS,
                "face_boundary artifact semantics mismatch.",
            )
        if "hair_boundary" in src_files:
            _require(
                src_files["hair_boundary"].get("mask_semantics") == HAIR_BOUNDARY_SEMANTICS,
                "hair_boundary artifact semantics mismatch.",
            )
        if "hand_boundary" in src_files:
            _require(
                src_files["hand_boundary"].get("mask_semantics") == HAND_BOUNDARY_SEMANTICS,
                "hand_boundary artifact semantics mismatch.",
            )
        if "cloth_boundary" in src_files:
            _require(
                src_files["cloth_boundary"].get("mask_semantics") == CLOTH_BOUNDARY_SEMANTICS,
                "cloth_boundary artifact semantics mismatch.",
            )
        if "occluded_boundary" in src_files:
            _require(
                src_files["occluded_boundary"].get("mask_semantics") == OCCLUDED_BOUNDARY_SEMANTICS,
                "occluded_boundary artifact semantics mismatch.",
            )

    root = Path(src_root_path)
    for name, artifact in src_files.items():
        _require("path" in artifact, f"metadata artifact {name} is missing its relative path.")
        artifact_path = root / artifact["path"]
        _require(artifact_path.exists(), f"Required preprocess artifact is missing: {artifact_path}")


def resolve_preprocess_artifacts(
    src_root_path: str | Path,
    replace_flag: bool,
    logger: logging.Logger | None = None,
) -> tuple[dict, dict | None]:
    metadata = load_preprocess_metadata(src_root_path, logger=logger)
    root = Path(src_root_path)
    if metadata is None:
        artifacts = {
            "pose": {"path": root / "src_pose.mp4", "format": "mp4"},
            "face": {"path": root / "src_face.mp4", "format": "mp4"},
            "reference": {"path": root / "src_ref.png", "format": "png"},
        }
        if replace_flag:
            artifacts["background"] = {"path": root / "src_bg.mp4", "format": "mp4"}
            artifacts["person_mask"] = {"path": root / "src_mask.mp4", "format": "mp4"}
        for name, artifact in artifacts.items():
            path = Path(artifact["path"])
            if not path.exists():
                raise FileNotFoundError(f"Missing required preprocess artifact: {path} ({name})")
        return {
            key: {"path": str(Path(artifact["path"]).resolve()), "format": artifact["format"]}
            for key, artifact in artifacts.items()
        }, None

    if replace_flag and not metadata["replace_flag"]:
        raise ValueError(
            "generate was invoked with --replace_flag, but preprocess metadata declares replace_flag=false."
        )

    artifacts = {
        name: {
            "path": str((root / artifact["path"]).resolve()),
            "format": artifact.get("format"),
        }
        for name, artifact in metadata["src_files"].items()
        if name in {
            "pose",
            "face",
            "reference",
            "background",
            "person_mask",
            "soft_band",
            "hard_foreground",
                "soft_alpha",
                "boundary_band",
                "background_keep_prior",
                "visible_support",
                "unresolved_region",
                "background_confidence",
                "background_source_provenance",
                "occlusion_band",
                "uncertainty_map",
                "face_landmarks",
                "face_pose",
                "face_expression",
                "face_alpha",
                "face_parsing",
                "face_uncertainty",
                "face_boundary",
                "hair_boundary",
                "hand_boundary",
                "cloth_boundary",
                "occluded_boundary",
                "pose_tracks",
                "limb_tracks",
                "hand_tracks",
                "pose_visibility",
                "pose_uncertainty",
            }
    }
    return artifacts, metadata


def validate_loaded_preprocess_bundle(
    *,
    cond_images: np.ndarray,
    face_images: np.ndarray,
    refer_image_rgb: np.ndarray,
    metadata: dict | None = None,
    bg_images: np.ndarray | None = None,
    person_mask_images: np.ndarray | None = None,
    soft_band_images: np.ndarray | None = None,
    hard_foreground_images: np.ndarray | None = None,
    soft_alpha_images: np.ndarray | None = None,
    boundary_band_images: np.ndarray | None = None,
    background_keep_prior_images: np.ndarray | None = None,
    visible_support_images: np.ndarray | None = None,
    unresolved_region_images: np.ndarray | None = None,
    background_confidence_images: np.ndarray | None = None,
    background_source_provenance_images: np.ndarray | None = None,
    occlusion_band_images: np.ndarray | None = None,
    uncertainty_map_images: np.ndarray | None = None,
    alpha_v2_images: np.ndarray | None = None,
    trimap_v2_images: np.ndarray | None = None,
    alpha_uncertainty_v2_images: np.ndarray | None = None,
    fine_boundary_mask_images: np.ndarray | None = None,
    hair_edge_mask_images: np.ndarray | None = None,
    alpha_confidence_images: np.ndarray | None = None,
    alpha_source_provenance_images: np.ndarray | None = None,
    face_boundary_images: np.ndarray | None = None,
    hair_boundary_images: np.ndarray | None = None,
    hand_boundary_images: np.ndarray | None = None,
    cloth_boundary_images: np.ndarray | None = None,
    occluded_boundary_images: np.ndarray | None = None,
) -> None:
    validate_rgb_video("conditioning frames", cond_images)
    validate_rgb_video("face frames", face_images)
    validate_rgb_image("reference image", refer_image_rgb)

    frame_count = cond_images.shape[0]
    cond_height, cond_width = cond_images.shape[1:3]
    _require(face_images.shape[0] == frame_count,
             f"Face video frame count {face_images.shape[0]} does not match pose frame count {frame_count}.")

    if bg_images is not None:
        validate_rgb_video("background frames", bg_images)
        _require(bg_images.shape[0] == frame_count,
                 f"Background video frame count {bg_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(bg_images.shape[1:3] == (cond_height, cond_width),
                 "Background video size must match pose video size.")

    if person_mask_images is not None:
        validate_person_mask_frames("person mask frames", person_mask_images)
        _require(person_mask_images.shape[0] == frame_count,
                 f"Person mask frame count {person_mask_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(person_mask_images.shape[1:3] == (cond_height, cond_width),
                 "Person mask size must match pose video size.")
    if soft_band_images is not None:
        validate_person_mask_frames("soft band frames", soft_band_images)
        _require(soft_band_images.shape[0] == frame_count,
                 f"Soft band frame count {soft_band_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(soft_band_images.shape[1:3] == (cond_height, cond_width),
                 "Soft band size must match pose video size.")
    if hard_foreground_images is not None:
        validate_person_mask_frames("hard foreground frames", hard_foreground_images)
        _require(hard_foreground_images.shape[0] == frame_count,
                 f"Hard foreground frame count {hard_foreground_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(hard_foreground_images.shape[1:3] == (cond_height, cond_width),
                 "Hard foreground size must match pose video size.")
    if soft_alpha_images is not None:
        validate_person_mask_frames("soft alpha frames", soft_alpha_images)
        _require(soft_alpha_images.shape[0] == frame_count,
                 f"Soft alpha frame count {soft_alpha_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(soft_alpha_images.shape[1:3] == (cond_height, cond_width),
                 "Soft alpha size must match pose video size.")
    if boundary_band_images is not None:
        validate_person_mask_frames("boundary band frames", boundary_band_images)
        _require(boundary_band_images.shape[0] == frame_count,
                 f"Boundary band frame count {boundary_band_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(boundary_band_images.shape[1:3] == (cond_height, cond_width),
                 "Boundary band size must match pose video size.")
    if background_keep_prior_images is not None:
        validate_person_mask_frames("background keep prior frames", background_keep_prior_images)
        _require(background_keep_prior_images.shape[0] == frame_count,
                 f"Background keep prior frame count {background_keep_prior_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(background_keep_prior_images.shape[1:3] == (cond_height, cond_width),
                 "Background keep prior size must match pose video size.")
    if visible_support_images is not None:
        validate_person_mask_frames("background visible support frames", visible_support_images)
        _require(visible_support_images.shape[0] == frame_count,
                 f"Background visible support frame count {visible_support_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(visible_support_images.shape[1:3] == (cond_height, cond_width),
                 "Background visible support size must match pose video size.")
    if unresolved_region_images is not None:
        validate_person_mask_frames("background unresolved region frames", unresolved_region_images)
        _require(unresolved_region_images.shape[0] == frame_count,
                 f"Background unresolved region frame count {unresolved_region_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(unresolved_region_images.shape[1:3] == (cond_height, cond_width),
                 "Background unresolved region size must match pose video size.")
    if background_confidence_images is not None:
        validate_person_mask_frames("background confidence frames", background_confidence_images)
        _require(background_confidence_images.shape[0] == frame_count,
                 f"Background confidence frame count {background_confidence_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(background_confidence_images.shape[1:3] == (cond_height, cond_width),
                 "Background confidence size must match pose video size.")
    if background_source_provenance_images is not None:
        validate_person_mask_frames("background provenance frames", background_source_provenance_images)
        _require(background_source_provenance_images.shape[0] == frame_count,
                 f"Background provenance frame count {background_source_provenance_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(background_source_provenance_images.shape[1:3] == (cond_height, cond_width),
                 "Background provenance size must match pose video size.")
    if occlusion_band_images is not None:
        validate_person_mask_frames("occlusion band frames", occlusion_band_images)
        _require(occlusion_band_images.shape[0] == frame_count,
                 f"Occlusion band frame count {occlusion_band_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(occlusion_band_images.shape[1:3] == (cond_height, cond_width),
                 "Occlusion band size must match pose video size.")
    if uncertainty_map_images is not None:
        validate_person_mask_frames("uncertainty map frames", uncertainty_map_images)
        _require(uncertainty_map_images.shape[0] == frame_count,
                 f"Uncertainty map frame count {uncertainty_map_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(uncertainty_map_images.shape[1:3] == (cond_height, cond_width),
                 "Uncertainty map size must match pose video size.")
    if alpha_v2_images is not None:
        validate_person_mask_frames("alpha_v2 frames", alpha_v2_images)
        _require(alpha_v2_images.shape[0] == frame_count,
                 f"alpha_v2 frame count {alpha_v2_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(alpha_v2_images.shape[1:3] == (cond_height, cond_width),
                 "alpha_v2 size must match pose video size.")
    if trimap_v2_images is not None:
        validate_person_mask_frames("trimap_v2 frames", trimap_v2_images)
        _require(trimap_v2_images.shape[0] == frame_count,
                 f"trimap_v2 frame count {trimap_v2_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(trimap_v2_images.shape[1:3] == (cond_height, cond_width),
                 "trimap_v2 size must match pose video size.")
    if alpha_uncertainty_v2_images is not None:
        validate_person_mask_frames("alpha uncertainty v2 frames", alpha_uncertainty_v2_images)
        _require(alpha_uncertainty_v2_images.shape[0] == frame_count,
                 f"alpha uncertainty v2 frame count {alpha_uncertainty_v2_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(alpha_uncertainty_v2_images.shape[1:3] == (cond_height, cond_width),
                 "alpha uncertainty v2 size must match pose video size.")
    if fine_boundary_mask_images is not None:
        validate_person_mask_frames("fine boundary mask frames", fine_boundary_mask_images)
        _require(fine_boundary_mask_images.shape[0] == frame_count,
                 f"fine boundary mask frame count {fine_boundary_mask_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(fine_boundary_mask_images.shape[1:3] == (cond_height, cond_width),
                 "fine boundary mask size must match pose video size.")
    if hair_edge_mask_images is not None:
        validate_person_mask_frames("hair edge mask frames", hair_edge_mask_images)
        _require(hair_edge_mask_images.shape[0] == frame_count,
                 f"hair edge mask frame count {hair_edge_mask_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(hair_edge_mask_images.shape[1:3] == (cond_height, cond_width),
                 "hair edge mask size must match pose video size.")
    if alpha_confidence_images is not None:
        validate_person_mask_frames("alpha confidence frames", alpha_confidence_images)
        _require(alpha_confidence_images.shape[0] == frame_count,
                 f"alpha confidence frame count {alpha_confidence_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(alpha_confidence_images.shape[1:3] == (cond_height, cond_width),
                 "alpha confidence size must match pose video size.")
    if alpha_source_provenance_images is not None:
        validate_person_mask_frames("alpha source provenance frames", alpha_source_provenance_images)
        _require(alpha_source_provenance_images.shape[0] == frame_count,
                 f"alpha source provenance frame count {alpha_source_provenance_images.shape[0]} does not match pose frame count {frame_count}.")
        _require(alpha_source_provenance_images.shape[1:3] == (cond_height, cond_width),
                 "alpha source provenance size must match pose video size.")
    for name, value in (
        ("face boundary", face_boundary_images),
        ("hair boundary", hair_boundary_images),
        ("hand boundary", hand_boundary_images),
        ("cloth boundary", cloth_boundary_images),
        ("occluded boundary", occluded_boundary_images),
    ):
        if value is not None:
            validate_person_mask_frames(f"{name} frames", value)
            _require(value.shape[0] == frame_count,
                     f"{name} frame count {value.shape[0]} does not match pose frame count {frame_count}.")
            _require(value.shape[1:3] == (cond_height, cond_width),
                     f"{name} size must match pose video size.")

    if metadata is None:
        return

    _require(metadata["frame_count"] == frame_count,
             f"metadata frame_count={metadata['frame_count']} does not match pose frames={frame_count}.")
    _require(metadata["height"] == cond_height and metadata["width"] == cond_width,
             "metadata height/width does not match pose video size.")
    _require(metadata["channels"] == 3, "metadata channels must be 3 for RGB outputs.")

    pose_meta = metadata["src_files"]["pose"]
    _require(pose_meta["frame_count"] == frame_count, "pose artifact frame_count mismatch.")
    _require(pose_meta["height"] == cond_height and pose_meta["width"] == cond_width,
             "pose artifact size mismatch.")

    face_meta = metadata["src_files"]["face"]
    _require(face_meta["frame_count"] == frame_count, "face artifact frame_count mismatch.")
    _require(face_meta["height"] == face_images.shape[1] and face_meta["width"] == face_images.shape[2],
             "face artifact size mismatch.")

    reference_meta = metadata["src_files"]["reference"]
    _require(reference_meta["channels"] == 3, "reference artifact channels must be 3.")
    _require(reference_meta.get("color_space") == PREPROCESS_COLOR_SPACE,
             "reference artifact color_space mismatch.")
    _require(
        refer_image_rgb.shape[0] == reference_meta["resized_height"]
        and refer_image_rgb.shape[1] == reference_meta["resized_width"],
        "resized reference image size does not match metadata target size.",
    )

    if bg_images is not None:
        background_meta = metadata["src_files"]["background"]
        _require(background_meta["frame_count"] == bg_images.shape[0], "background artifact frame_count mismatch.")
        _require(background_meta["height"] == bg_images.shape[1] and background_meta["width"] == bg_images.shape[2],
                 "background artifact size mismatch.")

    if person_mask_images is not None:
        person_mask_meta = metadata["src_files"]["person_mask"]
        _require(person_mask_meta["frame_count"] == person_mask_images.shape[0],
                 "person_mask artifact frame_count mismatch.")
        _require(person_mask_meta["height"] == person_mask_images.shape[1]
                 and person_mask_meta["width"] == person_mask_images.shape[2],
                 "person_mask artifact size mismatch.")
        _require(
            person_mask_meta.get("mask_semantics") == PERSON_MASK_SEMANTICS,
            "person_mask artifact semantics mismatch.",
        )
    if soft_band_images is not None:
        _require("soft_band" in metadata["src_files"], "soft_band frames were loaded but metadata has no soft_band artifact.")
        soft_band_meta = metadata["src_files"]["soft_band"]
        _require(soft_band_meta["frame_count"] == soft_band_images.shape[0], "soft_band artifact frame_count mismatch.")
        _require(
            soft_band_meta["height"] == soft_band_images.shape[1]
            and soft_band_meta["width"] == soft_band_images.shape[2],
            "soft_band artifact size mismatch.",
        )
        _require(
            soft_band_meta.get("mask_semantics") == SOFT_BAND_SEMANTICS,
            "soft_band artifact semantics mismatch.",
        )
    if hard_foreground_images is not None:
        _require("hard_foreground" in metadata["src_files"], "hard_foreground frames were loaded but metadata has no hard_foreground artifact.")
        hard_foreground_meta = metadata["src_files"]["hard_foreground"]
        _require(hard_foreground_meta["frame_count"] == hard_foreground_images.shape[0], "hard_foreground artifact frame_count mismatch.")
        _require(
            hard_foreground_meta["height"] == hard_foreground_images.shape[1]
            and hard_foreground_meta["width"] == hard_foreground_images.shape[2],
            "hard_foreground artifact size mismatch.",
        )
        _require(
            hard_foreground_meta.get("mask_semantics") == HARD_FOREGROUND_SEMANTICS,
            "hard_foreground artifact semantics mismatch.",
        )
    if soft_alpha_images is not None:
        _require("soft_alpha" in metadata["src_files"], "soft_alpha frames were loaded but metadata has no soft_alpha artifact.")
        soft_alpha_meta = metadata["src_files"]["soft_alpha"]
        _require(soft_alpha_meta["frame_count"] == soft_alpha_images.shape[0], "soft_alpha artifact frame_count mismatch.")
        _require(
            soft_alpha_meta["height"] == soft_alpha_images.shape[1]
            and soft_alpha_meta["width"] == soft_alpha_images.shape[2],
            "soft_alpha artifact size mismatch.",
        )
        _require(
            soft_alpha_meta.get("mask_semantics") == SOFT_ALPHA_SEMANTICS,
            "soft_alpha artifact semantics mismatch.",
        )
    if boundary_band_images is not None:
        _require("boundary_band" in metadata["src_files"], "boundary_band frames were loaded but metadata has no boundary_band artifact.")
        boundary_band_meta = metadata["src_files"]["boundary_band"]
        _require(boundary_band_meta["frame_count"] == boundary_band_images.shape[0], "boundary_band artifact frame_count mismatch.")
        _require(
            boundary_band_meta["height"] == boundary_band_images.shape[1]
            and boundary_band_meta["width"] == boundary_band_images.shape[2],
            "boundary_band artifact size mismatch.",
        )
        _require(
            boundary_band_meta.get("mask_semantics") == BOUNDARY_BAND_SEMANTICS,
            "boundary_band artifact semantics mismatch.",
        )
    if background_keep_prior_images is not None:
        _require(
            "background_keep_prior" in metadata["src_files"],
            "background_keep_prior frames were loaded but metadata has no background_keep_prior artifact.",
        )
        background_keep_meta = metadata["src_files"]["background_keep_prior"]
        _require(
            background_keep_meta["frame_count"] == background_keep_prior_images.shape[0],
            "background_keep_prior artifact frame_count mismatch.",
        )
        _require(
            background_keep_meta["height"] == background_keep_prior_images.shape[1]
            and background_keep_meta["width"] == background_keep_prior_images.shape[2],
            "background_keep_prior artifact size mismatch.",
        )
        _require(
            background_keep_meta.get("mask_semantics") == BACKGROUND_KEEP_PRIOR_SEMANTICS,
            "background_keep_prior artifact semantics mismatch.",
        )
    if visible_support_images is not None:
        _require("visible_support" in metadata["src_files"], "visible_support frames were loaded but metadata has no visible_support artifact.")
        visible_support_meta = metadata["src_files"]["visible_support"]
        _require(visible_support_meta["frame_count"] == visible_support_images.shape[0], "visible_support artifact frame_count mismatch.")
        _require(
            visible_support_meta["height"] == visible_support_images.shape[1]
            and visible_support_meta["width"] == visible_support_images.shape[2],
            "visible_support artifact size mismatch.",
        )
        _require(
            visible_support_meta.get("mask_semantics") == BACKGROUND_VISIBLE_SUPPORT_SEMANTICS,
            "visible_support artifact semantics mismatch.",
        )
    if unresolved_region_images is not None:
        _require("unresolved_region" in metadata["src_files"], "unresolved_region frames were loaded but metadata has no unresolved_region artifact.")
        unresolved_meta = metadata["src_files"]["unresolved_region"]
        _require(unresolved_meta["frame_count"] == unresolved_region_images.shape[0], "unresolved_region artifact frame_count mismatch.")
        _require(
            unresolved_meta["height"] == unresolved_region_images.shape[1]
            and unresolved_meta["width"] == unresolved_region_images.shape[2],
            "unresolved_region artifact size mismatch.",
        )
        _require(
            unresolved_meta.get("mask_semantics") == UNRESOLVED_REGION_SEMANTICS,
            "unresolved_region artifact semantics mismatch.",
        )
    if background_confidence_images is not None:
        _require("background_confidence" in metadata["src_files"], "background_confidence frames were loaded but metadata has no background_confidence artifact.")
        background_confidence_meta = metadata["src_files"]["background_confidence"]
        _require(background_confidence_meta["frame_count"] == background_confidence_images.shape[0], "background_confidence artifact frame_count mismatch.")
        _require(
            background_confidence_meta["height"] == background_confidence_images.shape[1]
            and background_confidence_meta["width"] == background_confidence_images.shape[2],
            "background_confidence artifact size mismatch.",
        )
        _require(
            background_confidence_meta.get("mask_semantics") == BACKGROUND_CONFIDENCE_SEMANTICS,
            "background_confidence artifact semantics mismatch.",
        )
    if background_source_provenance_images is not None:
        _require("background_source_provenance" in metadata["src_files"], "background_source_provenance frames were loaded but metadata has no background_source_provenance artifact.")
        background_provenance_meta = metadata["src_files"]["background_source_provenance"]
        _require(background_provenance_meta["frame_count"] == background_source_provenance_images.shape[0], "background_source_provenance artifact frame_count mismatch.")
        _require(
            background_provenance_meta["height"] == background_source_provenance_images.shape[1]
            and background_provenance_meta["width"] == background_source_provenance_images.shape[2],
            "background_source_provenance artifact size mismatch.",
        )
        _require(
            background_provenance_meta.get("mask_semantics") == BACKGROUND_SOURCE_PROVENANCE_SEMANTICS,
            "background_source_provenance artifact semantics mismatch.",
        )
    if occlusion_band_images is not None:
        _require("occlusion_band" in metadata["src_files"], "occlusion_band frames were loaded but metadata has no occlusion_band artifact.")
        occlusion_meta = metadata["src_files"]["occlusion_band"]
        _require(occlusion_meta["frame_count"] == occlusion_band_images.shape[0], "occlusion_band artifact frame_count mismatch.")
        _require(
            occlusion_meta["height"] == occlusion_band_images.shape[1]
            and occlusion_meta["width"] == occlusion_band_images.shape[2],
            "occlusion_band artifact size mismatch.",
        )
        _require(
            occlusion_meta.get("mask_semantics") == OCCLUSION_BAND_SEMANTICS,
            "occlusion_band artifact semantics mismatch.",
        )
    if uncertainty_map_images is not None:
        _require("uncertainty_map" in metadata["src_files"], "uncertainty_map frames were loaded but metadata has no uncertainty_map artifact.")
        uncertainty_meta = metadata["src_files"]["uncertainty_map"]
        _require(uncertainty_meta["frame_count"] == uncertainty_map_images.shape[0], "uncertainty_map artifact frame_count mismatch.")
        _require(
            uncertainty_meta["height"] == uncertainty_map_images.shape[1]
            and uncertainty_meta["width"] == uncertainty_map_images.shape[2],
            "uncertainty_map artifact size mismatch.",
        )
        _require(
            uncertainty_meta.get("mask_semantics") == UNCERTAINTY_MAP_SEMANTICS,
            "uncertainty_map artifact semantics mismatch.",
        )
    for key, value, semantics in (
        ("face_boundary", face_boundary_images, FACE_BOUNDARY_SEMANTICS),
        ("hair_boundary", hair_boundary_images, HAIR_BOUNDARY_SEMANTICS),
        ("hand_boundary", hand_boundary_images, HAND_BOUNDARY_SEMANTICS),
        ("cloth_boundary", cloth_boundary_images, CLOTH_BOUNDARY_SEMANTICS),
        ("occluded_boundary", occluded_boundary_images, OCCLUDED_BOUNDARY_SEMANTICS),
    ):
        if value is not None:
            _require(key in metadata["src_files"], f"{key} frames were loaded but metadata has no {key} artifact.")
            artifact_meta = metadata["src_files"][key]
            _require(artifact_meta["frame_count"] == value.shape[0], f"{key} artifact frame_count mismatch.")
            _require(
                artifact_meta["height"] == value.shape[1] and artifact_meta["width"] == value.shape[2],
                f"{key} artifact size mismatch.",
            )
            _require(
                artifact_meta.get("mask_semantics") == semantics,
                f"{key} artifact semantics mismatch.",
            )
