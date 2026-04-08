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
SOFT_ALPHA_SEMANTICS = "soft_alpha"
SOFT_BAND_SEMANTICS = "boundary_transition_band"
BOUNDARY_BAND_SEMANTICS = SOFT_BAND_SEMANTICS
OCCLUSION_BAND_SEMANTICS = "occlusion_band"
UNCERTAINTY_MAP_SEMANTICS = "uncertainty_map"
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
                background_mode in {"hole", "clean_plate_image", "clean_plate_video"},
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
            "occlusion_band",
            "uncertainty_map",
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
    occlusion_band_images: np.ndarray | None = None,
    uncertainty_map_images: np.ndarray | None = None,
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
