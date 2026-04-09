#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.modules.animate.preprocess.external_alpha_base import build_external_alpha_adapter
from wan.utils.external_alpha_registry import get_external_model_entry
from wan.utils.media_io import load_mask_artifact, load_person_mask_artifact, load_rgb_artifact


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_rgb(frame: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _read_video_frame(path: Path, frame_index: int, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if shape_hw is not None:
        frame = _resize_rgb(frame, shape_hw)
    return frame


def _load_initial_mask_volume(preprocess_dir: Path, metadata: dict) -> np.ndarray:
    hard_foreground = metadata["src_files"].get("hard_foreground")
    if hard_foreground is not None:
        return load_mask_artifact(
            preprocess_dir / hard_foreground["path"],
            hard_foreground.get("format"),
        )
    person_mask = metadata["src_files"]["person_mask"]
    return load_person_mask_artifact(
        preprocess_dir / person_mask["path"],
        person_mask.get("format"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--background_preprocess_dir", required=True)
    parser.add_argument("--registry_path", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    summary = _read_json(dataset_dir / "summary.json")
    record = summary["cases"][0]["records"][0]
    case = summary["cases"][0]
    label_json = _read_json(Path(record["label_json_path"]))
    preprocess_idx = int(label_json["preprocess_frame_index"])
    image_rgb = _read_rgb(Path(record["image_path"]))
    model_entry = get_external_model_entry(args.model_id, path=args.registry_path)

    meta = _read_json(Path(args.background_preprocess_dir) / "metadata.json")
    bg_artifact = meta["src_files"]["background"]
    bg_volume = load_rgb_artifact(
        Path(args.background_preprocess_dir) / bg_artifact["path"],
        bg_artifact.get("format"),
    )
    background_rgb = bg_volume[preprocess_idx]
    height, width = image_rgb.shape[:2]
    scaled_height = max(384, int(round(height * 0.75 / 32.0) * 32))
    scaled_width = max(384, int(round(width * 0.75 / 32.0) * 32))

    adapter = build_external_alpha_adapter(
        model_id=args.model_id,
        registry_path=args.registry_path,
    )
    if model_entry.get("task_type") == "video_matting_with_first_frame_mask":
        mask_volume = _load_initial_mask_volume(Path(args.background_preprocess_dir), meta)
        first_mask = (mask_volume[0] * 255.0).astype(np.uint8)
        adapter.reset_sequence_state()
        adapter.set_sequence_context(initial_mask=first_mask)
        source_stride = 1
        if len(case["records"]) > 1:
            probe = case["records"][1]
            denom = max(int(probe["preprocess_frame_index"]), 1)
            source_stride = max(1, int(round(int(probe["source_frame_index"]) / denom)))
        second_frame = _read_video_frame(
            Path(case["video_path"]),
            source_stride,
            shape_hw=image_rgb.shape[:2],
        )
        output = adapter.infer(image_rgb, None)
        resized_first_mask = _resize_mask(first_mask, (scaled_height, scaled_width))
        adapter.reset_sequence_state()
        adapter.set_sequence_context(initial_mask=resized_first_mask)
        alpha_small = adapter.infer(_resize_rgb(second_frame, (scaled_height, scaled_width)), None)
    else:
        output = adapter.infer(image_rgb, background_rgb)
        resized_image = _resize_rgb(image_rgb, (scaled_height, scaled_width))
        resized_background = _resize_rgb(background_rgb, (scaled_height, scaled_width))
        adapter.reset_sequence_state()
        alpha_small = adapter.infer(resized_image, resized_background)

    alpha = output["alpha"]

    ok = (
        alpha.shape == image_rgb.shape[:2]
        and alpha_small["alpha"].shape == (scaled_height, scaled_width)
        and np.isfinite(alpha).all()
        and np.isfinite(alpha_small["alpha"]).all()
        and 0.0 <= float(alpha.min()) <= float(alpha.max()) <= 1.0
        and 0.0 <= float(alpha_small["alpha"].min()) <= float(alpha_small["alpha"].max()) <= 1.0
        and output.get("soft_alpha_ext") is not None
        and output.get("trimap_unknown_ext") is not None
        and output.get("hair_alpha_ext") is not None
        and output.get("alpha_confidence_ext") is not None
        and output.get("alpha_source_provenance_ext") is not None
    )
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "alpha_shape": list(alpha.shape),
                "alpha_min": float(alpha.min()),
                "alpha_max": float(alpha.max()),
                "resized_alpha_shape": list(alpha_small["alpha"].shape),
                "runtime_sec": float(output["runtime_sec"]),
                "provenance_keys_present": bool(output.get("sha256") and output.get("license")),
                "synthetic_passed": bool(ok),
            },
            ensure_ascii=False,
        )
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
