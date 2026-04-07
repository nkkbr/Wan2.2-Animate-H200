import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_DIR = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from sam_prompting import (
    build_mask_stats,
    build_prompt_for_frame,
    compute_prompt_frames,
    make_mask_overlay,
    make_sam_prompts_overlay,
    plan_chunk_prompts,
    write_prompt_keyframes,
)


def _make_meta(width, height, shift_x=0.0):
    body = np.zeros((20, 3), dtype=np.float32)
    body_points = {
        0: (0.50 + shift_x, 0.18),
        1: (0.50 + shift_x, 0.28),
        2: (0.42 + shift_x, 0.30),
        3: (0.36 + shift_x, 0.40),
        4: (0.30 + shift_x, 0.52),
        5: (0.58 + shift_x, 0.30),
        6: (0.64 + shift_x, 0.40),
        7: (0.70 + shift_x, 0.52),
        8: (0.45 + shift_x, 0.52),
        9: (0.44 + shift_x, 0.70),
        10: (0.43 + shift_x, 0.88),
        11: (0.55 + shift_x, 0.52),
        12: (0.56 + shift_x, 0.70),
        13: (0.57 + shift_x, 0.88),
    }
    for index, (x, y) in body_points.items():
        body[index] = [x, y, 0.95]

    face = np.zeros((69, 3), dtype=np.float32)
    for index in range(69):
        angle = 2 * np.pi * index / 69.0
        x = 0.50 + shift_x + 0.035 * np.cos(angle)
        y = 0.18 + 0.05 * np.sin(angle)
        face[index] = [x, y, 0.92]

    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)
    for index in range(21):
        left_hand[index] = [0.28 + shift_x + 0.015 * (index % 4), 0.52 + 0.01 * (index // 4), 0.88]
        right_hand[index] = [0.68 + shift_x + 0.015 * (index % 4), 0.52 + 0.01 * (index // 4), 0.88]

    return {
        "width": width,
        "height": height,
        "keypoints_body": body,
        "keypoints_face": face,
        "keypoints_left_hand": left_hand,
        "keypoints_right_hand": right_hand,
    }


def main():
    width, height = 640, 360
    frame_count = 30
    metas = [_make_meta(width, height, shift_x=0.001 * index) for index in range(frame_count)]

    prompt_frames, prompt_tags, reprompt_frames = compute_prompt_frames(
        frame_count=frame_count,
        keyframes_per_chunk=6,
        reprompt_interval=10,
    )
    assert 0 in prompt_frames and frame_count - 1 in prompt_frames
    assert len(prompt_frames) >= 6
    assert any("reprompt" in prompt_tags[idx] for idx in reprompt_frames if idx in prompt_tags)

    single_prompt = build_prompt_for_frame(
        metas[0],
        image_shape=(height, width),
        use_negative_points=True,
        negative_margin=0.08,
    )
    assert single_prompt["positive_count"] >= 10
    assert single_prompt["negative_count"] >= 4
    bbox = single_prompt["person_bbox"]
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    for x, y in single_prompt["negative_points"].tolist():
        assert x < x1 or x > x2 or y < y1 or y > y2

    plan = plan_chunk_prompts(
        metas,
        image_shape=(height, width),
        keyframes_per_chunk=6,
        reprompt_interval=10,
        use_negative_points=True,
        negative_margin=0.08,
    )
    assert len(plan["prompt_entries"]) >= len(prompt_frames) - 1

    frames = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    frames[:, :, :, 1] = np.linspace(32, 128, frame_count, dtype=np.uint8)[:, None, None]
    masks = np.zeros((frame_count, height, width), dtype=np.uint8)
    for index in range(frame_count):
        x_start = 220 + index
        masks[index, 80:320, x_start:x_start + 140] = 1

    prompt_entries = []
    for entry in plan["prompt_entries"]:
        global_entry = dict(entry)
        global_entry["global_frame_idx"] = int(entry["frame_idx"])
        prompt_entries.append(global_entry)
    chunk_plans = [
        {
            "chunk_index": 0,
            "start_frame": 0,
            "end_frame": frame_count - 1,
            "prompt_entries": prompt_entries,
            "reprompt_frames": plan["reprompt_frames"],
        }
    ]

    stats = build_mask_stats(
        masks=masks,
        chunk_plans=chunk_plans,
        sam_chunk_len=120,
        sam_keyframes_per_chunk=6,
        sam_reprompt_interval=10,
        sam_use_negative_points=True,
        sam_negative_margin=0.08,
    )
    assert stats["frame_count"] == frame_count
    assert stats["total_prompt_frames"] == len(prompt_entries)
    assert len(stats["mask_area_ratio"]) == frame_count

    overlay = make_mask_overlay(frames, masks, prompt_entries)
    prompts_overlay = make_sam_prompts_overlay(frames, prompt_entries)
    assert overlay.shape == frames.shape
    assert prompts_overlay.shape == frames.shape
    assert overlay.dtype == np.uint8
    assert prompts_overlay.dtype == np.uint8

    with tempfile.TemporaryDirectory() as tmpdir:
        written = write_prompt_keyframes(tmpdir, frames, prompt_entries)
        keyframe_dir = Path(tmpdir) / written["path"]
        assert keyframe_dir.exists()
        assert len(list(keyframe_dir.glob("*.png"))) == len(prompt_entries)

    print("Synthetic SAM2 prompt planning: PASS")
    print("Synthetic SAM2 QA artifacts: PASS")


if __name__ == "__main__":
    main()
