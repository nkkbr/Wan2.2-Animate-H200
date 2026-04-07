import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_DIR = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess"

DEFAULT_VIDEO_PATH = "/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4"
DEFAULT_CKPT_PATH = "/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B/process_checkpoint/sam2/sam2_hiera_large.pt"


def _signal_name(returncode: int):
    if returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except Exception:
        return f"SIG{-returncode}"


def _worker():
    os.chdir(PREPROCESS_DIR)
    sys.path.insert(0, str(PREPROCESS_DIR))

    import cv2
    import numpy as np
    from decord import VideoReader

    from sam_runtime import apply_sam_runtime_profile, resolve_sam_runtime_profile

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--profile", type=str, required=True)
    parser.add_argument("--backend", type=str, choices=["custom", "upstream"], required=True)
    parser.add_argument("--interaction", type=str, choices=["mask", "points"], required=True)
    parser.add_argument("--num_frames", type=int, default=2)
    parser.add_argument("--apply_postprocessing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload_video_to_cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload_state_to_cpu", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    apply_sam_runtime_profile(resolve_sam_runtime_profile(args.profile))

    if args.backend == "custom":
        from sam_utils import build_sam2_video_predictor
    else:
        from sam2.build_sam import build_sam2_video_predictor

    vr = VideoReader(args.video_path)
    frames = vr.get_batch(list(range(args.num_frames))).asnumpy()
    predictor = build_sam2_video_predictor(
        "sam2_hiera_l.yaml",
        args.ckpt_path,
        apply_postprocessing=args.apply_postprocessing,
    )

    if args.backend == "custom":
        state = predictor.init_state_v2(
            frames=frames,
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="sam2_repro_frames_") as jpg_dir:
            jpg_dir = Path(jpg_dir)
            for index, frame in enumerate(frames):
                path = jpg_dir / f"{index}.jpg"
                cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            state = predictor.init_state(
                str(jpg_dir),
                offload_video_to_cpu=args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
            )

    height, width = frames[0].shape[:2]
    print(
        json.dumps(
            {
                "stage": "before_interaction",
                "backend": args.backend,
                "interaction": args.interaction,
                "frame_shape": [int(height), int(width), int(frames[0].shape[2])],
                "num_frames": int(args.num_frames),
                "profile": args.profile,
                "apply_postprocessing": bool(args.apply_postprocessing),
                "offload_video_to_cpu": bool(args.offload_video_to_cpu),
                "offload_state_to_cpu": bool(args.offload_state_to_cpu),
            }
        ),
        flush=True,
    )

    if args.interaction == "mask":
        mask = np.zeros((height, width), dtype=np.uint8)
        x1 = max(0, width // 2 - max(8, width // 12))
        x2 = min(width, width // 2 + max(8, width // 12))
        y1 = max(0, height // 2 - max(8, height // 6))
        y2 = min(height, height // 2 + max(8, height // 6))
        mask[y1:y2, x1:x2] = 1
        predictor.add_new_mask(state, frame_idx=0, obj_id=0, mask=mask)
    else:
        points = [[width // 2, height // 2]]
        labels = [1]
        predictor.add_new_points(state, frame_idx=0, obj_id=0, points=points, labels=labels)

    print(json.dumps({"stage": "after_interaction"}), flush=True)


def _outer():
    parser = argparse.ArgumentParser(description="Minimal subprocess-based SAM2 interaction crash repro.")
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--profile", type=str, default="h200_safe", choices=["legacy_safe", "h200_safe", "h200_aggressive"])
    parser.add_argument("--backend", type=str, default="upstream", choices=["custom", "upstream"])
    parser.add_argument("--interaction", type=str, default="mask", choices=["mask", "points"])
    parser.add_argument("--num_frames", type=int, default=2)
    parser.add_argument("--apply_postprocessing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload_video_to_cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload_state_to_cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--worker", action="store_true", default=False)
    args = parser.parse_args()

    if args.worker:
        _worker()
        return

    if args.summary_path is None:
        summary_path = REPO_ROOT / "runs" / f"sam2_repro_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    else:
        summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-X",
        "faulthandler",
        str(Path(__file__).resolve()),
        "--worker",
        "--video_path",
        args.video_path,
        "--ckpt_path",
        args.ckpt_path,
        "--profile",
        args.profile,
        "--backend",
        args.backend,
        "--interaction",
        args.interaction,
        "--num_frames",
        str(args.num_frames),
    ]
    command.append("--apply_postprocessing" if args.apply_postprocessing else "--no-apply_postprocessing")
    command.append("--offload_video_to_cpu" if args.offload_video_to_cpu else "--no-offload_video_to_cpu")
    command.append("--offload_state_to_cpu" if args.offload_state_to_cpu else "--no-offload_state_to_cpu")

    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONFAULTHANDLER": "1"},
    )
    summary = {
        "video_path": args.video_path,
        "ckpt_path": args.ckpt_path,
        "profile": args.profile,
        "backend": args.backend,
        "interaction": args.interaction,
        "num_frames": int(args.num_frames),
        "apply_postprocessing": bool(args.apply_postprocessing),
        "offload_video_to_cpu": bool(args.offload_video_to_cpu),
        "offload_state_to_cpu": bool(args.offload_state_to_cpu),
        "returncode": int(result.returncode),
        "signal": _signal_name(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
        "command": command,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "passed": summary["passed"], "signal": summary["signal"]}, ensure_ascii=False))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    _outer()
