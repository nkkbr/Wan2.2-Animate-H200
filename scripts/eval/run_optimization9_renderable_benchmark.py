#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.media_io import load_mask_artifact, load_rgb_artifact, write_output_frames


RUNS_ROOT = REPO_ROOT / "runs"
REVIEWED_DATASET_DIR = REPO_ROOT / "runs" / "optimization8_step01_round3" / "reviewed_edge_benchmark_v3_candidate"
BASELINE_PREPROCESS_DIR = REPO_ROOT / "runs" / "optimization5_step03_round1_preprocess" / "preprocess"
SOURCE_VIDEO_PATH = Path("/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4")
COMPUTE_REVIEWED_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_reviewed_edge_metrics_v3.py"
COMPUTE_REPLACEMENT_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_replacement_metrics.py"
COMPUTE_RENDERABLE_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_renderable_foreground_metrics.py"
INFER_SCRIPT = REPO_ROOT / "scripts" / "eval" / "infer_renderable_foreground_proto.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval" / "evaluate_optimization9_renderable_benchmark.py"

ROUND_CONFIGS = {
    1: {"mode": "round1"},
    2: {"mode": "round2"},
    3: {"mode": "round3"},
}


def _default_python_bin() -> str:
    override = os.environ.get("WAN_PYTHON")
    for candidate in [override, "/home/user1/miniconda3/envs/wan/bin/python", sys.executable]:
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())
    return sys.executable


PYTHON = _default_python_bin()


def _run(command: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)


def _write_log(directory: Path, stem: str, result: subprocess.CompletedProcess[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (directory / f"{stem}.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{stem} failed with exit code {result.returncode}")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_video_frames(video_path: Path, frame_count: int, shape_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or frame_count)
    if total_frames <= 0:
        total_frames = frame_count
    frame_indices = np.linspace(0, max(total_frames - 1, 0), num=frame_count).round().astype(int)
    frames = []
    target_h, target_w = shape_hw
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0).astype(np.uint8)


def _write_proxy_smoke(*, suite_dir: Path, name: str, src_root_path: Path, source_video_path: Path, logs_dir: Path) -> tuple[Path, Path]:
    output_dir = suite_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / "replacement_best.mkv"

    metadata = _read_json(src_root_path / "metadata.json")
    artifacts = metadata["src_files"]
    background_rgb = load_rgb_artifact(src_root_path / artifacts["background_rgb"]["path"], artifacts["background_rgb"].get("format")).astype(np.uint8)
    foreground_rgb = load_rgb_artifact(src_root_path / artifacts["foreground_rgb"]["path"], artifacts["foreground_rgb"].get("format")).astype(np.float32)
    foreground_alpha = load_mask_artifact(src_root_path / artifacts["foreground_alpha"]["path"], artifacts["foreground_alpha"].get("format")).astype(np.float32)
    frame_count = len(foreground_alpha)
    shape_hw = foreground_alpha.shape[1:3]
    source_rgb = _load_video_frames(source_video_path, frame_count, shape_hw)

    if "renderable_composite_preview" in artifacts:
        proxy = load_rgb_artifact(src_root_path / artifacts["renderable_composite_preview"]["path"], artifacts["renderable_composite_preview"].get("format")).astype(np.uint8)[:frame_count]
        mode = "renderable_composite_preview"
    else:
        proxy = np.clip(
            foreground_rgb + background_rgb.astype(np.float32) * (1.0 - foreground_alpha[..., None]),
            0.0,
            255.0,
        ).astype(np.uint8)
        mode = "renderable_fg_composite"

    write_output_frames(proxy, save_file, fps=float(metadata.get("fps") or 5.0), output_format="ffv1")
    (logs_dir / f"{name}_proxy.stdout.log").write_text(
        json.dumps(
            {
                "src_root_path": str(src_root_path.resolve()),
                "source_video_path": str(source_video_path.resolve()),
                "frame_count": int(frame_count),
                "mode": mode,
                "source_video_shape": list(source_rgb.shape),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (logs_dir / f"{name}_proxy.stderr.log").write_text("", encoding="utf-8")

    metrics_dir = suite_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{name}_replacement_metrics.json"
    person_mask_path = src_root_path / artifacts["person_mask"]["path"]
    repl = _run(
        [
            PYTHON,
            str(COMPUTE_REPLACEMENT_METRICS),
            "--video_path",
            str(save_file),
            "--mask_path",
            str(person_mask_path),
            "--clip_len",
            "45",
            "--refert_num",
            "5",
            "--output_json",
            str(metrics_path),
        ]
    )
    _write_log(logs_dir, f"{name}_replacement_metrics", repl)
    return save_file, metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optimization9 Step05 renderable foreground benchmark.")
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--reviewed_dataset_dir", type=str, default=str(REVIEWED_DATASET_DIR))
    parser.add_argument("--src_root_path", type=str, default=str(BASELINE_PREPROCESS_DIR))
    parser.add_argument("--source_video_path", type=str, default=str(SOURCE_VIDEO_PATH))
    args = parser.parse_args()

    config = ROUND_CONFIGS[int(args.round)]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"optimization9_step05_round{args.round}_{timestamp}"
    suite_dir = RUNS_ROOT / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = suite_dir / "logs"

    baseline_metrics = suite_dir / "baseline_metrics_v3.json"
    reviewed_base = _run([PYTHON, str(COMPUTE_REVIEWED_METRICS), "--dataset_dir", str(Path(args.reviewed_dataset_dir).resolve()), "--prediction_preprocess_dir", str(Path(args.src_root_path).resolve()), "--output_json", str(baseline_metrics)])
    _write_log(logs_dir, "baseline_reviewed_metrics", reviewed_base)

    candidate_preprocess = suite_dir / "preprocess_renderable"
    infer = _run([PYTHON, str(INFER_SCRIPT), "--src_root_path", str(Path(args.src_root_path).resolve()), "--source_video_path", str(Path(args.source_video_path).resolve()), "--output_root_path", str(candidate_preprocess), "--mode", config["mode"]])
    _write_log(logs_dir, "infer_renderable_proto", infer)

    candidate_metrics = suite_dir / "candidate_metrics_v3.json"
    reviewed = _run([PYTHON, str(COMPUTE_REVIEWED_METRICS), "--dataset_dir", str(Path(args.reviewed_dataset_dir).resolve()), "--prediction_preprocess_dir", str(candidate_preprocess), "--output_json", str(candidate_metrics)])
    _write_log(logs_dir, "candidate_reviewed_metrics", reviewed)

    baseline_video, baseline_repl = _write_proxy_smoke(suite_dir=suite_dir, name="baseline_smoke", src_root_path=Path(args.src_root_path).resolve(), source_video_path=Path(args.source_video_path).resolve(), logs_dir=logs_dir)
    candidate_video, candidate_repl = _write_proxy_smoke(suite_dir=suite_dir, name="candidate_smoke", src_root_path=candidate_preprocess, source_video_path=Path(args.source_video_path).resolve(), logs_dir=logs_dir)

    renderable_metrics = suite_dir / "renderable_vs_baseline_metrics.json"
    renderable = _run([PYTHON, str(COMPUTE_RENDERABLE_METRICS), "--before", str(baseline_video), "--after", str(candidate_video), "--baseline_root_path", str(Path(args.src_root_path).resolve()), "--src_root_path", str(candidate_preprocess), "--output_json", str(renderable_metrics)])
    _write_log(logs_dir, "renderable_metrics", renderable)

    gate_json = suite_dir / "gate_result.json"
    evaluate = _run([PYTHON, str(EVAL_SCRIPT), "--baseline_metrics", str(baseline_metrics), "--candidate_metrics", str(candidate_metrics), "--baseline_replacement_metrics", str(baseline_repl), "--candidate_replacement_metrics", str(candidate_repl), "--renderable_metrics", str(renderable_metrics), "--output_json", str(gate_json)])
    _write_log(logs_dir, "evaluate_gate", evaluate)

    summary = {
        "suite_dir": str(suite_dir.resolve()),
        "round": int(args.round),
        "config": config,
        "baseline_metrics": str(baseline_metrics.resolve()),
        "candidate_metrics": str(candidate_metrics.resolve()),
        "baseline_replacement_metrics": str(baseline_repl.resolve()),
        "candidate_replacement_metrics": str(candidate_repl.resolve()),
        "renderable_metrics": str(renderable_metrics.resolve()),
        "gate_result": str(gate_json.resolve()),
    }
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
