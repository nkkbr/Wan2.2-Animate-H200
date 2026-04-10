#!/usr/bin/env python
import argparse
import cv2
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
ROI_DATASET_DIR = REPO_ROOT / "runs" / "optimization8_step02_round3" / "roi_dataset_v2"
STABLE_PREPROCESS_DIR = REPO_ROOT / "runs" / "optimization3_step06_round5_ab" / "preprocess_video_v2" / "preprocess"
BASELINE_REVIEWED_METRICS = REPO_ROOT / "runs" / "optimization8_step01_round3" / "baseline_metrics_v3.json"
SOURCE_VIDEO_PATH = Path("/home/user1/wan2.2/test1/260306_01_step_1_0000_0010.mp4")
COMPUTE_REVIEWED_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_reviewed_edge_metrics_v3.py"
COMPUTE_REPLACEMENT_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_replacement_metrics.py"
COMPUTE_BOUNDARY_ROI_METRICS = REPO_ROOT / "scripts" / "eval" / "compute_boundary_roi_metrics.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "eval" / "train_matte_bridge_model.py"
INFER_SCRIPT = REPO_ROOT / "scripts" / "eval" / "infer_matte_bridge_model.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval" / "evaluate_optimization9_bridge_benchmark.py"


ROUND_CONFIGS = {
    1: {
        "width": 24,
        "epochs": 1,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "gate_strength": 1.00,
        "bridge_error_threshold": 0.04,
        "final_weight": 1.0,
        "pred_weight": 0.35,
        "gate_weight": 0.20,
        "composite_weight": 0.0,
        "gradient_weight": 0.0,
        "contrast_weight": 0.0,
        "hair_weight": 2.5,
        "boundary_weight": 4.0,
        "trimap_weight": 2.0,
        "semi_weight": 2.0,
        "hard_negative_weight": 1.5,
        "gate_target_mode": "binary",
    },
    2: {
        "width": 32,
        "epochs": 2,
        "batch_size": 64,
        "learning_rate": 8e-4,
        "gate_strength": 1.15,
        "bridge_error_threshold": 0.03,
        "final_weight": 1.0,
        "pred_weight": 0.50,
        "gate_weight": 0.35,
        "composite_weight": 0.35,
        "gradient_weight": 0.20,
        "contrast_weight": 0.12,
        "hair_weight": 4.0,
        "boundary_weight": 4.0,
        "trimap_weight": 2.0,
        "semi_weight": 2.0,
        "hard_negative_weight": 1.5,
        "gate_target_mode": "binary",
    },
    3: {
        "width": 32,
        "epochs": 2,
        "batch_size": 64,
        "learning_rate": 7e-4,
        "gate_strength": 0.95,
        "bridge_error_threshold": 0.025,
        "final_weight": 1.0,
        "pred_weight": 0.55,
        "gate_weight": 0.30,
        "composite_weight": 0.45,
        "gradient_weight": 0.35,
        "contrast_weight": 0.22,
        "hair_weight": 4.5,
        "boundary_weight": 2.5,
        "trimap_weight": 1.25,
        "semi_weight": 2.5,
        "hard_negative_weight": 1.25,
        "gate_target_mode": "continuous",
    },
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


def _mask_path_from_preprocess(src_root: Path) -> Path:
    metadata = _read_json(src_root / "metadata.json")
    artifact = metadata["src_files"]["person_mask"]
    return src_root / artifact["path"]


def _load_video_frames(video_path: Path, frame_count: int, shape_hw: tuple[int, int]) -> np.ndarray:
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


def _write_proxy_smoke(
    *,
    suite_dir: Path,
    name: str,
    src_root_path: Path,
    source_video_path: Path,
    logs_dir: Path,
    frame_num: int,
) -> tuple[Path, Path]:
    output_dir = suite_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / "replacement_best.mkv"

    metadata = _read_json(src_root_path / "metadata.json")
    background_artifact = metadata["src_files"]["background"]
    alpha_artifact = metadata["src_files"]["soft_alpha"]
    background = load_rgb_artifact(src_root_path / background_artifact["path"], background_artifact.get("format")).astype(np.uint8)
    alpha = load_mask_artifact(src_root_path / alpha_artifact["path"], alpha_artifact.get("format")).astype(np.float32)
    frame_count = min(int(frame_num), len(background), len(alpha))
    video_rgb = _load_video_frames(source_video_path, frame_count, background.shape[1:3])
    proxy = np.clip(
        video_rgb[:frame_count].astype(np.float32) * alpha[:frame_count][..., None]
        + background[:frame_count].astype(np.float32) * (1.0 - alpha[:frame_count][..., None]),
        0.0,
        255.0,
    ).astype(np.uint8)
    write_output_frames(proxy, save_file, fps=float(metadata.get("fps") or 5.0), output_format="ffv1")
    (logs_dir / f"{name}_proxy.stdout.log").write_text(
        json.dumps(
            {
                "src_root_path": str(src_root_path.resolve()),
                "source_video_path": str(source_video_path.resolve()),
                "frame_count": int(frame_count),
                "mode": "proxy_composite_smoke",
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
    repl = _run(
        [
            PYTHON,
            str(COMPUTE_REPLACEMENT_METRICS),
            "--video_path",
            str(save_file),
            "--mask_path",
            str(_mask_path_from_preprocess(src_root_path)),
            "--clip_len",
            str(min(frame_num, 45)),
            "--refert_num",
            "5",
            "--output_json",
            str(metrics_path),
        ]
    )
    _write_log(logs_dir, f"{name}_replacement_metrics", repl)
    return save_file, metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optimization9 Step02 matte bridge benchmark.")
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--dataset_npz", type=str, default=str(ROI_DATASET_DIR / "roi_dataset_v2.npz"))
    parser.add_argument("--dataset_json", type=str, default=str(ROI_DATASET_DIR / "roi_dataset_v2.json"))
    parser.add_argument("--reviewed_dataset_dir", type=str, default=str(REVIEWED_DATASET_DIR))
    parser.add_argument("--baseline_metrics", type=str, default=str(BASELINE_REVIEWED_METRICS))
    parser.add_argument("--src_root_path", type=str, default=str(STABLE_PREPROCESS_DIR))
    parser.add_argument("--source_video_path", type=str, default=str(SOURCE_VIDEO_PATH))
    parser.add_argument("--frame_num", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = ROUND_CONFIGS[int(args.round)]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"optimization9_step02_round{args.round}_{timestamp}"
    suite_dir = RUNS_ROOT / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = suite_dir / "logs"
    train_dir = suite_dir / "train"
    candidate_preprocess = suite_dir / "preprocess_bridge"

    train_cmd = [
        PYTHON,
        str(TRAIN_SCRIPT),
        "--dataset_npz",
        str(Path(args.dataset_npz).resolve()),
        "--dataset_json",
        str(Path(args.dataset_json).resolve()),
        "--output_dir",
        str(train_dir),
        "--width",
        str(config["width"]),
        "--epochs",
        str(config["epochs"]),
        "--batch_size",
        str(config["batch_size"]),
        "--max_train_samples",
        "32",
        "--max_val_samples",
        "16",
        "--learning_rate",
        str(config["learning_rate"]),
        "--gate_strength",
        str(config["gate_strength"]),
        "--bridge_error_threshold",
        str(config["bridge_error_threshold"]),
        "--final_weight",
        str(config["final_weight"]),
        "--pred_weight",
        str(config["pred_weight"]),
        "--gate_weight",
        str(config["gate_weight"]),
        "--composite_weight",
        str(config["composite_weight"]),
        "--gradient_weight",
        str(config["gradient_weight"]),
        "--contrast_weight",
        str(config["contrast_weight"]),
        "--boundary_weight",
        str(config["boundary_weight"]),
        "--trimap_weight",
        str(config["trimap_weight"]),
        "--semi_weight",
        str(config["semi_weight"]),
        "--hair_weight",
        str(config["hair_weight"]),
        "--hard_negative_weight",
        str(config["hard_negative_weight"]),
        "--gate_target_mode",
        str(config["gate_target_mode"]),
        "--device",
        args.device,
    ]
    train = _run(train_cmd)
    _write_log(logs_dir, "train_bridge", train)

    infer = _run(
        [
            PYTHON,
            str(INFER_SCRIPT),
            "--checkpoint",
            str(train_dir / "matte_bridge_model.pt"),
            "--src_root_path",
            str(Path(args.src_root_path).resolve()),
            "--source_video_path",
            str(Path(args.source_video_path).resolve()),
            "--output_root_path",
            str(candidate_preprocess),
            "--device",
            args.device,
        ]
    )
    _write_log(logs_dir, "infer_bridge", infer)

    candidate_metrics = suite_dir / "candidate_metrics_v3.json"
    reviewed = _run(
        [
            PYTHON,
            str(COMPUTE_REVIEWED_METRICS),
            "--dataset_dir",
            str(Path(args.reviewed_dataset_dir).resolve()),
            "--prediction_preprocess_dir",
            str(candidate_preprocess),
            "--output_json",
            str(candidate_metrics),
        ]
    )
    _write_log(logs_dir, "reviewed_metrics", reviewed)

    baseline_video, baseline_repl = _write_proxy_smoke(
        suite_dir=suite_dir,
        name="baseline_smoke",
        src_root_path=Path(args.src_root_path).resolve(),
        source_video_path=Path(args.source_video_path).resolve(),
        logs_dir=logs_dir,
        frame_num=int(args.frame_num),
    )
    candidate_video, candidate_repl = _write_proxy_smoke(
        suite_dir=suite_dir,
        name="candidate_smoke",
        src_root_path=candidate_preprocess,
        source_video_path=Path(args.source_video_path).resolve(),
        logs_dir=logs_dir,
        frame_num=int(args.frame_num),
    )

    roi_metrics = suite_dir / "bridge_vs_baseline_roi_metrics.json"
    roi = _run(
        [
            PYTHON,
            str(COMPUTE_BOUNDARY_ROI_METRICS),
            "--before",
            str(baseline_video),
            "--after",
            str(candidate_video),
            "--src_root_path",
            str(candidate_preprocess),
            "--debug_dir",
            str(suite_dir / "candidate_smoke" / "debug"),
            "--output_json",
            str(roi_metrics),
        ]
    )
    _write_log(logs_dir, "bridge_roi_metrics", roi)

    gate_path = suite_dir / "gate_result.json"
    evaluate = _run(
        [
            PYTHON,
            str(EVAL_SCRIPT),
            "--baseline_metrics",
            str(Path(args.baseline_metrics).resolve()),
            "--candidate_metrics",
            str(candidate_metrics),
            "--baseline_replacement_metrics",
            str(baseline_repl),
            "--candidate_replacement_metrics",
            str(candidate_repl),
            "--roi_metrics",
            str(roi_metrics),
            "--output_json",
            str(gate_path),
        ]
    )
    _write_log(logs_dir, "bridge_gate", evaluate)

    summary = {
        "suite_dir": str(suite_dir.resolve()),
        "round": int(args.round),
        "config": config,
        "train_summary": str((train_dir / "train_summary.json").resolve()),
        "candidate_preprocess": str(candidate_preprocess.resolve()),
        "candidate_metrics": str(candidate_metrics.resolve()),
        "baseline_replacement_metrics": str(baseline_repl.resolve()),
        "candidate_replacement_metrics": str(candidate_repl.resolve()),
        "roi_metrics": str(roi_metrics.resolve()),
        "gate_result": str(gate_path.resolve()),
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
