import argparse
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch


MANIFEST_VERSION = 1


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_run_name(name: str) -> str:
    keep = []
    for char in name.strip():
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    sanitized = "".join(keep).strip("._")
    return sanitized or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / ".git").exists():
            return path
    return current


def _serialize(value):
    if isinstance(value, argparse.Namespace):
        return _serialize(vars(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _run_git(args, repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    output = result.stdout.strip()
    return output or None


def get_repo_info(repo_root: Path) -> dict:
    commit = _run_git(["rev-parse", "HEAD"], repo_root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    status = _run_git(["status", "--short"], repo_root) or ""
    return {
        "root": str(repo_root),
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(status.strip()),
    }


def get_hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["gpus"].append({
                "index": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "multi_processor_count": props.multi_processor_count,
            })
    return info


def create_run_layout(
    run_name: str | None = None,
    run_dir: str | None = None,
    repo_root: Path | None = None,
) -> dict:
    repo_root = (repo_root or find_repo_root()).resolve()
    if run_dir is None:
        if run_name is None:
            raise ValueError("run_name or run_dir must be provided")
        run_dir_path = repo_root / "runs" / sanitize_run_name(run_name)
    else:
        run_dir_path = Path(run_dir)
        if not run_dir_path.is_absolute():
            run_dir_path = (repo_root / run_dir_path).resolve()
    if run_name is None:
        run_name = run_dir_path.name
    layout = {
        "run_name": sanitize_run_name(run_name),
        "run_dir": run_dir_path,
        "preprocess_dir": run_dir_path / "preprocess",
        "generate_dir": run_dir_path / "generate",
        "outputs_dir": run_dir_path / "outputs",
        "debug_dir": run_dir_path / "debug",
        "metrics_dir": run_dir_path / "metrics",
        "manifest_path": run_dir_path / "manifest.json",
    }
    for key, path in layout.items():
        if key.endswith("_dir"):
            Path(path).mkdir(parents=True, exist_ok=True)
    return layout


def _new_manifest(layout: dict, repo_root: Path) -> dict:
    now = utc_now_iso()
    return {
        "manifest_version": MANIFEST_VERSION,
        "run_name": layout["run_name"],
        "run_dir": str(layout["run_dir"]),
        "created_at_utc": now,
        "updated_at_utc": now,
        "repo": get_repo_info(repo_root),
        "hardware": get_hardware_info(),
        "stages": {},
    }


def load_manifest(layout: dict, repo_root: Path | None = None) -> dict:
    repo_root = repo_root or find_repo_root(layout["run_dir"])
    manifest_path = Path(layout["manifest_path"])
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    manifest = _new_manifest(layout, repo_root)
    write_manifest(layout, manifest)
    return manifest


def write_manifest(layout: dict, manifest: dict) -> None:
    manifest["updated_at_utc"] = utc_now_iso()
    manifest_path = Path(layout["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def start_stage_manifest(
    layout: dict,
    stage: str,
    args,
    inputs: dict | None = None,
    extra: dict | None = None,
) -> dict:
    manifest = load_manifest(layout)
    stage_id = str(uuid.uuid4())
    entry = {
        "id": stage_id,
        "status": "started",
        "started_at_utc": utc_now_iso(),
        "completed_at_utc": None,
        "command": sys.argv,
        "cwd": os.getcwd(),
        "args": _serialize(args),
        "inputs": _serialize(inputs or {}),
        "outputs": {},
        "metrics": {},
        "extra": _serialize(extra or {}),
    }
    manifest.setdefault("stages", {}).setdefault(stage, []).append(entry)
    write_manifest(layout, manifest)
    return {"stage": stage, "id": stage_id}


def finalize_stage_manifest(
    layout: dict,
    token: dict | None,
    status: str,
    outputs: dict | None = None,
    metrics: dict | None = None,
    extra: dict | None = None,
    error: str | None = None,
) -> None:
    if layout is None or token is None:
        return
    manifest = load_manifest(layout)
    stage_entries = manifest.setdefault("stages", {}).get(token["stage"], [])
    for entry in reversed(stage_entries):
        if entry.get("id") != token["id"]:
            continue
        entry["status"] = status
        entry["completed_at_utc"] = utc_now_iso()
        if outputs is not None:
            entry["outputs"] = _serialize(outputs)
        if metrics is not None:
            entry["metrics"] = _serialize(metrics)
        if extra is not None:
            merged = dict(entry.get("extra", {}))
            merged.update(_serialize(extra))
            entry["extra"] = merged
        if error is not None:
            entry["error"] = error
        write_manifest(layout, manifest)
        return
    raise RuntimeError(f"Failed to finalize manifest entry for stage={token['stage']}")


def should_write_manifest(args) -> bool:
    return bool(
        getattr(args, "save_manifest", False)
        or getattr(args, "run_name", None)
        or getattr(args, "run_dir", None)
    )

