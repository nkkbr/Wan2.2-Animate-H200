from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "docs" / "optimization6" / "benchmark" / "external_model_registry.step02.json"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "wan2_2_external_models"


def read_external_model_registry(path: str | Path = DEFAULT_REGISTRY) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_external_model_entry(model_id: str, path: str | Path = DEFAULT_REGISTRY) -> dict:
    registry = read_external_model_registry(path)
    for entry in registry.get("models", []):
        if entry.get("model_id") == model_id:
            return entry
    raise KeyError(f"Unknown external alpha model id: {model_id}")


def compute_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_external_model_weight(model_id: str, cache_root: str | Path | None = None, registry_path: str | Path = DEFAULT_REGISTRY) -> Path:
    entry = get_external_model_entry(model_id, registry_path)
    cache_root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    model_dir = cache_root / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    url = entry["weight_url"]
    filename = Path(url).name
    if filename in {"view", ""}:
        filename = f"{model_id}.weights"
    weight_path = model_dir / filename
    if not weight_path.exists():
        urllib.request.urlretrieve(url, weight_path)
    expected = entry.get("sha256")
    actual = compute_sha256(weight_path)
    if expected and actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {model_id}: expected {expected}, got {actual} ({weight_path})"
        )
    return weight_path
