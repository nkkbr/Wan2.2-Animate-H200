#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.media_io import load_mask_artifact


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Compute background precision proxy metrics for optimization3.")
    parser.add_argument("--src_root_path", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    src_root = Path(args.src_root_path)
    runtime_stats = _read_json(src_root / "preprocess_runtime_stats.json")
    metadata = _read_json(src_root / "metadata.json")
    artifacts = metadata.get("src_files", {})

    def _load_mask(name: str):
        artifact = artifacts.get(name)
        if not artifact:
            return None
        return load_mask_artifact(src_root / artifact["path"], artifact.get("format"))

    def _corr(a: np.ndarray | None, b: np.ndarray | None, mask: np.ndarray | None = None):
        if a is None or b is None:
            return None
        aa = np.asarray(a, dtype=np.float32).reshape(-1)
        bb = np.asarray(b, dtype=np.float32).reshape(-1)
        if mask is not None:
            mm = np.asarray(mask, dtype=np.float32).reshape(-1) > 0.5
            aa = aa[mm]
            bb = bb[mm]
        if aa.size < 2 or bb.size < 2:
            return None
        if np.allclose(aa.std(), 0.0) or np.allclose(bb.std(), 0.0):
            return None
        return float(np.corrcoef(aa, bb)[0, 1])

    background_stats = runtime_stats.get("background", {})
    multistage = runtime_stats.get("multistage", {})
    person_mask = _load_mask("person_mask")
    visible_support = _load_mask("visible_support")
    unresolved_region = _load_mask("unresolved_region")
    background_confidence = _load_mask("background_confidence")
    background_source_provenance = _load_mask("background_source_provenance")
    result = {
        "mode": "proxy",
        "background_mode": background_stats.get("mode"),
        "background_stats": background_stats.get("stats", {}),
        "metadata_background_mode": metadata.get("src_files", {}).get("background", {}).get("background_mode"),
        "background_artifact": metadata.get("src_files", {}).get("background", {}).get("path"),
        "visible_support_artifact": metadata.get("src_files", {}).get("visible_support", {}).get("path"),
        "unresolved_region_artifact": metadata.get("src_files", {}).get("unresolved_region", {}).get("path"),
        "background_confidence_artifact": metadata.get("src_files", {}).get("background_confidence", {}).get("path"),
        "background_source_provenance_artifact": metadata.get("src_files", {}).get("background_source_provenance", {}).get("path"),
        "visible_support_mean": None if visible_support is None else float(visible_support.mean()),
        "unresolved_region_mean": None if unresolved_region is None else float(unresolved_region.mean()),
        "background_confidence_mean": None if background_confidence is None else float(background_confidence.mean()),
        "support_confidence_corr": _corr(visible_support, background_confidence, person_mask),
        "support_unresolved_corr": _corr(visible_support, unresolved_region, person_mask),
        "confidence_unresolved_corr": _corr(background_confidence, unresolved_region, person_mask),
        "background_provenance_mean": None if background_source_provenance is None else float(background_source_provenance.mean()),
        "person_roi_coverage_ratio": ((multistage.get("person_roi_analysis") or {}).get("proposal_stats") or {}).get("coverage_ratio"),
        "face_roi_coverage_ratio": ((multistage.get("face_roi_analysis") or {}).get("proposal_stats") or {}).get("coverage_ratio"),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
