#!/usr/bin/env python
import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.generate_candidate_selection import score_generate_candidates


def _make_row(name: str, *, default: bool, halo=0.0, gradient=0.0, contrast=0.0, seam=0.0, bg=0.0, runtime=0.0):
    return {
        "candidate_name": name,
        "is_default": default,
        "generate_returncode": 0,
        "output_video": f"/tmp/{name}.mkv",
        "replacement_metrics": {},
        "runtime_stats": {"total_generate_sec": 10.0 * (1.0 + runtime / 100.0)},
        "boundary_metrics": {
            "halo_ratio_before": 1.0,
            "halo_ratio_after": 1.0 - halo / 100.0,
            "band_gradient_before_mean": 1.0,
            "band_gradient_after_mean": 1.0 + gradient / 100.0,
            "band_edge_contrast_before_mean": 1.0,
            "band_edge_contrast_after_mean": 1.0 + contrast / 100.0,
        },
        "seam_score_mean": 1.0 + seam / 100.0,
        "background_fluctuation_mean": 1.0 + bg / 100.0,
        "total_generate_sec": 10.0 * (1.0 + runtime / 100.0),
    }


def main():
    summary = {
        "default_candidate": "default_legacy",
        "cases": [
            {
                "case_name": "case_a",
                "rows": [
                    _make_row("default_legacy", default=True),
                    _make_row("balanced", default=False, halo=5.0, gradient=4.0, contrast=3.0, seam=1.0, bg=1.0, runtime=10.0),
                    _make_row("aggressive", default=False, halo=8.0, gradient=7.0, contrast=6.0, seam=8.0, bg=6.0, runtime=40.0),
                ],
            },
            {
                "case_name": "case_b",
                "rows": [
                    _make_row("default_legacy", default=True),
                    _make_row("balanced", default=False, halo=4.0, gradient=4.5, contrast=2.0, seam=1.0, bg=0.5, runtime=9.0),
                    _make_row("aggressive", default=False, halo=7.0, gradient=6.0, contrast=5.5, seam=7.0, bg=5.0, runtime=38.0),
                ],
            },
        ],
    }
    result = score_generate_candidates(summary)
    assert result["selected_candidate"] == "balanced", result
    assert result["selection_stable_ratio"] >= 1.0, result
    assert result["selected_better_than_default_ratio"] >= 1.0, result
    assert result["positive_edge_triplet_ratio"] >= 1.0, result

    with tempfile.TemporaryDirectory() as tmpdir:
        payload = Path(tmpdir) / "selection.json"
        payload.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "PASS", "selection_json": str(payload)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
