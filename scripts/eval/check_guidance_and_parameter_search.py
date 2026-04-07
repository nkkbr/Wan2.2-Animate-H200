from pathlib import Path
import json
import sys
import tempfile

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.run_animate_parameter_search import (
    build_generate_command,
    build_run_name,
    iter_search_combinations,
)
from scripts.eval.summarize_animate_parameter_search import extract_summary_row
from wan.utils.guidance import combine_animate_guidance_predictions


def main():
    cond = torch.tensor([10.0])
    legacy_null = torch.tensor([6.0])
    face_null = torch.tensor([7.0])
    text_null = torch.tensor([8.0])

    legacy = combine_animate_guidance_predictions(
        cond_pred=cond,
        guidance_mode="legacy_both",
        legacy_scale=1.5,
        legacy_null_pred=legacy_null,
    )
    decoupled = combine_animate_guidance_predictions(
        cond_pred=cond,
        guidance_mode="decoupled",
        face_scale=1.5,
        text_scale=1.2,
        face_null_pred=face_null,
        text_null_pred=text_null,
    )
    face_only = combine_animate_guidance_predictions(
        cond_pred=cond,
        guidance_mode="face_only",
        face_scale=1.5,
        face_null_pred=face_null,
    )
    text_only = combine_animate_guidance_predictions(
        cond_pred=cond,
        guidance_mode="text_only",
        text_scale=1.3,
        text_null_pred=text_null,
    )

    assert torch.allclose(legacy, torch.tensor([12.0]))
    assert torch.allclose(decoupled, torch.tensor([11.9]))
    assert torch.allclose(face_only, torch.tensor([11.5]))
    assert torch.allclose(text_only, torch.tensor([10.6]))

    axes = {"sample_steps": [40, 60], "face_guide_scale": [1.0, 1.2]}
    combos = list(iter_search_combinations(axes))
    assert len(combos) == 4
    run_name = build_run_name("search", "caseA", combos[0])
    assert "sample_steps-40" in run_name and "face_guide_scale-1.0" in run_name

    command = build_generate_command(
        python_bin=sys.executable,
        ckpt_dir="/tmp/ckpt",
        src_root_path="/tmp/preprocess",
        run_dir=Path("/tmp/run"),
        task="animate-14B",
        replace_flag=True,
        base_args={"sample_solver": "dpm++", "guidance_uncond_mode": "decoupled"},
        overrides={"sample_steps": 60, "face_guide_scale": 1.2},
    )
    command_str = " ".join(command)
    assert "--guidance-uncond-mode decoupled" in command_str
    assert "--face-guide-scale 1.2" in command_str
    assert "--sample-steps 60" in command_str

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_a"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_name": "run_a",
            "stages": {
                "generate": [
                    {
                        "status": "completed",
                        "args": {
                            "sample_solver": "dpm++",
                            "sample_steps": 60,
                            "sample_shift": 5.0,
                            "guidance_uncond_mode": "decoupled",
                            "sample_guide_scale": 1.0,
                            "face_guide_scale": 1.2,
                            "text_guide_scale": 1.0,
                            "refert_num": 5,
                            "quality_preset": "hq_h200",
                        },
                        "metrics": {
                            "total_generate_sec": 12.3,
                            "peak_memory_gb": 45.6,
                            "guidance_forward_passes_per_step": 2,
                            "seam_boundary_before_mean": 0.2,
                            "seam_boundary_after_mean": 0.1,
                            "overlap_mad_before_mean": 0.15,
                            "overlap_mad_after_prev_mean": 0.08,
                        },
                    }
                ]
            },
        }
        with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        with (run_dir / "metrics" / "basic_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "seam_score": {"mean": 0.11},
                    "background_fluctuation": {"mean": 0.07},
                },
                handle,
            )
        row = extract_summary_row(run_dir, {})
        assert row["guidance_uncond_mode"] == "decoupled"
        assert row["face_guide_scale"] == 1.2
        assert row["seam_boundary_after_mean"] == 0.1
        assert row["background_fluctuation_mean"] == 0.07

    print("Synthetic guidance decoupling math: PASS")
    print("Synthetic parameter search planning: PASS")
    print("Synthetic parameter search summary: PASS")


if __name__ == "__main__":
    main()
