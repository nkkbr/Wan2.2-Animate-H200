#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path


def latest_stage(manifest: dict, stage: str) -> dict | None:
    stage_entries = manifest.get("stages", {}).get(stage, [])
    return stage_entries[-1] if stage_entries else None


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize animate parameter-search runs from manifests and metrics.")
    parser.add_argument("--runs_root", type=str, required=True, help="Directory containing per-run folders with manifest.json.")
    parser.add_argument("--manual_scores_csv", type=str, default=None, help="Optional CSV produced from docs/benchmark/manual_score_template.csv.")
    parser.add_argument("--output_csv", type=str, default=None, help="Summary CSV path. Defaults to <runs_root>/parameter_search_summary.csv.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional summary JSON path.")
    return parser.parse_args()


def load_manual_scores(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["run_name"]: row for row in reader if row.get("run_name")}


def extract_summary_row(run_dir: Path, manual_scores: dict[str, dict]) -> dict | None:
    manifest = load_json(run_dir / "manifest.json")
    if manifest is None:
        return None
    generate_entry = latest_stage(manifest, "generate")
    if generate_entry is None:
        return None

    args = generate_entry.get("args", {})
    metrics = generate_entry.get("metrics", {})
    basic_metrics = load_json(run_dir / "metrics" / "basic_metrics.json") or {}
    seam_score = basic_metrics.get("seam_score") or {}
    background_fluctuation = basic_metrics.get("background_fluctuation") or {}
    manual = manual_scores.get(manifest.get("run_name"), {})

    return {
        "run_name": manifest.get("run_name"),
        "status": generate_entry.get("status"),
        "sample_solver": args.get("sample_solver"),
        "sample_steps": args.get("sample_steps"),
        "sample_shift": args.get("sample_shift"),
        "guidance_uncond_mode": args.get("guidance_uncond_mode"),
        "sample_guide_scale": args.get("sample_guide_scale"),
        "face_guide_scale": args.get("face_guide_scale"),
        "text_guide_scale": args.get("text_guide_scale"),
        "refert_num": args.get("refert_num"),
        "quality_preset": args.get("quality_preset"),
        "total_generate_sec": metrics.get("total_generate_sec"),
        "peak_memory_gb": metrics.get("peak_memory_gb"),
        "guidance_forward_passes_per_step": metrics.get("guidance_forward_passes_per_step"),
        "seam_boundary_before_mean": metrics.get("seam_boundary_before_mean"),
        "seam_boundary_after_mean": metrics.get("seam_boundary_after_mean"),
        "overlap_mad_before_mean": metrics.get("overlap_mad_before_mean"),
        "overlap_mad_after_prev_mean": metrics.get("overlap_mad_after_prev_mean"),
        "seam_score_mean": seam_score.get("mean"),
        "background_fluctuation_mean": background_fluctuation.get("mean"),
        "identity_consistency": manual.get("identity_consistency"),
        "expression_following": manual.get("expression_following"),
        "motion_naturalness": manual.get("motion_naturalness"),
        "background_stability": manual.get("background_stability"),
        "edge_quality": manual.get("edge_quality"),
        "seam_visibility": manual.get("seam_visibility"),
        "notes": manual.get("notes"),
    }


def main():
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    manual_scores = load_manual_scores(Path(args.manual_scores_csv).resolve() if args.manual_scores_csv else None)
    rows = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        row = extract_summary_row(child, manual_scores)
        if row is not None:
            rows.append(row)

    output_csv = Path(args.output_csv).resolve() if args.output_csv else runs_root / "parameter_search_summary.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "run_name",
        "status",
        "sample_solver",
        "sample_steps",
        "sample_shift",
        "guidance_uncond_mode",
        "sample_guide_scale",
        "face_guide_scale",
        "text_guide_scale",
        "refert_num",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")

    print(f"Summarized {len(rows)} runs to {output_csv}")


if __name__ == "__main__":
    main()
