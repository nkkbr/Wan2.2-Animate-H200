#!/usr/bin/env python
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.experiment import sanitize_run_name


def parse_scalar(raw: str):
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw or "e" in lowered:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_assignment(raw: str) -> tuple[str, object]:
    if "=" not in raw:
        raise ValueError(f"Expected KEY=VALUE assignment. Got: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip().replace("-", "_")
    if not key:
        raise ValueError(f"Empty assignment key in: {raw}")
    return key, parse_scalar(value)


def parse_axis(raw: str) -> tuple[str, list[object]]:
    if "=" not in raw:
        raise ValueError(f"Expected KEY=v1,v2 axis syntax. Got: {raw}")
    key, values = raw.split("=", 1)
    key = key.strip().replace("-", "_")
    parsed_values = [parse_scalar(item) for item in values.split(",") if item.strip()]
    if not parsed_values:
        raise ValueError(f"Axis {key} has no values: {raw}")
    return key, parsed_values


def build_run_name(run_prefix: str, case_id: str | None, overrides: dict[str, object]) -> str:
    segments = [run_prefix]
    if case_id:
        segments.append(case_id)
    for key in sorted(overrides):
        value = overrides[key]
        segments.append(f"{key}-{value}")
    return sanitize_run_name("__".join(str(segment) for segment in segments))


def iter_search_combinations(axes: dict[str, list[object]]):
    if not axes:
        yield {}
        return
    keys = list(axes.keys())
    for combo in itertools.product(*(axes[key] for key in keys)):
        yield dict(zip(keys, combo))


def append_cli_arg(command: list[str], key: str, value) -> None:
    flag = f"--{key.replace('_', '-')}"
    if isinstance(value, bool):
        command.extend([flag, "true" if value else "false"])
    else:
        command.extend([flag, str(value)])


def build_generate_command(
    *,
    python_bin: str,
    ckpt_dir: str,
    src_root_path: str,
    run_dir: Path,
    task: str,
    replace_flag: bool,
    base_args: dict[str, object],
    overrides: dict[str, object],
) -> list[str]:
    command = [
        python_bin,
        "generate.py",
        "--task",
        task,
        "--ckpt_dir",
        ckpt_dir,
        "--src_root_path",
        src_root_path,
        "--run_dir",
        str(run_dir),
        "--save_manifest",
        "--save_file",
        str((run_dir / "outputs" / "result.mp4").resolve()),
    ]
    if replace_flag:
        command.append("--replace_flag")
    merged_args = dict(base_args)
    merged_args.update(overrides)
    for key in sorted(merged_args):
        append_cli_arg(command, key, merged_args[key])
    return command


def parse_args():
    parser = argparse.ArgumentParser(description="Build and optionally execute Wan-Animate parameter search runs.")
    parser.add_argument("--python_bin", type=str, default=sys.executable, help="Python interpreter used to launch generate.py.")
    parser.add_argument("--task", type=str, default="animate-14B", help="Generate task. Defaults to animate-14B.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory passed to generate.py.")
    parser.add_argument("--src_root_path", type=str, required=True, help="Preprocess output directory passed to generate.py.")
    parser.add_argument("--save_root", type=str, default="runs/parameter_search", help="Directory that will contain per-run folders.")
    parser.add_argument("--run_prefix", type=str, default="animate_search", help="Prefix used when constructing run names.")
    parser.add_argument("--case_id", type=str, default=None, help="Optional benchmark case identifier embedded into run names.")
    parser.add_argument("--base_arg", action="append", default=[], help="Static generate arg assignment, e.g. sample_solver=dpm++.")
    parser.add_argument("--axis", action="append", default=[], help="Search axis assignment, e.g. sample_steps=40,60.")
    parser.add_argument("--execute", action="store_true", default=False, help="Execute generated commands. Without this flag the script only writes the plan.")
    parser.add_argument("--compute_metrics", action="store_true", default=False, help="After a successful run, invoke compute_replacement_metrics.py on the run directory.")
    parser.add_argument("--max_runs", type=int, default=None, help="Optional cap on generated combinations.")
    parser.add_argument("--output_plan_json", type=str, default=None, help="Optional JSON file path for the generated search plan.")
    return parser.parse_args()


def main():
    args = parse_args()
    save_root = Path(args.save_root).resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    base_args = dict(parse_assignment(item) for item in args.base_arg)
    axes = dict(parse_axis(item) for item in args.axis)

    plan = {
        "task": args.task,
        "ckpt_dir": str(Path(args.ckpt_dir).resolve()),
        "src_root_path": str(Path(args.src_root_path).resolve()),
        "save_root": str(save_root),
        "case_id": args.case_id,
        "base_args": base_args,
        "axes": axes,
        "runs": [],
    }

    run_count = 0
    for overrides in iter_search_combinations(axes):
        if args.max_runs is not None and run_count >= args.max_runs:
            break
        run_name = build_run_name(args.run_prefix, args.case_id, overrides)
        run_dir = save_root / run_name
        command = build_generate_command(
            python_bin=args.python_bin,
            ckpt_dir=args.ckpt_dir,
            src_root_path=args.src_root_path,
            run_dir=run_dir,
            task=args.task,
            replace_flag=True,
            base_args=base_args,
            overrides=overrides,
        )
        entry = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "overrides": overrides,
            "command": command,
            "status": "planned",
        }
        if args.execute:
            result = subprocess.run(command, cwd=REPO_ROOT)
            entry["returncode"] = result.returncode
            entry["status"] = "completed" if result.returncode == 0 else "failed"
            if result.returncode == 0 and args.compute_metrics:
                metrics_command = [
                    args.python_bin,
                    "scripts/eval/compute_replacement_metrics.py",
                    "--run_dir",
                    str(run_dir),
                ]
                metrics_result = subprocess.run(metrics_command, cwd=REPO_ROOT)
                entry["metrics_returncode"] = metrics_result.returncode
                if metrics_result.returncode != 0:
                    entry["status"] = "metrics_failed"
        plan["runs"].append(entry)
        run_count += 1

    output_plan_path = Path(args.output_plan_json).resolve() if args.output_plan_json else save_root / "parameter_search_plan.json"
    with output_plan_path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    print(f"Wrote parameter search plan to {output_plan_path}")
    print(f"Planned runs: {len(plan['runs'])}")
    if args.execute:
        completed = sum(1 for item in plan["runs"] if item["status"] == "completed")
        print(f"Completed runs: {completed}")


if __name__ == "__main__":
    main()
