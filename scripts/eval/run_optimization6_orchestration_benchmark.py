#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / 'runs'
GENERATE = REPO_ROOT / 'generate.py'
REPLACEMENT_METRICS = REPO_ROOT / 'scripts' / 'eval' / 'compute_replacement_metrics.py'
ROI_METRICS = REPO_ROOT / 'scripts' / 'eval' / 'compute_boundary_roi_metrics.py'


def _python_bin() -> str:
    override = os.environ.get('WAN_PYTHON')
    for candidate in [override, '/home/user1/miniconda3/envs/wan/bin/python', sys.executable]:
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())
    return sys.executable


PYTHON = _python_bin()


def _run(cmd: list[str], cwd: Path = REPO_ROOT):
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)


def _write_logs(logs_dir: Path, stem: str, result: subprocess.CompletedProcess):
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / f'{stem}.stdout.log').write_text(result.stdout, encoding='utf-8')
    (logs_dir / f'{stem}.stderr.log').write_text(result.stderr, encoding='utf-8')


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description='Run optimization6 Step06 H200 orchestration benchmark.')
    parser.add_argument('--suite_name', type=str, default=None)
    parser.add_argument('--manifest_json', type=str, default='docs/optimization6/benchmark/tier_manifest.step06.json')
    parser.add_argument('--src_root_path', type=str, default='runs/optimization3_step06_round5_ab/preprocess_video_v2/preprocess')
    parser.add_argument('--ckpt_dir', type=str, default='/home/user1/wan2.2/Wan2.2/Wan2.2-Animate-14B')
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest_json).read_text(encoding='utf-8'))
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    suite_name = args.suite_name or f'optimization6_step06_ab_{timestamp}'
    suite_dir = RUNS_ROOT / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = suite_dir / 'logs'
    src_root = Path(args.src_root_path).resolve()

    rows = []
    default_video = None
    for tier_name in ['default', 'high_quality', 'extreme']:
        tier = manifest[tier_name]
        out_dir = suite_dir / tier_name / 'outputs'
        out_dir.mkdir(parents=True, exist_ok=True)
        debug_dir = suite_dir / tier_name / 'debug' / 'generate'
        video_path = out_dir / f'{tier_name}.mkv'
        cmd = [
            PYTHON, str(GENERATE),
            '--task', 'animate-14B',
            '--ckpt_dir', str(Path(args.ckpt_dir).resolve()),
            '--src_root_path', str(src_root),
            '--save_file', str(video_path),
            '--output_format', 'ffv1',
            '--replace_flag',
            '--use_relighting_lora',
            '--offload_model', 'False',
            '--frame_num', str(tier['frame_num']),
            '--refert_num', str(tier['refert_num']),
            '--sample_solver', str(tier['sample_solver']),
            '--sample_steps', str(tier['sample_steps']),
            '--sample_shift', '5.0',
            '--sample_guide_scale', '1.0',
            '--replacement_mask_mode', 'soft_band',
            '--replacement_conditioning_mode', str(tier['replacement_conditioning_mode']),
            '--replacement_mask_downsample_mode', 'area',
            '--replacement_boundary_strength', '0.5',
            '--replacement_transition_low', '0.1',
            '--replacement_transition_high', '0.9',
            '--overlap_blend_mode', 'mask_aware',
            '--temporal_handoff_mode', 'pixel',
            '--boundary_refine_mode', str(tier['boundary_refine_mode']),
            '--log_runtime_stats',
            '--save_debug_dir', str(debug_dir),
        ]
        result = _run(cmd)
        _write_logs(logs_dir, f'{tier_name}_generate', result)
        runtime_stats = _read_json(debug_dir / 'wan_animate_runtime_stats.json')

        repl_json = suite_dir / f'{tier_name}_replacement_metrics.json'
        repl = _run([
            PYTHON, str(REPLACEMENT_METRICS),
            '--video_path', str(video_path),
            '--mask_path', str(src_root / 'src_mask.npz'),
            '--clip_len', str(tier['frame_num']),
            '--refert_num', str(tier['refert_num']),
            '--output_json', str(repl_json),
        ])
        _write_logs(logs_dir, f'{tier_name}_replacement_metrics', repl)
        replacement_metrics = _read_json(repl_json)

        row = {
            'tier': tier_name,
            'video_path': str(video_path.resolve()),
            'runtime_stats': runtime_stats,
            'replacement_metrics': replacement_metrics,
            'seam_score_mean': ((replacement_metrics or {}).get('seam_score') or {}).get('mean'),
            'background_fluctuation_mean': ((replacement_metrics or {}).get('background_fluctuation') or {}).get('mean'),
            'total_generate_sec': (runtime_stats or {}).get('total_generate_sec'),
            'max_allocated_gb': (runtime_stats or {}).get('peak_memory_gb'),
        }
        rows.append(row)
        if tier_name == 'default':
            default_video = video_path
        else:
            roi_json = suite_dir / f'{tier_name}_vs_default_roi_metrics.json'
            roi = _run([
                PYTHON, str(ROI_METRICS),
                '--before', str(default_video),
                '--after', str(video_path),
                '--src_root_path', str(src_root),
                '--output_json', str(roi_json),
            ])
            _write_logs(logs_dir, f'{tier_name}_vs_default_roi_metrics', roi)
            row['roi_metrics'] = _read_json(roi_json)

    summary = {
        'suite_dir': str(suite_dir.resolve()),
        'manifest_json': str(Path(args.manifest_json).resolve()),
        'src_root_path': str(src_root),
        'rows': rows,
    }
    summary_json = suite_dir / 'summary.json'
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'suite_dir': str(suite_dir.resolve()), 'summary_json': str(summary_json.resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
