#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _pct(before: float, after: float) -> float:
    if before is None or after is None or abs(before) <= 1e-9:
        return 0.0
    return (after / before - 1.0) * 100.0


def _halo_reduction(roi_metrics: dict | None) -> float:
    if not roi_metrics:
        return 0.0
    before = float(roi_metrics.get('roi_halo_ratio_before', 0.0) or 0.0)
    after = float(roi_metrics.get('roi_halo_ratio_after', 0.0) or 0.0)
    if before <= 1e-9:
        return 0.0
    return (before - after) / before * 100.0


def main():
    parser = argparse.ArgumentParser(description='Evaluate optimization6 Step06 orchestration benchmark.')
    parser.add_argument('--summary_json', required=True, type=str)
    parser.add_argument('--output_json', default=None, type=str)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_json).read_text(encoding='utf-8'))
    rows = {row['tier']: row for row in summary['rows']}
    if 'high_quality' not in rows and 'high' in rows:
        rows['high_quality'] = rows['high']

    def compare(a: str, b: str):
        before = rows[a]
        after = rows[b]
        roi = after.get('roi_metrics') or {}
        return {
            'roi_gradient_gain_pct': _pct(float(roi.get('roi_band_gradient_before_mean', 0.0) or 0.0), float(roi.get('roi_band_gradient_after_mean', 0.0) or 0.0)),
            'roi_edge_contrast_gain_pct': _pct(float(roi.get('roi_band_edge_contrast_before_mean', 0.0) or 0.0), float(roi.get('roi_band_edge_contrast_after_mean', 0.0) or 0.0)),
            'roi_halo_reduction_pct': _halo_reduction(roi),
            'seam_degradation_pct': _pct(float(before.get('seam_score_mean') or 0.0), float(after.get('seam_score_mean') or 0.0)),
            'runtime_increase_pct': _pct(float(before.get('total_generate_sec') or 0.0), float(after.get('total_generate_sec') or 0.0)),
            'max_allocated_gb_delta': float(after.get('max_allocated_gb') or 0.0) - float(before.get('max_allocated_gb') or 0.0),
        }

    high_vs_default = compare('default', 'high_quality')
    extreme_vs_high = compare('high_quality', 'extreme')
    result = {
        'summary_json': str(Path(args.summary_json).resolve()),
        'metrics': {
            'high_vs_default': high_vs_default,
            'extreme_vs_high': extreme_vs_high,
        },
        'gates': {
            'high_quality_roi_gradient_ge_8pct': high_vs_default['roi_gradient_gain_pct'] >= 8.0,
            'high_quality_roi_edge_contrast_ge_6pct': high_vs_default['roi_edge_contrast_gain_pct'] >= 6.0,
            'high_quality_roi_halo_reduction_ge_6pct': high_vs_default['roi_halo_reduction_pct'] >= 6.0,
            'high_quality_seam_degradation_le_3pct': high_vs_default['seam_degradation_pct'] <= 3.0,
            'high_quality_runtime_le_2_5x_default': high_vs_default['runtime_increase_pct'] <= 150.0,
            'extreme_has_additional_key_gain': max(
                extreme_vs_high['roi_gradient_gain_pct'],
                extreme_vs_high['roi_edge_contrast_gain_pct'],
                extreme_vs_high['roi_halo_reduction_pct'],
            ) >= 6.0,
            'extreme_runtime_le_2x_high_quality': extreme_vs_high['runtime_increase_pct'] <= 100.0,
        },
    }
    result['passed'] = all(result['gates'].values())
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + '\n', encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
