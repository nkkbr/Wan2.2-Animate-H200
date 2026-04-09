#!/usr/bin/env python
import json
from pathlib import Path

manifest = json.loads(Path('docs/optimization6/benchmark/tier_manifest.step06.json').read_text(encoding='utf-8'))
required = ['default', 'high_quality', 'extreme']
for name in required:
    assert name in manifest, name
    tier = manifest[name]
    assert int(tier['sample_steps']) > 0
    assert int(tier['frame_num']) > int(tier['refert_num'])
    assert tier['replacement_conditioning_mode'] == 'legacy'
    assert tier['boundary_refine_mode'] == 'none'
print(json.dumps({'status': 'PASS', 'tiers': required}, ensure_ascii=False))
