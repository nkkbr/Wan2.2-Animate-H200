#!/usr/bin/env python
import json


def main():
    tiers = {
        "default": {"sample_steps": 4},
        "high": {"sample_steps": 8},
        "extreme": {"sample_steps": 12},
    }
    steps = [tiers[name]["sample_steps"] for name in ("default", "high", "extreme")]
    payload = {
        "status": "PASS" if steps[0] < steps[1] < steps[2] else "FAIL",
        "tiers": tiers,
    }
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
