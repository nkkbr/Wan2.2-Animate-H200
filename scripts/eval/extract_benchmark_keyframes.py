#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import cv2


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_frame(video_path: Path, frame_index: int, output_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write extracted frame to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract benchmark keyframes from a benchmark manifest.")
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()

    manifest = _read_json(Path(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for case in manifest.get("labeled_cases", []):
        case_id = case["case_id"]
        video_path = Path(case["video_path"])
        for frame_index in case.get("keyframes", []):
            out_path = output_dir / case_id / f"frame_{frame_index:06d}.png"
            _extract_frame(video_path, int(frame_index), out_path)
            records.append(
                {
                    "case_id": case_id,
                    "video_path": str(video_path),
                    "frame_index": int(frame_index),
                    "output_path": str(out_path.resolve()),
                }
            )

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "output_dir": str(output_dir.resolve()),
        "extracted_frame_count": len(records),
        "records": records,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "extracted_frame_count": len(records)}))


if __name__ == "__main__":
    main()

