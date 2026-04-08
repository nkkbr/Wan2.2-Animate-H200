#!/usr/bin/env python
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREPROCESS_DIR = REPO_ROOT / "wan" / "modules" / "animate" / "preprocess"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from face_analysis import run_face_analysis, write_face_analysis_artifacts


def _make_case(frame_count=8, height=128, width=128):
    frames = np.full((frame_count, height, width, 3), np.array([42, 56, 68], dtype=np.uint8), dtype=np.uint8)
    source = frames.copy()
    metas = []
    bboxes = []
    hard = np.zeros((frame_count, height, width), dtype=np.float32)
    soft = np.zeros_like(hard)
    occ = np.zeros_like(hard)
    unc = np.zeros_like(hard)
    rng = np.random.default_rng(0)
    angles = np.linspace(0.0, 2.0 * np.pi, 69, endpoint=False)
    for index in range(frame_count):
        cx = 62 + index * 1.5 + float(rng.normal(0.0, 0.8))
        cy = 46 + float(rng.normal(0.0, 0.8))
        rx = 11 + (index % 2)
        ry = 15
        pts = []
        for angle in angles:
            x = (cx + rx * np.cos(angle)) / width
            y = (cy + ry * np.sin(angle)) / height
            conf = 0.92
            pts.append([x, y, conf])
        pts = np.asarray(pts, dtype=np.float32)
        if index == 4:
            pts[:12, 2] = 0.20
        metas.append({"keypoints_face": pts})
        bbox = [cx - 12 + float(rng.normal(0.0, 2.0)), cx + 12 + float(rng.normal(0.0, 2.0)), cy - 16 + float(rng.normal(0.0, 2.0)), cy + 10 + float(rng.normal(0.0, 2.0))]
        bboxes.append(bbox)
        cv2.ellipse(frames[index], (int(round(cx)), int(round(cy))), (14, 18), 0, 0, 360, (210, 184, 170), -1)
        hard[index, int(max(cy - 20, 0)):int(min(cy + 22, height)), int(max(cx - 18, 0)):int(min(cx + 18, width))] = 1.0
        soft[index] = cv2.GaussianBlur(hard[index], (9, 9), 0)
        if index == 4:
            occ[index, int(max(cy - 4, 0)):int(min(cy + 8, height)), int(max(cx - 2, 0)):int(min(cx + 14, width))] = 1.0
            cv2.rectangle(frames[index], (int(cx - 2), int(cy - 4)), (int(cx + 14), int(cy + 8)), (120, 110, 100), -1)
            unc[index] = cv2.GaussianBlur(occ[index], (9, 9), 0)
    return frames, source, metas, bboxes, hard, soft, occ, unc


def main():
    export_frames, source_frames, metas, face_bboxes, hard, soft, occ, unc = _make_case()
    raw_centers = np.array([[(bbox[0] + bbox[1]) * 0.5, (bbox[2] + bbox[3]) * 0.5] for bbox in face_bboxes], dtype=np.float32)
    raw_jitter = float(np.linalg.norm(np.diff(raw_centers, axis=0), axis=1).mean())
    analysis = run_face_analysis(
        export_frames=export_frames,
        face_source_frames=source_frames,
        pose_metas=metas,
        face_bboxes=face_bboxes,
        export_shape=export_frames.shape[1:3],
        analysis_shape=export_frames.shape[1:3],
        face_source_shape=source_frames.shape[1:3],
        hard_foreground=hard,
        soft_alpha=soft,
        occlusion_band=occ,
        uncertainty_map=unc,
    )
    tracked_frames = analysis["bbox_curve"]["frames"]
    tracked_centers = np.array([[frame["center_x"], frame["center_y"]] for frame in tracked_frames], dtype=np.float32)
    tracked_jitter = float(np.linalg.norm(np.diff(tracked_centers, axis=0), axis=1).mean())
    if tracked_jitter >= raw_jitter:
        raise AssertionError(f"tracking-aware face jitter did not improve: raw={raw_jitter:.4f} tracked={tracked_jitter:.4f}")
    assert analysis["face_images"].shape == (8, 512, 512, 3)
    assert analysis["face_alpha"].shape == hard.shape
    assert analysis["face_parsing"].shape == hard.shape
    assert analysis["face_uncertainty"].shape == hard.shape
    assert analysis["landmarks_json"]["frame_count"] == 8
    assert analysis["head_pose_json"]["frame_count"] == 8
    assert analysis["expression_json"]["frame_count"] == 8
    occ_frame_unc = float(analysis["face_uncertainty"][4].mean())
    clean_frame_unc = float(analysis["face_uncertainty"][0].mean())
    if occ_frame_unc <= clean_frame_unc:
        raise AssertionError("face uncertainty did not increase on the synthetic occluded frame.")

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = write_face_analysis_artifacts(tmpdir, analysis, fps=5.0)
        for artifact in artifacts.values():
            path = Path(tmpdir) / artifact["path"]
            if not path.exists():
                raise AssertionError(f"Expected face analysis artifact missing: {path}")

    print("Synthetic face analysis stack: PASS")


if __name__ == "__main__":
    main()
