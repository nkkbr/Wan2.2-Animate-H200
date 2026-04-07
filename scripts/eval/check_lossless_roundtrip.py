import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.media_io import (
    load_person_mask_artifact,
    load_rgb_artifact,
    write_output_frames,
    write_person_mask_artifact,
    write_rgb_artifact,
)
from wan.utils.utils import tensor_to_video_frames


def assert_equal(name: str, lhs: np.ndarray, rhs: np.ndarray) -> None:
    if lhs.dtype != rhs.dtype:
        raise AssertionError(f"{name}: dtype mismatch {lhs.dtype} != {rhs.dtype}")
    if lhs.shape != rhs.shape:
        raise AssertionError(f"{name}: shape mismatch {lhs.shape} != {rhs.shape}")
    if not np.array_equal(lhs, rhs):
        diff = np.abs(lhs.astype(np.int64) - rhs.astype(np.int64)).max()
        raise AssertionError(f"{name}: arrays differ (max abs diff={diff})")


def assert_allclose(name: str, lhs: np.ndarray, rhs: np.ndarray, atol: float = 1e-7) -> None:
    if lhs.shape != rhs.shape:
        raise AssertionError(f"{name}: shape mismatch {lhs.shape} != {rhs.shape}")
    if not np.allclose(lhs, rhs, atol=atol, rtol=0.0):
        diff = np.abs(lhs - rhs).max()
        raise AssertionError(f"{name}: arrays differ (max abs diff={diff})")


def main():
    with tempfile.TemporaryDirectory(prefix="wan_lossless_roundtrip_") as tmp_dir:
        root = Path(tmp_dir)

        rgb_frames = np.zeros((3, 32, 48, 3), dtype=np.uint8)
        rgb_frames[0, :, :16] = [255, 0, 0]
        rgb_frames[1, :, 16:32] = [0, 255, 0]
        rgb_frames[2, :, 32:] = [0, 0, 255]

        person_mask = np.zeros((3, 32, 48), dtype=np.float32)
        person_mask[:, 8:24, 12:36] = np.array([0.25, 0.5, 1.0], dtype=np.float32)[:, None, None]

        pose_info = write_rgb_artifact(
            frames=rgb_frames,
            output_root=root,
            stem="src_pose",
            artifact_format="png_seq",
            fps=30.0,
        )
        bg_info = write_rgb_artifact(
            frames=rgb_frames,
            output_root=root,
            stem="src_bg",
            artifact_format="npz",
            fps=30.0,
        )
        mask_info = write_person_mask_artifact(
            mask_frames=person_mask,
            output_root=root,
            stem="src_mask",
            artifact_format="npz",
            fps=30.0,
        )

        pose_roundtrip = load_rgb_artifact(root / pose_info["path"], pose_info["format"])
        bg_roundtrip = load_rgb_artifact(root / bg_info["path"], bg_info["format"])
        mask_roundtrip = load_person_mask_artifact(root / mask_info["path"], mask_info["format"])

        assert_equal("rgb png_seq roundtrip", rgb_frames, pose_roundtrip)
        assert_equal("rgb npz roundtrip", rgb_frames, bg_roundtrip)
        assert_allclose("person mask npz roundtrip", person_mask, mask_roundtrip)

        video_tensor = torch.from_numpy(rgb_frames).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1.0
        writer_frames = tensor_to_video_frames(video_tensor, nrow=1, normalize=True, value_range=(-1, 1))
        output_dir = write_output_frames(writer_frames, root / "debug_output", fps=12.0, output_format="png_seq")
        output_roundtrip = load_rgb_artifact(output_dir, "png_seq")
        assert_equal("png_seq output writer roundtrip", writer_frames, output_roundtrip)

        print("Lossless RGB png_seq roundtrip: PASS")
        print("Lossless RGB npz roundtrip: PASS")
        print("Lossless person_mask npz roundtrip: PASS")
        print("High-fidelity png_seq output writer roundtrip: PASS")


if __name__ == "__main__":
    main()
