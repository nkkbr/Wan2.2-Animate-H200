from pathlib import Path
import sys
import tempfile

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.temporal_handoff import (
    compose_temporal_handoff_latents,
    overlap_frames_to_latent_slots,
    pack_overlap_tensor_to_latent_slots,
    write_temporal_handoff_debug,
)


def main():
    assert overlap_frames_to_latent_slots(1) == 1
    assert overlap_frames_to_latent_slots(5) == 2
    assert overlap_frames_to_latent_slots(9) == 3
    assert overlap_frames_to_latent_slots(13) == 4

    frames = torch.arange(5 * 2 * 2, dtype=torch.float32).view(5, 2, 2)
    packed = pack_overlap_tensor_to_latent_slots(frames, reduction="mean")
    assert packed.shape == (2, 2, 2)
    assert torch.allclose(packed[0], frames[0])
    assert torch.allclose(packed[1], frames[1:5].mean(dim=0))

    base = torch.zeros((16, 20, 4, 4), dtype=torch.float32)
    memory = torch.ones((16, 20, 4, 4), dtype=torch.float32)

    composed_latent, latent_stats = compose_temporal_handoff_latents(
        base_latents=base,
        previous_output_latents=memory,
        overlap_frames=5,
        mode="latent",
        strength=1.0,
    )
    assert latent_stats["applied"] is True
    assert latent_stats["latent_slots"] == 2
    assert torch.allclose(composed_latent[:, :2], torch.ones_like(composed_latent[:, :2]))
    assert torch.allclose(composed_latent[:, 2:], base[:, 2:])

    replacement_strength = torch.zeros((2, 4, 4), dtype=torch.float32)
    replacement_strength[:, :2, :2] = 1.0
    composed_hybrid, hybrid_stats = compose_temporal_handoff_latents(
        base_latents=base,
        previous_output_latents=memory,
        overlap_frames=5,
        mode="hybrid",
        strength=0.5,
        replacement_strength_slots=replacement_strength,
    )
    assert hybrid_stats["applied"] is True
    assert hybrid_stats["latent_slots"] == 2
    assert torch.allclose(composed_hybrid[:, :2, :2, :2], torch.full_like(composed_hybrid[:, :2, :2, :2], 0.5))
    assert torch.allclose(composed_hybrid[:, :2, 2:, 2:], torch.zeros_like(composed_hybrid[:, :2, 2:, 2:]))

    with tempfile.TemporaryDirectory() as tmpdir:
        debug_paths = write_temporal_handoff_debug(
            save_debug_dir=tmpdir,
            handoff_index=1,
            stats=hybrid_stats,
            base_latents=base[:, :2],
            memory_latents=memory[:, -2:],
            composed_latents=composed_hybrid[:, :2],
            blend_mask=replacement_strength,
        )
        assert Path(debug_paths["stats"]).exists()
        assert Path(debug_paths["latents"]).exists()

    print("Synthetic temporal handoff packing: PASS")
    print("Synthetic temporal handoff composition: PASS")
    print("Synthetic temporal handoff debug export: PASS")


if __name__ == "__main__":
    main()
