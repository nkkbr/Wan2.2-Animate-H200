import cv2
import numpy as np

from alpha_refinement import (
    make_binary_mask_preview,
    make_trimap_preview,
    run_alpha_refinement_v2,
    run_alpha_refinement_v3,
)


def _lazy_import_backgroundmattingv2():
    from external_alpha_backgroundmattingv2 import BackgroundMattingV2Adapter, BackgroundMattingV2Config

    return BackgroundMattingV2Adapter, BackgroundMattingV2Config


def _mean_color(frame: np.ndarray, mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    pixels = frame[mask > 0]
    if pixels.size == 0:
        return fallback.astype(np.float32)
    return pixels.reshape(-1, 3).mean(axis=0).astype(np.float32)


def _distance_alpha(frame: np.ndarray, fg_color: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
    frame_f = frame.astype(np.float32)
    d_fg = np.linalg.norm(frame_f - fg_color[None, None, :], axis=-1)
    d_bg = np.linalg.norm(frame_f - bg_color[None, None, :], axis=-1)
    return np.clip(d_bg / (d_fg + d_bg + 1e-6), 0.0, 1.0).astype(np.float32)


def _run_heuristic_matting(
    *,
    frames: np.ndarray,
    hard_mask: np.ndarray,
    soft_band: np.ndarray,
    parsing_boundary_prior: np.ndarray,
    trimap_inner_erode: int,
    trimap_outer_dilate: int,
    blur_kernel: int,
    spatial_weight: float,
    color_weight: float,
) -> dict:
    frame_count, height, width = hard_mask.shape
    erode_kernel = max(1, int(trimap_inner_erode))
    dilate_kernel = max(1, int(trimap_outer_dilate))
    blur_kernel = max(1, int(blur_kernel))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    erode_element = np.ones((erode_kernel * 2 + 1, erode_kernel * 2 + 1), dtype=np.uint8)
    dilate_element = np.ones((dilate_kernel * 2 + 1, dilate_kernel * 2 + 1), dtype=np.uint8)

    soft_alpha_frames = []
    unknown_region_frames = []
    uncertainty_prior_frames = []
    support_region_frames = []
    trimap_frames = []
    spatial_weight = float(np.clip(spatial_weight, 0.0, 1.0))
    color_weight = float(np.clip(color_weight, 0.0, 1.0))
    total_weight = max(spatial_weight + color_weight, 1e-6)
    spatial_weight /= total_weight
    color_weight /= total_weight

    for frame, mask, band, parsing_prior in zip(frames, hard_mask, soft_band, parsing_boundary_prior):
        hard = (mask > 0.5).astype(np.uint8)
        sure_fg = cv2.erode(hard, erode_element, iterations=1)
        support = cv2.dilate(hard, dilate_element, iterations=1).astype(np.float32)
        support = np.maximum(support, np.clip(band + parsing_prior, 0.0, 1.0))
        sure_bg = (support < 0.5).astype(np.uint8)
        unknown = np.clip(1 - sure_fg - sure_bg, 0, 1).astype(np.uint8)

        inside_dist = cv2.distanceTransform(hard, cv2.DIST_L2, 5)
        outside_dist = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 5)
        spatial_alpha = inside_dist / (inside_dist + outside_dist + 1e-6)

        fg_color = _mean_color(frame, sure_fg, fallback=np.array([192.0, 160.0, 144.0], dtype=np.float32))
        bg_color = _mean_color(frame, sure_bg, fallback=frame.reshape(-1, 3).mean(axis=0))
        color_alpha = _distance_alpha(frame, fg_color, bg_color)
        alpha_disagreement = np.abs(spatial_alpha - color_alpha).astype(np.float32)

        alpha = np.where(
            sure_fg > 0,
            1.0,
            np.where(
                sure_bg > 0,
                0.0,
                spatial_weight * spatial_alpha + color_weight * color_alpha,
            ),
        ).astype(np.float32)
        alpha = np.maximum(alpha, mask * (1.0 - np.clip(band, 0.0, 1.0) * 0.15))
        alpha = np.minimum(alpha, np.clip(support, 0.0, 1.0))
        if blur_kernel > 1:
            alpha = cv2.GaussianBlur(alpha, (blur_kernel, blur_kernel), sigmaX=0)
        alpha = np.where(sure_fg > 0, 1.0, alpha)
        alpha = np.where(sure_bg > 0, 0.0, alpha)
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

        soft_alpha_frames.append(alpha)
        unknown_region_frames.append(unknown.astype(np.float32))
        trimap = np.full((height, width), 0.5, dtype=np.float32)
        trimap[sure_bg > 0] = 0.0
        trimap[sure_fg > 0] = 1.0
        trimap_frames.append(trimap)
        uncertainty_prior = np.clip(
            0.55 * unknown.astype(np.float32)
            + 0.30 * alpha_disagreement * np.clip(support, 0.0, 1.0)
            + 0.25 * np.clip(band + parsing_prior, 0.0, 1.0),
            0.0,
            1.0,
        ).astype(np.float32)
        uncertainty_prior_frames.append(uncertainty_prior)
        support_region_frames.append(np.clip(support, 0.0, 1.0).astype(np.float32))

    soft_alpha_frames = np.stack(soft_alpha_frames).astype(np.float32)
    unknown_region_frames = np.stack(unknown_region_frames).astype(np.float32)
    uncertainty_prior_frames = np.stack(uncertainty_prior_frames).astype(np.float32)
    support_region_frames = np.stack(support_region_frames).astype(np.float32)
    trimap_frames = np.stack(trimap_frames).astype(np.float32)
    return {
        "mode": "heuristic",
        "soft_alpha": soft_alpha_frames,
        "unknown_region": unknown_region_frames,
        "uncertainty_prior": uncertainty_prior_frames,
        "support_region": support_region_frames,
        "trimap": trimap_frames,
        "stats": {
            "soft_alpha_mean": float(soft_alpha_frames.mean()),
            "unknown_region_mean": float(unknown_region_frames.mean()),
            "uncertainty_prior_mean": float(uncertainty_prior_frames.mean()),
            "support_region_mean": float(support_region_frames.mean()),
            "trimap_unknown_mean": float((trimap_frames == 0.5).astype(np.float32).mean()),
        },
    }


def run_matting_adapter(
    *,
    frames: np.ndarray,
    hard_mask: np.ndarray,
    background_frames: np.ndarray | None = None,
    soft_band: np.ndarray | None = None,
    parsing_boundary_prior: np.ndarray | None = None,
    parsing_head_prior: np.ndarray | None = None,
    parsing_hand_prior: np.ndarray | None = None,
    parsing_occlusion_prior: np.ndarray | None = None,
    parsing_part_foreground_prior: np.ndarray | None = None,
    mode: str = "heuristic",
    external_alpha_model_id: str | None = None,
    external_alpha_blend: float = 0.35,
    external_alpha_delta_clip: float = 0.08,
    external_alpha_hair_boost: float = 0.12,
    trimap_inner_erode: int = 3,
    trimap_outer_dilate: int = 12,
    blur_kernel: int = 5,
    spatial_weight: float = 0.7,
    color_weight: float = 0.3,
    alpha_v2_detail_boost: float = 0.28,
    alpha_v2_shrink_strength: float = 0.34,
    alpha_v2_hair_boost: float = 0.42,
    alpha_v2_hard_threshold: float = 0.68,
    alpha_v2_bilateral_sigma_color: float = 0.12,
    alpha_v2_bilateral_sigma_space: float = 5.0,
    alpha_v3_detail_boost: float = 0.24,
    alpha_v3_color_mix: float = 0.42,
    alpha_v3_active_blend: float = 0.55,
    alpha_v3_delta_clip: float = 0.10,
) -> dict:
    frames = np.asarray(frames, dtype=np.uint8)
    hard_mask = np.asarray(hard_mask, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must have shape [T, H, W, 3]. Got {frames.shape}.")
    if hard_mask.shape != frames.shape[:3]:
        raise ValueError(f"hard_mask must match frames. Got {hard_mask.shape} vs {frames.shape[:3]}.")

    frame_count, height, width = hard_mask.shape
    zeros = np.zeros((frame_count, height, width), dtype=np.float32)
    if mode == "none":
        return {
            "mode": "none",
            "soft_alpha": hard_mask.astype(np.float32),
            "unknown_region": zeros,
            "uncertainty_prior": zeros,
            "support_region": (hard_mask > 0.0).astype(np.float32),
            "trimap": np.where(hard_mask > 0.5, 1.0, 0.0).astype(np.float32),
            "stats": {
                "soft_alpha_mean": float(hard_mask.mean()),
                "unknown_region_mean": 0.0,
                "uncertainty_prior_mean": 0.0,
            },
        }
    if mode not in {"heuristic", "high_precision_v2", "production_v1", "external_bmv2"}:
        raise ValueError(f"Unsupported matting adapter mode: {mode}")

    if soft_band is None:
        soft_band = zeros
    else:
        soft_band = np.asarray(soft_band, dtype=np.float32)
    if parsing_boundary_prior is None:
        parsing_boundary_prior = zeros
    else:
        parsing_boundary_prior = np.asarray(parsing_boundary_prior, dtype=np.float32)
    parsing_head_prior = zeros if parsing_head_prior is None else np.asarray(parsing_head_prior, dtype=np.float32)
    parsing_hand_prior = zeros if parsing_hand_prior is None else np.asarray(parsing_hand_prior, dtype=np.float32)
    parsing_occlusion_prior = zeros if parsing_occlusion_prior is None else np.asarray(parsing_occlusion_prior, dtype=np.float32)
    parsing_part_foreground_prior = zeros if parsing_part_foreground_prior is None else np.asarray(parsing_part_foreground_prior, dtype=np.float32)

    heuristic = _run_heuristic_matting(
        frames=frames,
        hard_mask=hard_mask,
        soft_band=soft_band,
        parsing_boundary_prior=parsing_boundary_prior,
        trimap_inner_erode=trimap_inner_erode,
        trimap_outer_dilate=trimap_outer_dilate,
        blur_kernel=blur_kernel,
        spatial_weight=spatial_weight,
        color_weight=color_weight,
    )
    if mode == "heuristic":
        return heuristic

    if mode == "external_bmv2":
        if not external_alpha_model_id:
            raise ValueError("external_alpha_model_id is required when mode=external_bmv2")
        if background_frames is None:
            raise ValueError("background_frames is required when mode=external_bmv2")
        background_frames = np.asarray(background_frames, dtype=np.uint8)
        if background_frames.shape != frames.shape:
            raise ValueError(f"background_frames must match frames shape. Got {background_frames.shape} vs {frames.shape}.")

        BackgroundMattingV2Adapter, BackgroundMattingV2Config = _lazy_import_backgroundmattingv2()
        adapter = BackgroundMattingV2Adapter(BackgroundMattingV2Config(model_id=external_alpha_model_id))

        external_alpha_frames = []
        external_trimap_unknown_frames = []
        external_hair_alpha_frames = []
        external_confidence_frames = []
        external_provenance_frames = []
        runtime_values = []
        source_repo = None
        release_or_commit = None
        weight_url = None
        sha256 = None
        license_name = None

        for frame, background in zip(frames, background_frames):
            out = adapter.infer(frame, background)
            external_alpha_frames.append(out["alpha"].astype(np.float32))
            external_trimap_unknown_frames.append(out["trimap_unknown"].astype(np.float32))
            hair_alpha = np.clip(
                out["hair_alpha"].astype(np.float32) * (1.0 + float(external_alpha_hair_boost)),
                0.0,
                1.0,
            )
            external_hair_alpha_frames.append(hair_alpha)
            external_confidence_frames.append(out["alpha_confidence"].astype(np.float32))
            external_provenance_frames.append(out["alpha_source_provenance"].astype(np.float32))
            runtime_values.append(float(out["runtime_sec"]))
            source_repo = out.get("source_repo")
            release_or_commit = out.get("release_or_commit")
            weight_url = out.get("weight_url")
            sha256 = out.get("sha256")
            license_name = out.get("license")

        external_alpha = np.stack(external_alpha_frames).astype(np.float32)
        external_trimap_unknown = np.stack(external_trimap_unknown_frames).astype(np.float32)
        external_hair_alpha = np.stack(external_hair_alpha_frames).astype(np.float32)
        external_confidence = np.stack(external_confidence_frames).astype(np.float32)
        external_provenance = np.stack(external_provenance_frames).astype(np.float32)

        compatibility_support = np.clip(
            0.45 * heuristic["unknown_region"].astype(np.float32)
            + 0.30 * np.clip(soft_band, 0.0, 1.0).astype(np.float32)
            + 0.25 * np.clip(parsing_boundary_prior, 0.0, 1.0).astype(np.float32),
            0.0,
            1.0,
        )
        compatibility_support = np.maximum(compatibility_support, external_trimap_unknown * 0.5).astype(np.float32)
        compatibility_delta = np.clip(
            external_alpha - heuristic["soft_alpha"].astype(np.float32),
            -float(external_alpha_delta_clip),
            float(external_alpha_delta_clip),
        ).astype(np.float32)
        active_soft_alpha = np.clip(
            heuristic["soft_alpha"].astype(np.float32) + compatibility_delta * compatibility_support * float(external_alpha_blend),
            0.0,
            1.0,
        ).astype(np.float32)

        trimap = heuristic["trimap"].astype(np.float32).copy()
        trimap = np.where(external_trimap_unknown > 0.5, 0.5, trimap)

        return {
            "mode": mode,
            "soft_alpha": active_soft_alpha,
            "unknown_region": heuristic["unknown_region"].astype(np.float32),
            "uncertainty_prior": np.maximum(
                heuristic["uncertainty_prior"].astype(np.float32),
                external_trimap_unknown.astype(np.float32) * 0.7,
            ),
            "support_region": heuristic["support_region"].astype(np.float32),
            "trimap": trimap,
            "trimap_unknown": external_trimap_unknown.astype(np.float32),
            "hair_alpha": external_hair_alpha.astype(np.float32),
            "alpha_confidence": external_confidence.astype(np.float32),
            "alpha_source_provenance": external_provenance.astype(np.float32),
            "alpha_v3": external_alpha.astype(np.float32),
            "stats": {
                **heuristic["stats"],
                "mode": mode,
                "external_alpha_model_id": external_alpha_model_id,
                "external_alpha_source_repo": source_repo,
                "external_alpha_release": release_or_commit,
                "external_alpha_weight_url": weight_url,
                "external_alpha_sha256": sha256,
                "external_alpha_license": license_name,
                "external_alpha_runtime_sec_mean": float(np.mean(runtime_values)) if runtime_values else 0.0,
                "external_alpha_mean": float(external_alpha.mean()),
                "external_trimap_unknown_mean": float(external_trimap_unknown.mean()),
                "external_hair_alpha_mean": float(external_hair_alpha.mean()),
                "active_soft_alpha_mean": float(active_soft_alpha.mean()),
                "active_soft_alpha_delta_mean": float(np.abs(active_soft_alpha - heuristic["soft_alpha"].astype(np.float32)).mean()),
            },
        }

    alpha_v2 = run_alpha_refinement_v2(
        frames=frames,
        hard_mask=hard_mask,
        base_soft_alpha=heuristic["soft_alpha"],
        unknown_region=heuristic["unknown_region"],
        uncertainty_prior=heuristic["uncertainty_prior"],
        support_region=heuristic["support_region"],
        soft_band=soft_band,
        parsing_boundary_prior=parsing_boundary_prior,
        head_prior=parsing_head_prior,
        hand_prior=parsing_hand_prior,
        occlusion_prior=parsing_occlusion_prior,
        part_foreground_prior=parsing_part_foreground_prior,
        trimap_inner_erode=trimap_inner_erode,
        trimap_outer_dilate=trimap_outer_dilate,
        blur_kernel=blur_kernel,
        detail_boost=alpha_v2_detail_boost,
        shrink_strength=alpha_v2_shrink_strength,
        hair_boost=alpha_v2_hair_boost,
        hard_threshold=alpha_v2_hard_threshold,
        bilateral_sigma_color=alpha_v2_bilateral_sigma_color,
        bilateral_sigma_space=alpha_v2_bilateral_sigma_space,
    )
    legacy_soft_alpha = heuristic["soft_alpha"].astype(np.float32)
    alpha_v2_soft = alpha_v2["alpha_v2"].astype(np.float32)
    compatibility_support = np.clip(
        0.45 * alpha_v2["fine_boundary_mask"].astype(np.float32)
        + 0.35 * alpha_v2["hair_edge_mask"].astype(np.float32)
        + 0.20 * heuristic["unknown_region"].astype(np.float32),
        0.0,
        1.0,
    ).astype(np.float32)
    compatibility_delta = np.clip(alpha_v2_soft - legacy_soft_alpha, -0.02, 0.02).astype(np.float32)
    active_soft_alpha = np.clip(
        legacy_soft_alpha + compatibility_delta * compatibility_support,
        0.0,
        1.0,
    ).astype(np.float32)
    output = {
        "mode": mode,
        "soft_alpha": active_soft_alpha,
        "unknown_region": heuristic["unknown_region"].astype(np.float32),
        "uncertainty_prior": np.maximum(
            heuristic["uncertainty_prior"].astype(np.float32),
            alpha_v2["alpha_uncertainty_v2"].astype(np.float32),
        ),
        "support_region": heuristic["support_region"].astype(np.float32),
        "trimap": heuristic["trimap"].astype(np.float32),
        "trimap_v2": alpha_v2["trimap_v2"].astype(np.float32),
        "alpha_v2": alpha_v2["alpha_v2"].astype(np.float32),
        "alpha_uncertainty_v2": alpha_v2["alpha_uncertainty_v2"].astype(np.float32),
        "alpha_confidence": alpha_v2["alpha_confidence"].astype(np.float32),
        "alpha_source_provenance": alpha_v2["alpha_source_provenance"].astype(np.float32),
        "fine_boundary_mask": alpha_v2["fine_boundary_mask"].astype(np.float32),
        "hair_edge_mask": alpha_v2["hair_edge_mask"].astype(np.float32),
        "refined_hard_foreground": alpha_v2["refined_hard_foreground"].astype(np.float32),
        "legacy_soft_alpha": legacy_soft_alpha,
        "stats": {
            **heuristic["stats"],
            **alpha_v2["stats"],
            "mode": mode,
            "active_soft_alpha_mean": float(active_soft_alpha.mean()),
            "active_soft_alpha_delta_mean": float(np.abs(active_soft_alpha - legacy_soft_alpha).mean()),
        },
    }
    if mode != "production_v1":
        return output

    alpha_v3 = run_alpha_refinement_v3(
        frames=frames,
        legacy_soft_alpha=legacy_soft_alpha,
        alpha_v2=alpha_v2["alpha_v2"],
        trimap_v2=alpha_v2["trimap_v2"],
        alpha_uncertainty_v2=alpha_v2["alpha_uncertainty_v2"],
        fine_boundary_mask=alpha_v2["fine_boundary_mask"],
        hair_edge_mask=alpha_v2["hair_edge_mask"],
        refined_hard_foreground=alpha_v2["refined_hard_foreground"],
        support_region=heuristic["support_region"],
        unknown_region=heuristic["unknown_region"],
        head_prior=parsing_head_prior,
        hand_prior=parsing_hand_prior,
        occlusion_prior=parsing_occlusion_prior,
        detail_boost=alpha_v3_detail_boost,
        color_mix=alpha_v3_color_mix,
        active_blend=alpha_v3_active_blend,
        delta_clip=alpha_v3_delta_clip,
    )
    output.update({
        "soft_alpha": alpha_v3["soft_alpha"].astype(np.float32),
        "trimap_unknown": alpha_v3["trimap_unknown"].astype(np.float32),
        "hair_alpha": alpha_v3["hair_alpha"].astype(np.float32),
        "alpha_confidence": alpha_v3["alpha_confidence"].astype(np.float32),
        "alpha_source_provenance": alpha_v3["alpha_source_provenance"].astype(np.float32),
        "stats": {
            **output["stats"],
            **alpha_v3["stats"],
            "mode": mode,
            "active_soft_alpha_mean": float(alpha_v3["soft_alpha"].mean()),
            "active_soft_alpha_delta_mean": float(np.abs(alpha_v3["soft_alpha"] - legacy_soft_alpha).mean()),
        },
    })
    return output


def make_matting_alpha_preview(soft_alpha: np.ndarray) -> np.ndarray:
    soft_alpha = np.asarray(soft_alpha, dtype=np.float32)
    if soft_alpha.ndim != 3:
        raise ValueError(f"soft_alpha must have shape [T, H, W]. Got {soft_alpha.shape}.")
    preview = np.clip(np.rint(soft_alpha[..., None] * 255.0), 0, 255).astype(np.uint8)
    return np.repeat(preview, repeats=3, axis=3)


def make_trimap_preview_rgb(trimap: np.ndarray) -> np.ndarray:
    return make_trimap_preview(trimap)


def make_alpha_mask_preview(mask: np.ndarray) -> np.ndarray:
    return make_binary_mask_preview(mask)
