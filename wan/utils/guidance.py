def combine_animate_guidance_predictions(
    *,
    cond_pred,
    guidance_mode: str,
    legacy_scale: float | None = None,
    face_scale: float = 1.0,
    text_scale: float = 1.0,
    legacy_null_pred=None,
    face_null_pred=None,
    text_null_pred=None,
):
    if guidance_mode == "legacy_both":
        if legacy_scale is None:
            raise ValueError("legacy_scale must be provided for legacy_both guidance.")
        if legacy_scale <= 1.0:
            return cond_pred
        if legacy_null_pred is None:
            raise ValueError("legacy_null_pred must be provided when legacy_scale > 1.")
        return legacy_null_pred + legacy_scale * (cond_pred - legacy_null_pred)

    noise_pred = cond_pred
    if guidance_mode not in {"face_only", "text_only", "decoupled"}:
        raise ValueError(f"Unsupported guidance_mode: {guidance_mode}")
    if face_scale > 1.0:
        if face_null_pred is None:
            raise ValueError("face_null_pred must be provided when face_scale > 1.")
        noise_pred = noise_pred + (face_scale - 1.0) * (cond_pred - face_null_pred)
    if text_scale > 1.0:
        if text_null_pred is None:
            raise ValueError("text_null_pred must be provided when text_scale > 1.")
        noise_pred = noise_pred + (text_scale - 1.0) * (cond_pred - text_null_pred)
    return noise_pred
