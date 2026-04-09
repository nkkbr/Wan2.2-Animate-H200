from __future__ import annotations

from typing import Iterable


ROI_TASK_TYPES = (
    "alpha_refinement",
    "matte_completion",
    "boundary_uncertainty_refinement",
    "semantic_boundary_expert",
    "compositing_aware_edge_correction",
)

ROI_SEMANTIC_TAGS = (
    "mixed_boundary",
    "face",
    "hair",
    "hand",
    "cloth",
    "occluded",
    "semi_transparent",
    "hard_negative",
)

ROI_DATASET_SPLITS = ("train", "val", "test")
ROI_SOURCE_SPLITS = ("seed_eval", "expansion_eval", "holdout_eval")
ROI_SPLIT_POLICIES = ("reviewed_split_v1", "temporal_quarantine_v1")

REQUIRED_RECORD_KEYS = (
    "sample_id",
    "dataset_split",
    "source_review_split",
    "case_id",
    "preprocess_frame_index",
    "source_frame_index",
    "task_type",
    "semantic_boundary_tag",
    "difficulty_score",
    "is_hard_negative",
    "roi_box_xyxy",
    "label_json_path",
)


def validate_record(record: dict) -> None:
    missing = [key for key in REQUIRED_RECORD_KEYS if key not in record]
    if missing:
        raise ValueError(f"ROI dataset record is missing keys: {missing}")
    if record["dataset_split"] not in ROI_DATASET_SPLITS:
        raise ValueError(f"Invalid dataset_split: {record['dataset_split']}")
    if record["source_review_split"] not in ROI_SOURCE_SPLITS:
        raise ValueError(f"Invalid source_review_split: {record['source_review_split']}")
    if record["task_type"] not in ROI_TASK_TYPES:
        raise ValueError(f"Invalid task_type: {record['task_type']}")
    if record["semantic_boundary_tag"] not in ROI_SEMANTIC_TAGS:
        raise ValueError(f"Invalid semantic_boundary_tag: {record['semantic_boundary_tag']}")
    if not isinstance(record["is_hard_negative"], bool):
        raise ValueError("is_hard_negative must be bool")
    box = record["roi_box_xyxy"]
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("roi_box_xyxy must be a list of four integers")


def validate_records(records: Iterable[dict]) -> None:
    for record in records:
        validate_record(record)
