#!/usr/bin/env python3
"""ทำความสะอาดและเตรียม train/val/test split สำหรับ TensorFlow pipeline."""

from __future__ import annotations

import json
import os
import sys

from utils.brain_tumor_training import (
    clean_corrupted_images,
    resolve_dataset_dir,
    resolve_output_dir,
)
from utils.brain_tumor_training_tf import (
    DEFAULT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    TF_SPLIT_MANIFEST,
    prepare_tf_split,
)


def _get_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _as_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    dataset_dir = resolve_dataset_dir()
    output_dir = resolve_output_dir()

    write_report = _as_bool(os.environ.get("TF_CLEAN_WRITE_REPORT"), True)

    print(f"[clean_data_tf] Cleaning dataset at {dataset_dir}")
    clean_stats = clean_corrupted_images(dataset_dir, write_report=write_report)
    print(f"[clean_data_tf] Cleaning stats: {json.dumps(clean_stats)}")

    train_ratio = _get_float("TF_TRAIN_RATIO", DEFAULT_TRAIN_RATIO)
    val_ratio = _get_float("TF_VAL_RATIO", DEFAULT_VAL_RATIO)
    test_ratio = _get_float("TF_TEST_RATIO", DEFAULT_TEST_RATIO)
    seed = _get_int("TF_SEED", DEFAULT_SEED)

    try:
        manifest = prepare_tf_split(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    except ValueError as exc:
        print(f"[clean_data_tf] Failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("[clean_data_tf] Completed dataset split")
    print(json.dumps(manifest, indent=2))
    manifest_path = output_dir / TF_SPLIT_MANIFEST
    print(f"[clean_data_tf] Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
