#!/usr/bin/env python3
"""Prepare TensorFlow-compatible dataset splits for retraining pipeline line 2."""

from __future__ import annotations

import json
import os
import sys

from utils.brain_tumor_training import resolve_dataset_dir, resolve_output_dir
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


def main() -> None:
    dataset_dir = resolve_dataset_dir()
    output_dir = resolve_output_dir()

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
        print(f"[prepare_tf_split] Failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("[prepare_tf_split] Completed dataset split")
    print(json.dumps(manifest, indent=2))
    manifest_path = output_dir / TF_SPLIT_MANIFEST
    print(f"[prepare_tf_split] Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
