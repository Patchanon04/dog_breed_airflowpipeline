#!/usr/bin/env python3
"""Train the TensorFlow CNN pipeline derived from the `brain_tumor_clf` notebooks."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from utils.brain_tumor_training import resolve_dataset_dir, resolve_output_dir
from utils.brain_tumor_training_tf import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    TF_METADATA_FILE,
    train_tf_model_pipeline,
)


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def main() -> None:
    dataset_dir = resolve_dataset_dir()
    output_dir = resolve_output_dir()

    params: dict[str, Any] = {
        "batch_size": _get_int("TF_BATCH_SIZE", DEFAULT_BATCH_SIZE),
        "epochs": _get_int("TF_EPOCHS", DEFAULT_EPOCHS),
        "img_size": _get_int("TF_IMG_SIZE", DEFAULT_IMG_SIZE),
        "patience": _get_int("TF_PATIENCE", DEFAULT_PATIENCE),
        "seed": _get_int("TF_SEED", DEFAULT_SEED),
    }

    print("[train_model_tf] Starting TensorFlow training with parameters:")
    print(json.dumps(params, indent=2))

    metadata = train_tf_model_pipeline(
        data_dir=dataset_dir,
        output_dir=output_dir,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        img_size=params["img_size"],
        patience=params["patience"],
        seed=params["seed"],
    )

    metadata_path = output_dir / TF_METADATA_FILE
    print(f"[train_model_tf] Training metadata saved to {metadata_path}")
    print(f"[train_model_tf] Metrics summary: {json.dumps(metadata['metrics'], indent=2)}")
    print("[train_model_tf] Completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - surface errors to Airflow logs
        print(f"[train_model_tf] Failed: {exc}", file=sys.stderr)
        raise
