#!/usr/bin/env python3
"""Train the brain tumor classification model using shared utilities."""

from __future__ import annotations

import json
import os
from typing import Any

from utils.brain_tumor_training import (
    METADATA_FILE,
    resolve_dataset_dir,
    resolve_output_dir,
    train_model_pipeline,
)


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else int(value)


def _get_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else float(value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    dataset_dir = resolve_dataset_dir()
    output_dir = resolve_output_dir()

    params: dict[str, Any] = {
        "batch_size": _get_int("TRAIN_BATCH_SIZE", 32),
        "epochs": _get_int("TRAIN_EPOCHS", 50),
        "lr": _get_float("TRAIN_LR", 1e-4),
        "patience": _get_int("TRAIN_PATIENCE", 5),
        "num_workers": _get_int("TRAIN_NUM_WORKERS", 4),
        "img_size": _get_int("TRAIN_IMG_SIZE", 224),
        "seed": _get_int("TRAIN_SEED", 42),
        "use_tensorboard": _get_bool("TRAIN_USE_TENSORBOARD", True),
    }

    display_params = {k: v for k, v in params.items() if k != "use_tensorboard"}
    print("[train_model] Starting training with parameters:")
    print(json.dumps(display_params, indent=2))
    print(f"[train_model] TensorBoard enabled: {params['use_tensorboard']}")

    metadata = train_model_pipeline(
        data_dir=dataset_dir,
        output_dir=output_dir,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        lr=params["lr"],
        patience=params["patience"],
        num_workers=params["num_workers"],
        img_size=params["img_size"],
        seed=params["seed"],
        use_tensorboard=params["use_tensorboard"],
    )

    metadata_path = output_dir / METADATA_FILE
    print(f"[train_model] Training metadata persisted to {metadata_path}")
    print(f"[train_model] Metadata summary: {json.dumps(metadata)}")
    print("[train_model] Completed")


if __name__ == "__main__":
    main()
