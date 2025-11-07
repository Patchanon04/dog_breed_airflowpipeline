#!/usr/bin/env python3
"""Evaluate the TensorFlow retrained model and enforce accuracy thresholds."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from utils.brain_tumor_training import resolve_output_dir
from utils.brain_tumor_training_tf import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMG_SIZE,
    TF_EVALUATION_FILE,
    evaluate_tf_best_model,
    load_tf_metadata,
)


def _get_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def main() -> None:
    output_dir = resolve_output_dir()

    try:
        metadata = load_tf_metadata(output_dir)
    except FileNotFoundError as exc:
        print(f"[evaluate_model_tf] Metadata missing: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[evaluate_model_tf] Evaluating TF model stored at {metadata['model_paths']['saved_model_dir']}")
    metrics = evaluate_tf_best_model(output_dir)

    metrics_path = Path(output_dir) / TF_EVALUATION_FILE
    print(f"[evaluate_model_tf] Metrics saved to {metrics_path}")
    print(f"[evaluate_model_tf] Metrics: {json.dumps(metrics, indent=2)}")

    acc_threshold = _get_float("TF_ACCURACY_THRESHOLD", 0.85)
    if metrics.get("accuracy", 0.0) < acc_threshold:
        raise SystemExit(
            f"Accuracy {metrics.get('accuracy'):.4f} below threshold {acc_threshold:.4f}"
        )

    print("[evaluate_model_tf] Accuracy threshold satisfied")


if __name__ == "__main__":
    main()
