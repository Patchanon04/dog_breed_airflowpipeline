#!/usr/bin/env python3
"""Evaluate the best trained model against the test split and enforce thresholds."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from utils.brain_tumor_training import (
    EVALUATION_FILE,
    METADATA_FILE,
    evaluate_best_model,
    load_metadata,
    resolve_dataset_dir,
    resolve_output_dir,
)


def _get_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def main() -> None:
    dataset_dir = resolve_dataset_dir()
    output_dir = resolve_output_dir()

    try:
        metadata = load_metadata(output_dir)
    except FileNotFoundError as exc:
        print(f"[evaluate_model] Metadata missing at {output_dir / METADATA_FILE}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[evaluate_model] Evaluating best checkpoint from {metadata['best_model_path']}")
    metrics = evaluate_best_model(dataset_dir, output_dir)
    metrics_path = Path(output_dir) / EVALUATION_FILE
    print(f"[evaluate_model] Metrics saved to {metrics_path}")
    print(f"[evaluate_model] Metrics: {json.dumps(metrics)}")

    acc_threshold = _get_float("ACCURACY_THRESHOLD", 0.85)
    if metrics.get("accuracy", 0.0) < acc_threshold:
        raise SystemExit(
            f"Accuracy {metrics.get('accuracy'):.4f} below threshold {acc_threshold:.4f}"
        )

    print("[evaluate_model] Passed accuracy threshold")


if __name__ == "__main__":
    main()
