#!/usr/bin/env python3
"""ทำ inference ตัวอย่างสำหรับโมเดล TensorFlow retrain line 2."""

from __future__ import annotations

import json
import os
import sys

from utils.brain_tumor_training import resolve_output_dir
from utils.brain_tumor_training_tf import (
    TF_TEST_RESULTS_FILE,
    load_tf_metadata,
    run_tf_inference_samples,
)


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def main() -> None:
    output_dir = resolve_output_dir()

    try:
        load_tf_metadata(output_dir)
    except FileNotFoundError as exc:
        print(f"[test_model_tf] Metadata missing at {output_dir}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    sample_count = _get_int("TF_TEST_SAMPLE_COUNT", 5)
    print(f"[test_model_tf] Running TF inference on {sample_count} samples")
    results = run_tf_inference_samples(output_dir, sample_count=sample_count)

    results_path = output_dir / TF_TEST_RESULTS_FILE
    print(f"[test_model_tf] Predictions saved to {results_path}")
    preview = json.dumps(results, indent=2)
    if len(preview) > 500:
        preview = preview[:500] + "..."
    print(f"[test_model_tf] Sample predictions: {preview}")
    print("[test_model_tf] Completed")


if __name__ == "__main__":
    main()
