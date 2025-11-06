#!/usr/bin/env python3
"""Run lightweight inference on a subset of samples for sanity checking."""

from __future__ import annotations

import json
import os
import sys

from utils.brain_tumor_training import (
    TEST_RESULTS_FILE,
    load_metadata,
    resolve_dataset_dir,
    resolve_output_dir,
    run_inference_samples,
)


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else int(value)


def main() -> None:
    dataset_dir = resolve_dataset_dir()
    output_dir = resolve_output_dir()

    try:
        load_metadata(output_dir)
    except FileNotFoundError as exc:
        print(f"[test_model] Metadata missing at {output_dir}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    sample_count = _get_int("TEST_SAMPLE_COUNT", 5)
    print(f"[test_model] Running inference on {sample_count} samples from {dataset_dir}")
    results = run_inference_samples(dataset_dir, output_dir, sample_count=sample_count)
    print(f"[test_model] Results saved to {output_dir / TEST_RESULTS_FILE}")
    preview = json.dumps(results, indent=2)
    if len(preview) > 500:
        preview = preview[:500] + "..."
    print(f"[test_model] Sample predictions: {preview}")
    print("[test_model] Completed")


if __name__ == "__main__":
    main()
