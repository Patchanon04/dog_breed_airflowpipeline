#!/usr/bin/env python3
"""Remove corrupted images from the local dataset copy."""

from __future__ import annotations

import json
import os

from utils.brain_tumor_training import clean_corrupted_images, resolve_dataset_dir


def _as_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    dataset_dir = resolve_dataset_dir()
    write_report = _as_bool(os.environ.get("CLEAN_WRITE_REPORT"), True)

    print(f"[clean_data] Scanning dataset at {dataset_dir}")
    stats = clean_corrupted_images(dataset_dir, write_report=write_report)
    print(f"[clean_data] Cleaning stats: {json.dumps(stats)}")
    print("[clean_data] Completed")


if __name__ == "__main__":
    main()
