#!/usr/bin/env python3
"""Download the training dataset from S3 and generate summary statistics."""

from __future__ import annotations

import json
import os
import sys

from utils import s3_helper
from utils.brain_tumor_training import (
    download_s3_prefix,
    resolve_dataset_dir,
    resolve_output_dir,
    save_dataset_summary,
    summarize_dataset,
)


def main() -> None:
    bucket = os.environ.get("S3_BUCKET")
    prefix = os.environ.get("S3_PREFIX")

    if not bucket or not prefix:
        print("[load_data] S3_BUCKET and S3_PREFIX are required", file=sys.stderr)
        sys.exit(1)

    dataset_dir = resolve_dataset_dir()
    output_dir = resolve_output_dir()

    print(f"[load_data] Downloading from s3://{bucket}/{prefix} -> {dataset_dir}")
    client = s3_helper.get_s3_client()
    result = download_s3_prefix(bucket, prefix, dataset_dir, client)
    print(f"[load_data] Download summary: {json.dumps(result)}")

    print("[load_data] Generating dataset summary...")
    summary = summarize_dataset(dataset_dir)
    summary_path = save_dataset_summary(output_dir, summary)
    print(f"[load_data] Dataset summary saved to {summary_path}")
    print(f"[load_data] Summary: {json.dumps(summary)}")
    print("[load_data] Completed")


if __name__ == "__main__":
    main()
