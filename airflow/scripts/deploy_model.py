#!/usr/bin/env python3
"""Upload the best trained model and related artifacts to S3."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from utils import s3_helper
from utils.brain_tumor_training import (
    EVALUATION_FILE,
    METADATA_FILE,
    load_metadata,
    resolve_output_dir,
)

DEFAULT_ARTIFACT_PREFIX = "retrain-brain-tumor-model/models"


def _strip_join(*parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if part and part.strip("/")]
    return "/".join(cleaned)


def main() -> None:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        print("[deploy_model] S3_BUCKET is required", file=sys.stderr)
        raise SystemExit(1)

    artifact_prefix = os.environ.get("MODEL_ARTIFACT_PREFIX") or os.environ.get("S3_PREFIX") or DEFAULT_ARTIFACT_PREFIX
    model_version = os.environ.get("MODEL_VERSION", "latest")

    output_dir = resolve_output_dir()

    try:
        metadata = load_metadata(output_dir)
    except FileNotFoundError as exc:
        print(f"[deploy_model] Metadata missing at {output_dir / METADATA_FILE}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    best_model_path = Path(metadata["best_model_path"])
    if not best_model_path.exists():
        print(f"[deploy_model] Best model not found at {best_model_path}", file=sys.stderr)
        raise SystemExit(1)

    evaluation_path = Path(output_dir) / EVALUATION_FILE
    if not evaluation_path.exists():
        print(f"[deploy_model] Evaluation file missing at {evaluation_path}. Run evaluate_model first.", file=sys.stderr)
        raise SystemExit(1)

    key_prefix = _strip_join(artifact_prefix, model_version)
    model_key = _strip_join(key_prefix, best_model_path.name)
    metadata_key = _strip_join(key_prefix, METADATA_FILE)
    evaluation_key = _strip_join(key_prefix, EVALUATION_FILE)

    client = s3_helper.get_s3_client()

    print(f"[deploy_model] Uploading model to s3://{bucket}/{model_key}")
    client.upload_file(str(best_model_path), bucket, model_key)

    print(f"[deploy_model] Uploading metadata to s3://{bucket}/{metadata_key}")
    client.upload_file(str(output_dir / METADATA_FILE), bucket, metadata_key)

    print(f"[deploy_model] Uploading evaluation metrics to s3://{bucket}/{evaluation_key}")
    client.upload_file(str(evaluation_path), bucket, evaluation_key)

    print(
        "[deploy_model] Deployment artifacts uploaded:",
        json.dumps(
            {
                "bucket": bucket,
                "keys": {
                    "model": model_key,
                    "metadata": metadata_key,
                    "evaluation": evaluation_key,
                },
            }
        ),
    )
    print("[deploy_model] Completed")


if __name__ == "__main__":
    main()
