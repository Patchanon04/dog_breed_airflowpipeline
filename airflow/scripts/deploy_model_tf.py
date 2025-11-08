#!/usr/bin/env python3
"""อัปโหลดโมเดล TensorFlow retrain line 2 และเมตริกขึ้น S3."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from utils import s3_helper
from utils.brain_tumor_training import resolve_output_dir
from utils.brain_tumor_training_tf import (
    TF_EVALUATION_FILE,
    TF_METADATA_FILE,
    load_tf_metadata,
)

DEFAULT_ARTIFACT_PREFIX = "retrain-brain-tumor-model/tf-models"


def _strip_join(*parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if part and part.strip("/")]
    return "/".join(cleaned)


def main() -> None:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        print("[deploy_model_tf] ต้องระบุ S3_BUCKET", file=sys.stderr)
        raise SystemExit(1)

    artifact_prefix = (
        os.environ.get("MODEL_ARTIFACT_PREFIX")
        or os.environ.get("S3_PREFIX")
        or DEFAULT_ARTIFACT_PREFIX
    )
    model_version = os.environ.get("MODEL_VERSION", "latest")

    output_dir = resolve_output_dir()

    try:
        metadata = load_tf_metadata(output_dir)
    except FileNotFoundError as exc:
        print(f"[deploy_model_tf] ไม่พบ metadata: {exc}", file=sys.stderr)
        raise SystemExit(1)

    model_paths = metadata.get("model_paths", {})
    saved_model_dir = Path(model_paths.get("saved_model_dir", ""))
    keras_path = Path(model_paths.get("keras_path", ""))
    h5_path = Path(model_paths.get("h5_path", ""))

    if not saved_model_dir.exists():
        print(f"[deploy_model_tf] ไม่พบ SavedModel directory ที่ {saved_model_dir}", file=sys.stderr)
        raise SystemExit(1)
    if not keras_path.exists():
        print(f"[deploy_model_tf] ไม่พบไฟล์ Keras ที่ {keras_path}", file=sys.stderr)
        raise SystemExit(1)
    if not h5_path.exists():
        print(f"[deploy_model_tf] ไม่พบไฟล์ H5 ที่ {h5_path}", file=sys.stderr)
        raise SystemExit(1)

    evaluation_path = Path(output_dir) / TF_EVALUATION_FILE
    if not evaluation_path.exists():
        print(
            f"[deploy_model_tf] ไม่พบ {TF_EVALUATION_FILE} กรุณารัน evaluate_model_tf ก่อน",
            file=sys.stderr,
        )
        raise SystemExit(1)

    metadata_path = Path(output_dir) / TF_METADATA_FILE

    archive_target = saved_model_dir.with_suffix("")
    archive_file = Path(shutil.make_archive(str(archive_target), "zip", root_dir=saved_model_dir))

    client = s3_helper.get_s3_client()
    key_prefix = _strip_join(artifact_prefix, model_version)

    artifacts = {
        "saved_model_zip": (archive_file, _strip_join(key_prefix, archive_file.name)),
        "keras_model": (keras_path, _strip_join(key_prefix, keras_path.name)),
        "h5_model": (h5_path, _strip_join(key_prefix, h5_path.name)),
        "metadata": (metadata_path, _strip_join(key_prefix, TF_METADATA_FILE)),
        "evaluation": (evaluation_path, _strip_join(key_prefix, TF_EVALUATION_FILE)),
    }

    uploaded = {}
    for label, (path, key) in artifacts.items():
        if not path.exists():
            print(f"[deploy_model_tf] ข้าม {label}: ไม่พบไฟล์ {path}")
            continue
        print(f"[deploy_model_tf] อัปโหลด {label} ไปที่ s3://{bucket}/{key}")
        client.upload_file(str(path), bucket, key)
        uploaded[label] = key

    # ลบไฟล์ zip ชั่วคราวเพื่อไม่ให้รกพื้นที่
    archive_file.unlink(missing_ok=True)

    print(
        "[deploy_model_tf] อัปโหลดสำเร็จ:",
        json.dumps({"bucket": bucket, "keys": uploaded}, indent=2, ensure_ascii=False),
    )
    print("[deploy_model_tf] เสร็จสิ้น")


if __name__ == "__main__":
    main()
