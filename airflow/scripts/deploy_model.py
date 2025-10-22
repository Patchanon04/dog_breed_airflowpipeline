#!/usr/bin/env python3
import os
from utils.s3_helper import get_s3_client

BUCKET = os.environ.get("S3_BUCKET")
PREFIX = os.environ.get("S3_PREFIX", "facerec")
MODEL_NAME = os.environ.get("MODEL_NAME", "face_recognition_model")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0")

print("[deploy_model] Uploading artifact to S3...")
s3 = get_s3_client()
artifact_path = "/opt/airflow/logs/model_artifact.bin"
key = f"{PREFIX}/models/{MODEL_NAME}/{MODEL_VERSION}/model_artifact.bin"
if not os.path.exists(artifact_path):
    raise SystemExit("Artifact missing from training step")
s3.upload_file(artifact_path, BUCKET, key)
print(f"[deploy_model] Uploaded to s3://{BUCKET}/{key}")
