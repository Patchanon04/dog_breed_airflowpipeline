import os
import boto3


def _resolve_region() -> str:
    # Prefer AWS_DEFAULT_REGION; fallback to AWS_S3_REGION_NAME
    return os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_S3_REGION_NAME") or "ap-southeast-1"


def get_s3_client():
    session = boto3.session.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=_resolve_region(),
    )
    return session.client("s3")
