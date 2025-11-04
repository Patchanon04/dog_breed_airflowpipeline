#!/usr/bin/env python3
import os
import sys
from utils.s3_helper import get_s3_client

BUCKET = os.environ.get("S3_BUCKET")
PREFIX = os.environ.get("S3_PREFIX", "retrain-brain-tumor-model")


def main():
    print("[load_data] Starting load from S3...")
    s3 = get_s3_client()
    if not BUCKET:
        print("[load_data] S3_BUCKET not set", file=sys.stderr)
        sys.exit(1)
    # Example: list objects under raw/
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/raw/")
    count = len(resp.get("Contents", []))
    print(f"[load_data] Found {count} raw objects in s3://{BUCKET}/{PREFIX}/raw/")
    print("[load_data] Completed")

if __name__ == "__main__":
    main()
