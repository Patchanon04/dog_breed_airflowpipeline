#!/usr/bin/env python3
import os
import time
print("[train_model] Training model...")
# Simulate work
time.sleep(2)
# Save a dummy artifact path for downstream tasks
artifact_path = "/opt/airflow/logs/model_artifact.bin"
open(artifact_path, "w").write("dummy-model")
print(f"[train_model] Artifact saved to {artifact_path}")
print("[train_model] Completed")
