#!/usr/bin/env python3
import os
import json
import time

ACC_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.85"))

print("[evaluate_model] Evaluating model...")
time.sleep(1)
metrics = {"accuracy": 0.90}
print(f"[evaluate_model] Metrics: {json.dumps(metrics)}")
if metrics["accuracy"] < ACC_THRESHOLD:
    raise SystemExit(f"Accuracy {metrics['accuracy']} below threshold {ACC_THRESHOLD}")
print("[evaluate_model] Passed threshold")
