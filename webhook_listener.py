#!/usr/bin/env python3
import hmac
import hashlib
import os
import subprocess
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, abort

LOG_DIR = os.environ.get("LOG_DIR", "/var/log/face-recognition")
LOG_FILE = os.path.join(LOG_DIR, "webhook.log")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/home/ubuntu/face_recognition_project")
BRANCH = os.environ.get("GIT_BRANCH", "main")

app = Flask(__name__)

os.makedirs(LOG_DIR, exist_ok=True)
handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger('webhook')
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def verify_signature(payload: bytes, signature_header: str) -> bool:
    if not WEBHOOK_SECRET:
        logger.error("WEBHOOK_SECRET not set")
        return False
    if not signature_header or not signature_header.startswith('sha256='):
        return False
    signature = signature_header.split('=')[1]
    digest = hmac.new(WEBHOOK_SECRET.encode('utf-8'), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature)


def run(cmd: str, cwd: str = None) -> int:
    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        logger.error(f"Command failed: {cmd} (code={result.returncode})")
    return result.returncode


@app.route('/webhook', methods=['POST'])
def webhook():
    event = request.headers.get('X-GitHub-Event', '')
    sig = request.headers.get('X-Hub-Signature-256', '')
    payload = request.get_data() or b''

    if not verify_signature(payload, sig):
        logger.warning("Invalid signature")
        abort(401)

    if event != 'push':
        logger.info(f"Ignoring event: {event}")
        return {"status": "ignored", "reason": "not a push event"}, 200

    data = request.json or {}
    ref = data.get('ref', '')
    if ref != f"refs/heads/{BRANCH}":
        logger.info(f"Ignoring ref: {ref}")
        return {"status": "ignored", "reason": "not target branch"}, 200

    logger.info("Valid push event received. Updating...")

    # Pull latest code and restart services
    steps = [
        f"git fetch --all",
        f"git checkout {BRANCH}",
        f"git reset --hard origin/{BRANCH}",
        "docker compose pull",
        "docker compose build --no-cache airflow-webserver airflow-scheduler airflow-init || true",
        "docker compose up -d --remove-orphans",
    ]

    for step in steps:
        rc = run(step, cwd=PROJECT_DIR)
        if rc != 0:
            return {"status": "error", "step": step}, 500

    logger.info("Update successful")
    return {"status": "ok"}, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('WEBHOOK_LISTENER_PORT', '9000')))
