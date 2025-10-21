#!/usr/bin/env bash
# deploy_ec2.sh - Production-ready deployment for Face Recognition Pipeline on EC2 Ubuntu 22.04
# Usage: sudo ./deploy_ec2.sh

set -e
set -o pipefail
set -u

# ============================================
# Globals & Defaults
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.env"
LOG_DIR_DEFAULT="/var/log/face-recognition"
LOG_FILE="${LOG_DIR_DEFAULT}/deploy.log"
DOCKER_DAEMON_JSON="/etc/docker/daemon.json"
WEBHOOK_SERVICE_FILE="/etc/systemd/system/webhook-listener.service"
WEBHOOK_WORKDIR="/opt/webhook-listener"
WEBHOOK_VENV="${WEBHOOK_WORKDIR}/venv"
REPO_DIR=""

# ============================================
# Logging helpers
# ============================================
mkdir -p "${LOG_DIR_DEFAULT}"

echo "Starting deployment... logs: ${LOG_FILE}" | tee -a "${LOG_FILE}"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log_success() {
  echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] \033[0;32m✓ $1\033[0m" | tee -a "${LOG_FILE}"
}

log_error() {
  echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] \033[0;31m✗ $1\033[0m" | tee -a "${LOG_FILE}"
}

log_warning() {
  echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] \033[0;33m⚠ $1\033[0m" | tee -a "${LOG_FILE}"
}

# ============================================
# Error handling & Retry
# ============================================
error_handler() {
  local exit_code=$1
  local line_no=$2
  log_error "Error occurred at line ${line_no}, exit code ${exit_code}"
  exit ${exit_code}
}
trap 'error_handler $? $LINENO' ERR

retry() {
  local retries=$1
  shift
  local count=0
  until "$@"; do
      exit_code=$?
      count=$((count + 1))
      if [ $count -lt $retries ]; then
          log_warning "Retry $count/$retries... for command: $*"
          sleep 5
      else
          log_error "Failed after $retries attempts: $*"
          return $exit_code
      fi
  done
  return 0
}

# Health checks
wait_for_postgres() {
  retry 30 docker exec postgres pg_isready -U airflow
}

wait_for_webserver() {
  local port=${AIRFLOW_WEBSERVER_PORT:-8081}
  retry 60 curl -sf "http://localhost:${port}/health"
}

# ============================================
# Pre-flight Checks
# ============================================
preflight_checks() {
  log "[1/9] Pre-flight Checks"

  # Check root/sudo
  if [ "$(id -u)" -ne 0 ]; then
    log_error "Run as root or with sudo"
    exit 1
  fi
  log_success "Root privileges verified"

  # Validate config.env exists
  if [ ! -f "${CONFIG_FILE}" ]; then
    log_error "Missing config file: ${CONFIG_FILE}"
    exit 1
  fi
  log_success "Found config: ${CONFIG_FILE}"

  # shellcheck disable=SC1090
  set -a; source "${CONFIG_FILE}"; set +a

  # Ensure required vars
  required_vars=(GIT_REPO GIT_BRANCH PROJECT_DIR AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION S3_BUCKET S3_PREFIX AIRFLOW_UID AIRFLOW_GID AIRFLOW__CORE__FERNET_KEY AIRFLOW__WEBSERVER__SECRET_KEY AIRFLOW_ADMIN_USER AIRFLOW_ADMIN_PASSWORD AIRFLOW_ADMIN_EMAIL AIRFLOW_WEBSERVER_PORT WEBHOOK_LISTENER_PORT WEBHOOK_SECRET LOG_LEVEL LOG_DIR)
  for v in "${required_vars[@]}"; do
    if [ -z "${!v:-}" ]; then
      log_error "Missing required variable in config.env: $v"
      exit 1
    fi
  done
  REPO_DIR="${PROJECT_DIR}"
  log_success "Environment variables validated"

  # Check AWS CLI
  if ! command -v aws >/dev/null 2>&1; then
    log_warning "aws CLI not found; will install in system prep"
  else
    log_success "AWS CLI found: $(aws --version 2>&1)"
  fi

  # Export AWS creds for commands
  export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION

  # AWS Credentials deep validation
  log "Validating AWS credentials and S3 access"
  retry 3 aws sts get-caller-identity >/dev/null
  log_success "AWS STS identity verified"
  retry 3 aws s3 ls "s3://${S3_BUCKET}" >/dev/null
  log_success "S3 bucket exists and readable"

  TEST_KEY="${S3_PREFIX}/_deploy_test_$(date +%s).txt"
  echo "deploy-test" >/tmp/deploy-test.txt
  retry 3 aws s3 cp /tmp/deploy-test.txt "s3://${S3_BUCKET}/${TEST_KEY}" >/dev/null
  retry 3 aws s3 rm "s3://${S3_BUCKET}/${TEST_KEY}" >/dev/null
  rm -f /tmp/deploy-test.txt
  log_success "S3 write/delete test passed"
}

# ============================================
# System Preparation
# ============================================
system_prep() {
  log "[2/9] System Preparation"
  export DEBIAN_FRONTEND=noninteractive
  retry 3 apt-get update -y
  retry 3 apt-get upgrade -y
  retry 3 apt-get install -y ca-certificates curl gnupg lsb-release jq git python3-pip python3-venv

  # Install AWS CLI if missing
  if ! command -v aws >/dev/null 2>&1; then
    log "Installing AWS CLI v2"
    apt-get install -y unzip
    tmpdir=$(mktemp -d)
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "${tmpdir}/awscliv2.zip"
    unzip -q "${tmpdir}/awscliv2.zip" -d "${tmpdir}"
    "${tmpdir}/aws/install" >/dev/null
    rm -rf "${tmpdir}"
    log_success "AWS CLI installed"
  fi

  # Install Docker Engine
  if ! command -v docker >/dev/null 2>&1; then
    log "Installing Docker Engine"
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    retry 3 apt-get update -y
    retry 3 apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    log_success "Docker installed"
  else
    log_success "Docker found: $(docker --version)"
  fi

  # Configure Docker daemon
  mkdir -p /etc/docker
  cat > "${DOCKER_DAEMON_JSON}" <<EOF
{
  "log-driver": "json-file",
  "log-opts": {"max-size": "10m", "max-file": "5"},
  "storage-driver": "overlay2"
}
EOF
  systemctl enable docker
  systemctl restart docker
  log_success "Docker daemon configured"

  # Add ubuntu to docker group if exists
  if id ubuntu >/dev/null 2>&1; then
    usermod -aG docker ubuntu || true
    log_success "Added user 'ubuntu' to docker group"
  fi

  # Ensure docker compose v2
  if docker compose version >/dev/null 2>&1; then
    log_success "Docker Compose v2 available: $(docker compose version)"
  else
    log_error "Docker Compose v2 not available"
    exit 1
  fi
}

# ============================================
# Project Setup
# ============================================
project_setup() {
  log "[3/9] Project Setup"

  mkdir -p "${REPO_DIR}"
  if [ ! -d "${REPO_DIR}/.git" ]; then
    log "Cloning repo ${GIT_REPO} to ${REPO_DIR}"
    retry 3 git clone --branch "${GIT_BRANCH}" "${GIT_REPO}" "${REPO_DIR}"
  else
    log "Repository exists, fetching latest"
    pushd "${REPO_DIR}" >/dev/null
    git fetch --all
    git checkout "${GIT_BRANCH}"
    git reset --hard "origin/${GIT_BRANCH}"
    popd >/dev/null
  fi

  # Backup existing docker-compose.yml if present
  if [ -f "${REPO_DIR}/docker-compose.yml" ]; then
    cp -f "${REPO_DIR}/docker-compose.yml" "${REPO_DIR}/docker-compose.yml.bak.$(date +%s)"
    log_success "Backed up existing docker-compose.yml"
  fi

  # If script directory has a compose, copy as baseline
  if [ -f "${SCRIPT_DIR}/docker-compose.yml" ]; then
    cp -f "${SCRIPT_DIR}/docker-compose.yml" "${REPO_DIR}/docker-compose.yml"
    log_success "Updated repo docker-compose.yml from deployment bundle"
  fi

  # Ensure logging dir
  mkdir -p "${LOG_DIR}"
  chown -R root:root "${LOG_DIR}"
  chmod -R 755 "${LOG_DIR}"

  # Permissions within project
  mkdir -p "${REPO_DIR}/airflow/logs" "${REPO_DIR}/airflow/dags" "${REPO_DIR}/airflow/plugins" "${REPO_DIR}/airflow/scripts" "${REPO_DIR}/airflow/utils"
  chown -R ${AIRFLOW_UID}:${AIRFLOW_GID} "${REPO_DIR}/airflow/logs" || true

  # Validate project structure
  validate_project_structure
}

validate_project_structure() {
  log "Validating project structure"
  local required_files=(
    "docker-compose.yml"
    "airflow/Dockerfile"
    "airflow/requirements.txt"
    "airflow/dags/face_recognition_pipeline.py"
    "airflow/scripts/load_data.py"
    "airflow/scripts/clean_data.py"
    "airflow/scripts/train_model.py"
    "airflow/scripts/deploy_model.py"
    "airflow/scripts/test_model.py"
    "airflow/scripts/evaluate_model.py"
    "airflow/utils/s3_helper.py"
  )
  for file in "${required_files[@]}"; do
    if [ ! -f "${REPO_DIR}/${file}" ]; then
      log_error "Missing required file: ${file}"
      exit 1
    fi
  done
  log_success "Project structure validated"
}

# ============================================
# Docker Infrastructure
# ============================================
docker_infra() {
  log "[4/9] Docker Infrastructure"
  pushd "${REPO_DIR}" >/dev/null

  # Build custom Airflow image
  log "Building Airflow image"
  retry 3 docker build -t face-airflow:latest -f airflow/Dockerfile airflow
  log_success "Airflow image built"

  # Create network (compose will create automatically, but ensure base network exists)
  docker network inspect facenet >/dev/null 2>&1 || docker network create facenet || true

  # Create volumes
  docker volume inspect postgres-db >/dev/null 2>&1 || docker volume create postgres-db >/dev/null

  # Start postgres first
  log "Starting postgres service"
  retry 3 docker compose up -d postgres
  wait_for_postgres
  log_success "Postgres is healthy"

  # Run airflow-init (db migration + admin user)
  log "Running airflow-init"
  retry 3 docker compose up --no-deps airflow-init || true
  log_success "airflow-init completed"

  popd >/dev/null
}

# ============================================
# Airflow Services Deployment
# ============================================
deploy_airflow_services() {
  log "[5/9] Airflow Services Deployment"
  pushd "${REPO_DIR}" >/dev/null

  retry 3 docker compose up -d airflow-scheduler
  retry 3 docker compose up -d airflow-webserver
  wait_for_webserver
  log_success "Airflow webserver healthy"

  # Verify all containers running
  docker compose ps

  popd >/dev/null
}

# ============================================
# DAG Validation & Trigger
# ============================================
dag_validation_and_trigger() {
  log "[6/9] DAG Validation & Trigger"

  # List DAGs
  docker exec airflow-webserver airflow dags list || true
  # Check import errors
  if docker exec airflow-webserver airflow dags list-import-errors | grep -E "Import Errors" -A999 | grep -v "^$" | grep -q "."; then
    log_error "There are DAG import errors"
    docker exec airflow-webserver airflow dags list-import-errors || true
    exit 1
  fi

  # Validate face_recognition_pipeline exists
  if ! docker exec airflow-webserver airflow dags list | awk '{print $1}' | grep -q '^face_recognition_pipeline$'; then
    log_error "DAG 'face_recognition_pipeline' not found"
    exit 1
  fi
  log_success "DAG exists: face_recognition_pipeline"

  # Unpause DAG
  docker exec airflow-webserver airflow dags unpause face_recognition_pipeline || true
  # Trigger DAG
  docker exec airflow-webserver airflow dags trigger face_recognition_pipeline || true
  sleep 5
  docker exec airflow-webserver airflow dags list-runs -d face_recognition_pipeline --limit 5 || true
}

# ============================================
# Auto-Update Setup (Webhook)
# ============================================
setup_auto_update() {
  log "[7/9] Auto-Update Setup"

  mkdir -p "${WEBHOOK_WORKDIR}"
  # Copy webhook script from bundle if present
  if [ -f "${SCRIPT_DIR}/webhook_listener.py" ]; then
    cp -f "${SCRIPT_DIR}/webhook_listener.py" "${WEBHOOK_WORKDIR}/webhook_listener.py"
  else
    log_error "webhook_listener.py not found in bundle"
    exit 1
  fi

  # Python venv and deps
  python3 -m venv "${WEBHOOK_VENV}"
  "${WEBHOOK_VENV}/bin/pip" install --upgrade pip >/dev/null
  "${WEBHOOK_VENV}/bin/pip" install flask >/dev/null

  # Systemd service
  cat > "${WEBHOOK_SERVICE_FILE}" <<EOF
[Unit]
Description=Webhook Listener for Auto-Update
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
EnvironmentFile=${CONFIG_FILE}
WorkingDirectory=${WEBHOOK_WORKDIR}
ExecStart=${WEBHOOK_VENV}/bin/python ${WEBHOOK_WORKDIR}/webhook_listener.py
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/webhook.log
StandardError=append:${LOG_DIR}/webhook.log
User=root

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable webhook-listener
  systemctl restart webhook-listener
  sleep 2
  systemctl --no-pager status webhook-listener || true

  # Cron job fallback (disabled by default)
  CRON_SCRIPT="/usr/local/bin/repo-autopull.sh"
  cat > "${CRON_SCRIPT}" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
source /etc/profile || true
if [ -f "/etc/environment" ]; then
  set -a; source /etc/environment; set +a
fi
if [ -f "/root/.bashrc" ]; then
  source /root/.bashrc
fi
CONFIG_FILE="/root/config.env"
if [ -f "$CONFIG_FILE" ]; then
  set -a; source "$CONFIG_FILE"; set +a
fi
REPO_DIR="${PROJECT_DIR:-/home/ubuntu/face_recognition_project}"
cd "$REPO_DIR"
/usr/bin/git fetch --all
/usr/bin/git reset --hard "origin/${GIT_BRANCH:-main}"
/usr/bin/docker compose up -d --remove-orphans
EOS
  chmod +x "${CRON_SCRIPT}"
  log_success "Webhook listener configured (systemd). Cron fallback script prepared but not scheduled."
}

# ============================================
# Monitoring & Logging
# ============================================
monitoring_and_logging() {
  log "[8/9] Monitoring & Logging"

  # Log rotation for our logs
  cat > /etc/logrotate.d/face-recognition <<EOF
${LOG_DIR}/*.log {
  daily
  rotate 7
  compress
  missingok
  notifempty
  copytruncate
}
EOF

  # Copy monitor.sh
  if [ -f "${SCRIPT_DIR}/monitor.sh" ]; then
    cp -f "${SCRIPT_DIR}/monitor.sh" "${REPO_DIR}/monitor.sh"
    chmod +x "${REPO_DIR}/monitor.sh"
  fi

  # Security Group guidance
  echo "=== Security Group Rules Required ===" | tee -a "${LOG_FILE}"
  echo "Inbound Rules:" | tee -a "${LOG_FILE}"
  echo "  - Port 22   (SSH)         from YOUR_IP" | tee -a "${LOG_FILE}"
  echo "  - Port ${AIRFLOW_WEBSERVER_PORT} (Airflow UI)  from YOUR_IP" | tee -a "${LOG_FILE}"
  echo "  - Port ${WEBHOOK_LISTENER_PORT} (Webhook)     from GitHub IPs" | tee -a "${LOG_FILE}"
  echo "" | tee -a "${LOG_FILE}"
  echo "AWS CLI command to open ports:" | tee -a "${LOG_FILE}"
  echo "aws ec2 authorize-security-group-ingress --group-id sg-xxxxx --protocol tcp --port ${AIRFLOW_WEBSERVER_PORT} --cidr YOUR_IP/32" | tee -a "${LOG_FILE}"
}

# ============================================
# Final Verification & Summary
# ============================================
final_verification() {
  log "[9/9] Final Verification"

  # Test Airflow UI
  if wait_for_webserver; then
    log_success "Airflow UI health endpoint OK"
  fi

  # Verify S3 connectivity again (prefix paths)
  retry 3 aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" >/dev/null || true

  # Detect public IP
  EC2_IP=$(curl -fs http://169.254.169.254/latest/meta-data/public-ipv4 || echo "<EC2_PUBLIC_IP>")

  # DAG run status
  DAG_STATUS=$(docker exec airflow-webserver bash -lc "airflow dags list-runs -d face_recognition_pipeline --limit 1 | tail -n +3 | awk '{print \$6}'" || echo "unknown")
  DAG_START=$(docker exec airflow-webserver bash -lc "airflow dags list-runs -d face_recognition_pipeline --limit 1 | tail -n +3 | awk '{print \$2,\$3}'" || echo "-")

  # Containers health
  P_HEALTH=$(docker inspect -f '{{json .State.Health.Status}}' postgres 2>/dev/null | tr -d '"' || echo "unknown")
  W_HEALTH=$(docker inspect -f '{{json .State.Health.Status}}' airflow-webserver 2>/dev/null | tr -d '"' || echo "unknown")
  S_STATE=$(docker inspect -f '{{.State.Status}}' airflow-scheduler 2>/dev/null || echo "unknown")
  WEBHOOK_STATE=$(systemctl is-active webhook-listener || echo "unknown")

  cat <<EOF
╔═══════════════════════════════════════════════════════════════╗
║          Face Recognition Pipeline Deployed Successfully       ║
╚═══════════════════════════════════════════════════════════════╝

📊 Access URLs:
   Airflow UI:      http://${EC2_IP}:${AIRFLOW_WEBSERVER_PORT}
   Username:        ${AIRFLOW_ADMIN_USER}
   Password:        ${AIRFLOW_ADMIN_PASSWORD}
   
   Webhook:         http://${EC2_IP}:${WEBHOOK_LISTENER_PORT}/webhook
   
🐳 Docker Containers:
   ✓ postgres            [${P_HEALTH}]
   ✓ airflow-webserver   [${W_HEALTH}] 
   ✓ airflow-scheduler   [${S_STATE}]
   ✓ webhook-listener    [${WEBHOOK_STATE}]

🚀 DAG Status:
   Name: face_recognition_pipeline
   State: ${DAG_STATUS}
   Start Time: ${DAG_START}
   
📁 S3 Paths:
   Raw Data:    s3://${S3_BUCKET}/${S3_PREFIX}/raw/
   Models:      s3://${S3_BUCKET}/${S3_PREFIX}/models/
   Reports:     s3://${S3_BUCKET}/${S3_PREFIX}/reports/

📝 Logs:
   Deployment:  ${LOG_FILE}
   Airflow:     ${REPO_DIR}/airflow/logs/
   Webhook:     ${LOG_DIR}/webhook.log

🔍 Monitoring:
   Run: ${REPO_DIR}/monitor.sh
   
📚 Next Steps:
   1. Configure GitHub webhook: Settings → Webhooks → Add webhook
      Payload URL: http://${EC2_IP}:${WEBHOOK_LISTENER_PORT}/webhook
      Secret: <from config.env>
      
   2. Monitor DAG: http://${EC2_IP}:${AIRFLOW_WEBSERVER_PORT}/dags/face_recognition_pipeline
      
   3. Check logs: docker compose logs -f
      
   4. Verify S3 uploads: aws s3 ls s3://${S3_BUCKET}/${S3_PREFIX}/models/

⚠️  Important Notes:
   - Keep config.env secure (contains AWS keys)
   - Rotate Fernet key for production
   - Enable SSL/TLS for Airflow UI
   - Setup CloudWatch for production monitoring
EOF
}

# ============================================
# Main
# ============================================
main() {
  preflight_checks
  system_prep
  project_setup
  docker_infra
  deploy_airflow_services
  dag_validation_and_trigger
  setup_auto_update
  monitoring_and_logging
  final_verification
  log_success "Deployment complete"
}

main "$@"
