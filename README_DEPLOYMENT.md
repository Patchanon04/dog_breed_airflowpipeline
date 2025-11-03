# Face Recognition Pipeline - EC2 Deployment

This guide describes how to deploy a production-ready Airflow-based Face Recognition pipeline on Ubuntu 22.04 (EC2).

## Prerequisites
- Ubuntu 22.04 EC2 with at least t3.large, 30GB disk
- Security Group:
  - Inbound 22 (SSH) from your IP
  - Inbound 8081 (Airflow UI) from your IP
- AWS CLI configured (access key, secret, region)
- GitHub repo with pipeline code following the required structure

## Files
- `deploy_ec2.sh` - main deployment script
- `config.env` - configuration (keep secret!)
- `docker-compose.yml` - Docker services
- `monitor.sh` - real-time monitoring

## Quick Start
```bash
sudo chmod +x deploy_ec2.sh monitor.sh
sudo ./deploy_ec2.sh
```

## S3 Structure
- `s3://<S3_BUCKET>/<S3_PREFIX>/raw/`
- `s3://<S3_BUCKET>/<S3_PREFIX>/models/`
- `s3://<S3_BUCKET>/<S3_PREFIX>/reports/`

## Troubleshooting
- Logs
  - Deployment: `/var/log/face-recognition/deploy.log`
  - Airflow: `<PROJECT_DIR>/airflow/logs/`
- Restart services
```bash
docker compose ps
docker compose logs -f airflow-webserver airflow-scheduler postgres
docker compose restart airflow-webserver airflow-scheduler
```

## Security Notes
- Never commit `config.env` with real secrets
- Rotate `FERNET_KEY` and Airflow secret keys periodically
- Consider enabling TLS/SSL for Airflow behind reverse proxy
- Use CloudWatch/Prometheus/Grafana for monitoring in production
