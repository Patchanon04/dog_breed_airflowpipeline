#!/bin/bash
# monitor.sh - Real-time monitoring

while true; do
    clear
    echo "=== Face Recognition Pipeline Status ==="
    echo "Time: $(date)"
    echo ""

    echo "--- Docker Containers ---"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

    echo ""
    echo "--- Airflow Health ---"
    curl -s http://localhost:8081/health | jq .

    echo ""
    echo "--- Recent Logs ---"
    docker compose logs --tail=10 airflow-scheduler

    echo ""
    echo "--- DAG Status ---"
    docker exec airflow-webserver airflow dags list-runs -d face_recognition_pipeline --limit 5 || true

    sleep 30
done
