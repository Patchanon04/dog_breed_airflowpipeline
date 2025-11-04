from __future__ import annotations
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retrain_tumor_classification",
    default_args=DEFAULT_ARGS,
    description="Retrain tumor classification end-to-end pipeline",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["healthcare"],
) as dag:

    env_exports = "export PYTHONUNBUFFERED=1; "

    load_data = BashOperator(
        task_id="load_data",
        bash_command=env_exports + "python /opt/airflow/scripts/load_data.py",
    )

    clean_data = BashOperator(
        task_id="clean_data",
        bash_command=env_exports + "python /opt/airflow/scripts/clean_data.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=env_exports + "python /opt/airflow/scripts/train_model.py",
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=env_exports + "python /opt/airflow/scripts/evaluate_model.py",
    )

    test_model = BashOperator(
        task_id="test_model",
        bash_command=env_exports + "python /opt/airflow/scripts/test_model.py",
    )

    deploy_model = BashOperator(
        task_id="deploy_model",
        bash_command=env_exports + "python /opt/airflow/scripts/deploy_model.py",
    )

    load_data >> clean_data >> train_model >> evaluate_model >> test_model >> deploy_model
