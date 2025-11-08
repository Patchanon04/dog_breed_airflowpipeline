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
    dag_id="retrain_tumor_classification_tf",
    default_args=DEFAULT_ARGS,
    description="TensorFlow retrain pipeline derived from brain_tumor_clf notebooks",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["healthcare", "tensorflow"],
) as dag:

    env_exports = "export PYTHONUNBUFFERED=1; "

    load_data = BashOperator(
        task_id="load_data",
        bash_command=env_exports + "python /opt/airflow/scripts/load_data.py",
    )

    clean_data = BashOperator(
        task_id="clean_data",
        bash_command=env_exports + "python /opt/airflow/scripts/clean_data_tf.py",
    )

    train_model_tf = BashOperator(
        task_id="train_model_tf",
        bash_command=env_exports + "python /opt/airflow/scripts/train_model_tf.py",
    )

    evaluate_model_tf = BashOperator(
        task_id="evaluate_model_tf",
        bash_command=env_exports + "python /opt/airflow/scripts/evaluate_model_tf.py",
    )

    test_model_tf = BashOperator(
        task_id="test_model_tf",
        bash_command=env_exports + "python /opt/airflow/scripts/test_model_tf.py",
    )

    deploy_model_tf = BashOperator(
        task_id="deploy_model_tf",
        bash_command=env_exports + "python /opt/airflow/scripts/deploy_model_tf.py",
    )

    load_data >> clean_data >> train_model_tf >> evaluate_model_tf >> test_model_tf >> deploy_model_tf
