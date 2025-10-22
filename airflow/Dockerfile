FROM apache/airflow:2.9.2-python3.10

# Install build tools if needed for some wheels (as root), then use constraints for python deps
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow
ARG AIRFLOW_VERSION=2.9.2
ARG PYTHON_VERSION=3.10
ENV AIRFLOW_VERSION=${AIRFLOW_VERSION} PYTHON_VERSION=${PYTHON_VERSION}
COPY requirements.txt /requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir \
       -r /requirements.txt \
       --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
