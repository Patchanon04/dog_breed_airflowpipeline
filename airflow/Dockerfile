FROM apache/airflow:2.9.2-python3.10

USER root
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt \
    && chown -R airflow:0 /opt/airflow \
    && chmod -R g=u /opt/airflow

USER airflow
