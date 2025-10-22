FROM apache/airflow:2.9.2-python3.10

# Install additional python dependencies as the airflow user
USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
