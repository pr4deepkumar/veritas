"""Daily ingestion: parse new PDFs from S3, then index changed documents into Pinecone.

Both steps are incremental and idempotent, so retries and re-runs only redo unfinished work.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:  # Airflow 3
    from airflow.sdk import DAG
except ImportError:  # Airflow 2.x
    from airflow import DAG

try:  # Airflow 3 ships PythonOperator in the standard provider
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.python import PythonOperator


# The pipeline modules are imported inside the tasks so that parsing this file stays fast and
# needs neither AWS nor Pinecone configuration.
def ingest_from_s3() -> None:
    from pipelines.ingest_from_s3 import run

    run()


def index_to_pinecone() -> None:
    from pipelines.index_to_pinecone import run

    run()


DEFAULT_ARGS = {
    "owner": "veritas",
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
}

with DAG(
    dag_id="document_research_pipeline",
    description="Parse new PDFs from S3 and index changed documents into Pinecone",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,  # runs share the index manifest; never overlap them
    tags=["research", "rag"],
) as dag:
    parse_task = PythonOperator(task_id="ingest_from_s3", python_callable=ingest_from_s3)
    index_task = PythonOperator(task_id="index_to_pinecone", python_callable=index_to_pinecone)

    parse_task >> index_task
