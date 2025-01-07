from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    dag_id='spark_submit_example',
    default_args=default_args,
    description='Run a Spark job using SparkSubmitOperator',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
) as dag:

    spark_submit_task = SparkSubmitOperator(
        task_id='submit_spark_job',
        application='/opt/airflow/src/data_preprocessing.py',  # Replace with the path to your Spark job
        conn_id='spark_local',
    )

    spark_submit_task
