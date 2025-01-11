from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator

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

    data_preprocessing = SparkSubmitOperator(
        task_id='submit_spark_job',
        application='/src/data_preprocessing.py',  
        conn_id='spark_local',
        application_args=[
            "--timestamp",
            "{{ ts }}"  # Pass the current timestamp from Airflow context
        ],
    )
    
    train_model = BashOperator(
        task_id='train_model',
        bash_command='python3 /src/model_training.py',
        application_args=[
            "--timestamp",
            "{{ ts }}"  # Pass the current timestamp from Airflow context
        ],
    )
    
    detect_drift = BashOperator(
        task_id='detect_drift',
        bash_command='python3 /src/drift_monitoring.py',
        application_args=[
            "--timestamp",
            "{{ ts }}"  # Pass the current timestamp from Airflow context
        ],
    )
    

    data_preprocessing >> train_model >> detect_drift
