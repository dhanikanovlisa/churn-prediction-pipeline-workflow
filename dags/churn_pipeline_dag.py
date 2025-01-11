from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator

def print_no_drift_message():
    print("No model drift detected")

def choose_next_task(task_instance, **kwargs):
    drift_detected = task_instance.xcom_pull(task_ids='detect_drift')
    return 'train_model' if drift_detected == 'True' else 'no_model_drift'

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
    )
    
    train_model = BashOperator(
        task_id='train_model',
        bash_command='python3 /src/model_training.py',
    )
    
    detect_drift = BashOperator(
        task_id='detect_drift',
        bash_command='python3 /src/drift_monitoring.py',
    )

    no_model_drift = PythonOperator(
        task_id='no_model_drift',
        python_callable=print_no_drift_message
    )

    branch_decision = BranchPythonOperator(
        task_id='branch_decision',
        python_callable=choose_next_task,
        provide_context=True
    )

    data_preprocessing >> detect_drift >> branch_decision
    branch_decision >> train_model
    branch_decision >> no_model_drift

