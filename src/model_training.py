from datetime import datetime, timedelta

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev
from pyspark.sql.types import DoubleType, IntegerType
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PROCESSED_PATH = "../../data/processed"

def train_and_log_models(**kwargs):
    timestamp = kwargs['ti'].xcom_pull(task_ids='data_processing', key='timestamp')

    X_train = pd.read_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_x_train.csv")
    y_train = pd.read_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_y_train.csv")
    X_test = pd.read_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_x_test.csv")
    y_test = pd.read_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_y_test.csv")

    # Model 1: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_preds)

    mlflow.set_experiment("Default")
    mlflow.set_tracking_uri("http://mlflow:5000")

    signature = infer_signature(X_train, rf_preds)
    with mlflow.start_run(run_name="Random_Forest"):
        mlflow.log_param("timestamp", timestamp)
        mlflow.sklearn.log_model(rf_model, "Random_Forest_Model", signature=signature)
        mlflow.log_metric("accuracy", rf_accuracy)
    
    # Model 2: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_preds)
    
    signature = infer_signature(X_train, lr_preds)
    with mlflow.start_run(run_name="Logistic_Regression"):
        mlflow.log_param("timestamp", timestamp)
        mlflow.sklearn.log_model(lr_model, "Logistic_Regression_Model", signature=signature)
        mlflow.log_metric("accuracy", lr_accuracy)
    
    client = MlflowClient()
    recent_runs = client.search_runs(
        experiment_ids=["0"],
        order_by=["start_time DESC"],
        max_results=2
    )

    if not recent_runs:
        raise ValueError("No runs found for this experiment.")
    
    best_run = max(recent_runs, key=lambda run: run.data.metrics.get('accuracy', 0))
    
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_version = mlflow.register_model(model_uri, "ChurnModel")

    client.transition_model_version_stage(
        name="ChurnModel",
        version=model_version.version,
        stage="Production"
    )
    
if __name__ == "__main__":
    train_and_log_models()