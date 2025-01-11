from datetime import datetime, timedelta

import joblib
# from airflow import DAG
# from airflow.operators.python import PythonOperator
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from pyspark.ml.feature import StandardScaler, StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev
from pyspark.sql.types import DoubleType, IntegerType
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import requests

DATA_RAW_PATH = "data/raw/data.csv"
DATA_PROCESSED_PATH = "data/processed"
MODEL_DEPLOYMENT_PORT = "1234"

# variable yang butuh dipassing di airflow
timestamp = "20250108165246"

# Load and preprocess data
def preprocess_data(**kwargs):
    spark = SparkSession.builder.appName('ChurnPrediction').getOrCreate()
    df = spark.read.option('header', 'true').csv(DATA_RAW_PATH, inferSchema=True)

    df = df.drop('customerID', 'gender') 
    df = df.withColumn("TotalCharges", col("TotalCharges").cast("double"))
    df = df.dropDuplicates().na.drop()

    numerical_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, (IntegerType, DoubleType))]
    categorical_cols = [field.name for field in df.schema.fields if field.name not in numerical_cols + ['customerID', 'gender']]
    for col_name in numerical_cols:
        col_mean, col_stddev = df.select(mean(col_name).alias('mean'), stddev(col_name).alias('stddev')).first()
        if col_stddev != 0:
            df = df.withColumn(col_name + "_scaled", (col(col_name) - col_mean) / col_stddev)
        else:
            df = df.withColumn(col_name + "_scaled", col(col_name) - col_mean)
    for col_name in categorical_cols:
        indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed")
        df = indexer.fit(df).transform(df)

    selected_cols = [col_name for col_name in df.columns if "_scaled" in col_name or "_indexed" in col_name]
    df = df.select(*selected_cols)
    major_df = df.filter(col('Churn_indexed') == 0)
    minor_df = df.filter(col('Churn_indexed') == 1)
    ratio = major_df.count() / minor_df.count()
    oversampled_minor_df = minor_df.sample(withReplacement=True, fraction=ratio)
    input_data = major_df.unionAll(oversampled_minor_df)

    pandas_df = input_data.toPandas()

    X = pandas_df.drop(columns=['Churn_indexed'])
    y = pandas_df['Churn_indexed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    X_train.to_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_x_train.csv", index=False)
    y_train.to_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_y_train.csv", index=False)
    X_test.to_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_x_test.csv", index=False)
    y_test.to_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_y_test.csv", index=False)

    spark.stop()

# Train models and log them to MLflow
def train_and_log_models():

    X_train = pd.read_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_x_train.csv")
    y_train = pd.read_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_y_train.csv")
    X_test = pd.read_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_x_test.csv")
    y_test = pd.read_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_y_test.csv")

    # Model 1: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_preds)
    
    signature = infer_signature(X_train, rf_preds)
    with mlflow.start_run(run_name="Random_Forest"):
        mlflow.log_param("timestamp", timestamp)
        mlflow.sklearn.log_model(rf_model, "Random_Forest_Model", registered_model_name="Random_Forest", signature=signature)
        mlflow.log_metric("accuracy", rf_accuracy)
    
    # Model 2: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_preds)
    
    signature = infer_signature(X_train, lr_preds)
    with mlflow.start_run(run_name="Logistic_Regression"):
        mlflow.log_param("timestamp", timestamp)
        mlflow.sklearn.log_model(lr_model, "Logistic_Regression_Model", registered_model_name="Logistic_Regression", signature=signature)
        mlflow.log_metric("accuracy", lr_accuracy)

# Deploy the best model
def deploy_model(**kwargs):
    client = mlflow.tracking.MlflowClient()
    recent_runs = client.search_runs(
        experiment_ids=["0"],
        order_by=["start_time DESC"],
        max_results=2
    )
    best_run = max(recent_runs, key=lambda run: run.data.metrics.get('accuracy', 0))
    best_model_uri = best_run.info.run_name + "/latest"
    print(best_model_uri)
    # mlflow.models.serve(model_uri=best_model_uri, port=1234) harus di terminal
    # mlflow models serve -m "models:/Random_Forest/latest" -p 1234
    #mlflow models serve -m "models:{{best_model_uri}}" -p {{MODEL_DEPLOYMENT_PORT}}

def is_model_deployed():
    dummy_payload = {
        "dataframe_record": [[0] * 18]
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post("http://127.0.0.1:"+ MODEL_DEPLOYMENT_PORT + "/invocations", json=dummy_payload, headers=headers)
        if response.status_code == 400:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        return False


def conditional_retraining_and_deployment():
    if not is_model_deployed() or detect_data_drift():
        train_and_log_models()
        deploy_model()
    print("conditional retraining and deployment")


def calculate_psi(training_series, new_series, bins=10):
    bin_edges = np.linspace(min(training_series), max(training_series), bins + 1)
    training_bin_counts = np.histogram(training_series, bins=bin_edges)[0]
    new_bin_counts = np.histogram(new_series, bins=bin_edges)[0]
    training_bin_counts = np.where(training_bin_counts == 0, 0.0001, training_bin_counts)
    new_bin_counts = np.where(new_bin_counts == 0, 0.0001, new_bin_counts)
    training_bin_densities = training_bin_counts / sum(training_bin_counts)
    new_bin_densities = new_bin_counts / sum(new_bin_counts)
    psi_values = (new_bin_densities - training_bin_densities) * np.log(new_bin_densities / training_bin_densities)
    psi_total = sum(psi_values)
    
    return psi_total

def detect_data_drift():
    try:
        df_train = pd.read_csv("data/processed/20250108165246_x_train.csv")
        df_new = pd.read_csv(f"data/processed/{timestamp}_y_train.csv")

        features_to_check = [col for col in df_train.columns]
        significant_drift = False
        psi_threshold = 0.25

        for feature in features_to_check:
            psi_value = calculate_psi(df_train[feature], df_new[feature])
            print(psi_value)
            if psi_value > psi_threshold:
                print(f"Significant drift detected in feature {feature} with PSI: {psi_value}")
                significant_drift = True
                break
        
        if significant_drift:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error in detecting data drift: {str(e)}")
        return False

preprocess_data()
# conditional_retraining_and_deployment()
# detect_data_drift()
# deploy_model()