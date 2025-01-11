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
import helpers

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


def detect_model_drift(**kwargs):
    timestamp = helpers.get_last_timestamp(helpers.TIMESTAMP_FILE)
    timestamp_model = helpers.get_last_timestamp(helpers.MODEL_LOG_FILE)
    try:
        df_train = pd.read_csv(f"data/processed/{timestamp_model}_x_train.csv")
        df_new = pd.read_csv(f"data/processed/{timestamp}_x_train.csv")

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
    
if __name__ == "__main__":
    detect_model_drift()
    