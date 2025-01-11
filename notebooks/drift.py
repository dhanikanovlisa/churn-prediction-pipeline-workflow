from datetime import datetime

import numpy as np
import pandas as pd
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev
from pyspark.sql.types import DoubleType, IntegerType
from sklearn.model_selection import train_test_split

DATA_RAW_PATH = "data/raw/data.csv"
DATA_DRIFT_PATH = "data/raw/drift.csv"
DATA_PROCESSED_PATH = "data/processed"
MODEL_DEPLOYMENT_PORT = "1234"

# Change the relationship between two numerical features
def introduce_covariate_drift(df, feature1, feature2, drift_factor=1.5):
    df[feature2] = df[feature1] * drift_factor + np.random.normal(0, 1, len(df))
    return df

def preprocess_data(**kwargs):
    spark = SparkSession.builder.appName('ChurnPrediction').getOrCreate()
    df = spark.read.option('header', 'true').csv(DATA_DRIFT_PATH, inferSchema=True)

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
    return timestamp

def calculate_psi(training_series, new_series, bins=10):
    # Calculate bin edges based on combined range
    combined_min = min(min(training_series), min(new_series))
    combined_max = max(max(training_series), max(new_series))
    bin_edges = np.linspace(combined_min, combined_max, bins + 1)

    # Calculate bin counts
    training_bin_counts = np.histogram(training_series, bins=bin_edges)[0]
    new_bin_counts = np.histogram(new_series, bins=bin_edges)[0]

    # Avoid zero bin counts
    training_bin_counts = np.where(training_bin_counts == 0, 0.0001, training_bin_counts)
    new_bin_counts = np.where(new_bin_counts == 0, 0.0001, new_bin_counts)

    # Calculate densities
    training_bin_densities = training_bin_counts / sum(training_bin_counts)
    new_bin_densities = new_bin_counts / sum(new_bin_counts)

    # Calculate PSI values
    psi_values = (new_bin_densities - training_bin_densities) * \
                 np.log(new_bin_densities / training_bin_densities)

    # Handle cases where densities are equal (to avoid NaN in log computation)
    psi_values = np.nan_to_num(psi_values, nan=0.0)

    # Calculate total PSI
    psi_total = sum(psi_values)

    return psi_total

def detect_data_drift(timestamp):
    try:
        print("timestamp: ", timestamp)
        df_train = pd.read_csv("data/processed/20250108165246_x_train.csv")
        df_new = pd.read_csv(f"data/processed/{timestamp}_x_train.csv")

        features_to_check = [col for col in df_train.columns]
        significant_drift = False
        psi_threshold = 0.25

        for feature in features_to_check:
            print("feature", feature)
            psi_value = calculate_psi(df_train[feature], df_new[feature])
            print(psi_value)
            if psi_value > psi_threshold:
                print(f"Significant drift detected in feature {feature} with PSI: {psi_value}")
                significant_drift = True
        
        if significant_drift:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error in detecting data drift: {str(e)}")
        return False


# Apply drift to a specific column
df = pd.read_csv(DATA_RAW_PATH)
df = introduce_covariate_drift(df, feature1='tenure', feature2='MonthlyCharges')
df.to_csv(DATA_DRIFT_PATH, index=False)
timestamp = preprocess_data()
detect_data_drift(timestamp)