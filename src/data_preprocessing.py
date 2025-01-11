"""
This module contains the data preprocessing pipeline for the churn prediction project.
"""
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import mean, stddev
from sklearn.model_selection import train_test_split
import helpers

DATA_RAW_PATH = "../../data/raw/drift.csv"
DATA_PROCESSED_PATH = "../../data/processed"
TIMESTAMP_FILE = "../../data/config.json"

# Initialize Spark Sessions
spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()


def load_and_clean_data(file_path):
    """
    Loads the data from a CSV file and cleans it by:
    - Dropping irrelevant columns.
    - Converting types.
    - Removing duplicates and null values.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pyspark.sql.DataFrame: The cleaned dataframe.
    """
    df = spark.read.option("header", "true").csv(file_path, inferSchema=True)
    df = df.drop("customerID")  # Drop irrelevant columns
    df = df.withColumn("TotalCharges", col("TotalCharges").cast("double"))
    df = df.dropDuplicates().na.drop()
    return df


def preprocess_numerical_feat(df):
    """
    Preprocesses numerical features by standardizing them.

    Args:
        df (pyspark.sql.DataFrame): The input dataframe.

    Returns:
        pyspark.sql.DataFrame: The dataframe with standardized numerical features.
    """
    numerical_cols = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, (IntegerType, DoubleType))
    ]
    for col_name in numerical_cols:
        col_mean, col_stddev = df.select(
            mean(col_name).alias("mean"), stddev(col_name).alias("stddev")
        ).first()
        if col_stddev != 0:
            df = df.withColumn(
                col_name + "_scaled", (col(col_name) - col_mean) / col_stddev
            )
        else:
            df = df.withColumn(col_name + "_scaled", col(col_name) - col_mean)
    return df


def preprocess_categorical_feat(df):
    """
    Preprocesses categorical features by encoding them with StringIndexer
    and OneHotEncoder.

    Args:
        df (pyspark.sql.DataFrame): The input dataframe.

    Returns:
        pyspark.sql.DataFrame: The dataframe with encoded categorical features.
    """
    categorical_cols = [
        field.name
        for field in df.schema.fields
        if not field.name.endswith("_scaled")
        and not isinstance(field.dataType, (IntegerType, DoubleType))
    ]
    for col_name in categorical_cols:
        # StringIndexer
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed")
        df = indexer.fit(df).transform(df)
        # OneHotEncoder
        encoder = OneHotEncoder(
            inputCol=f"{col_name}_indexed", outputCol=f"{col_name}_encoded"
        )
        df = encoder.fit(df).transform(df)
    return df


def balance_dataset(df):
    """
    Balances the dataset by oversampling the minority class.

    Args:
        df (pyspark.sql.DataFrame): The input dataframe.

    Returns:
        pyspark.sql.DataFrame: The balanced dataframe.
    """
    major_df = df.filter(col("Churn_indexed") == 0)
    minor_df = df.filter(col("Churn_indexed") == 1)
    major_count = major_df.count()
    minor_count = minor_df.count()
    ratio = major_count / minor_count
    oversampled_minor_df = minor_df.sample(withReplacement=True, fraction=ratio)
    balanced_df = major_df.unionAll(oversampled_minor_df)
    return balanced_df


def data_processing():
    """
    Main function to preprocess and save the data for model training and evaluation.
    """
    df = load_and_clean_data(DATA_RAW_PATH)
    df = preprocess_numerical_feat(df)
    df = preprocess_categorical_feat(df)
    df_selected = df.select(
        *[
            col_name
            for col_name in df.columns
            if "_scaled" in col_name or "_indexed" in col_name
        ]
    )
    balanced_df = balance_dataset(df_selected)
    pandas_df = balanced_df.toPandas()
    x = pandas_df.drop(columns=["Churn_indexed"])
    y = pandas_df["Churn_indexed"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    helpers.save_timestamp(timestamp, helpers.TIMESTAMP_FILE)
    x_train.to_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_x_train.csv", index=False)
    y_train.to_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_y_train.csv", index=False)
    x_test.to_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_x_test.csv", index=False)
    y_test.to_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_y_test.csv", index=False)


if __name__ == "__main__":
    data_processing()
