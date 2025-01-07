from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler, StringIndexer
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import mean, stddev

#Initialize Spark Sessions
spark = SparkSession.builder.appName('ChurnPrediction').getOrCreate()

#Function to load and clean data
def load_and_clean_data(file_path):
    df = spark.read.option('header', 'true').csv(file_path, inferSchema=True)
    df = df.drop('customerID', 'gender')  # Drop irrelevant columns
    df = df.withColumn("TotalCharges", col("TotalCharges").cast("double"))
    df = df.dropDuplicates().na.drop()  # Remove duplicates and handle missing values
    return df

#Function to preprocess numerical features
def preprocess_numerical_features(df):
    numerical_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, (IntegerType, DoubleType))]
    for col_name in numerical_cols:
        col_mean, col_stddev = df.select(mean(col_name).alias('mean'), stddev(col_name).alias('stddev')).first()
        if col_stddev != 0:
            df = df.withColumn(col_name + "_scaled", (col(col_name) - col_mean) / col_stddev)
        else:
            df = df.withColumn(col_name + "_scaled", col(col_name) - col_mean)
    return df

#Function to preprocess categorical features
def preprocess_categorical_features(df):
    categorical_cols = [field.name for field in df.schema.fields if not field.name.endswith("_scaled") and not isinstance(field.dataType, (IntegerType, DoubleType))]
    for col_name in categorical_cols:
        indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed")
        df = indexer.fit(df).transform(df)
    return df

#Function to balance dataset
def balance_dataset(df):
    major_df = df.filter(col('Churn_indexed') == 0)
    minor_df = df.filter(col('Churn_indexed') == 1)
    major_count = major_df.count()
    minor_count = minor_df.count()
    ratio = major_count / minor_count
    oversampled_minor_df = minor_df.sample(withReplacement=True, fraction=ratio)
    balanced_df = major_df.unionAll(oversampled_minor_df)
    return balanced_df

# Main processing function
def data_processing(input_path, output_path):
    df = load_and_clean_data(input_path)
    df = preprocess_numerical_features(df)
    df = preprocess_categorical_features(df)
    df_selected = df.select(*[col_name for col_name in df.columns if "_scaled" in col_name or "_indexed" in col_name])
    balanced_df = balance_dataset(df_selected)
    balanced_df.show(5)
    balanced_df.toPandas().to_csv(output_path, index=False)
    
if __name__ == '__main__':
    input_file_path = '../data/raw/data.csv'
    output_file_path = '../data/processed/data.csv'
    data_processing(input_file_path, output_file_path)