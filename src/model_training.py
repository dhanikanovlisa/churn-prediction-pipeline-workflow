"""
This script trains two models, a Random Forest and a Logistic Regression model,
and logs them to MLflow.
"""
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import helpers

DATA_PROCESSED_PATH = "../../data/processed"


def log_random_forest_model(x_train, y_train, x_test, y_test, timestamp):
    """
    Trains and logs a Random Forest model to MLflow.
    """
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)
    rf_preds = rf_model.predict(x_test)
    rf_accuracy = accuracy_score(y_test, rf_preds)

    signature = infer_signature(x_train, rf_preds)
    with mlflow.start_run(run_name="Random_Forest"):
        mlflow.log_param("timestamp", timestamp)
        mlflow.sklearn.log_model(rf_model, "Random_Forest_Model", signature=signature)
        mlflow.log_metric("accuracy", rf_accuracy)

    return rf_accuracy


def log_logistic_regression_model(x_train, y_train, x_test, y_test, timestamp):
    """
    Trains and logs a Logistic Regression model to MLflow.
    """
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(x_train, y_train)
    lr_preds = lr_model.predict(x_test)
    lr_accuracy = accuracy_score(y_test, lr_preds)

    signature = infer_signature(x_train, lr_preds)
    with mlflow.start_run(run_name="Logistic_Regression"):
        mlflow.log_param("timestamp", timestamp)
        mlflow.sklearn.log_model(
            lr_model, "Logistic_Regression_Model", signature=signature
        )
        mlflow.log_metric("accuracy", lr_accuracy)

    return lr_accuracy


def train_and_log_models():
    """
    Main function to train and log models.
    """
    timestamp = helpers.get_last_timestamp(helpers.TIMESTAMP_FILE)

    # x_train = pd.read_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_x_train.csv")
    # y_train = pd.read_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_y_train.csv")
    # x_test = pd.read_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_x_test.csv")
    # y_test = pd.read_csv(f"{DATA_PROCESSED_PATH}/{timestamp}_y_test.csv")

    # rf_accuracy = log_random_forest_model(x_train, y_train, x_test, y_test, timestamp)
    # lr_accuracy = log_logistic_regression_model(x_train, y_train, x_test, y_test, timestamp)

    client = MlflowClient()
    best_run = max(
        client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"], max_results=2),
        key=lambda run: run.data.metrics.get("accuracy", 0),
    )

    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_version = mlflow.register_model(model_uri, "ChurnModel")

    client.transition_model_version_stage(
        name="ChurnModel", version=model_version.version, stage="Production"
    )

    helpers.save_timestamp(timestamp, helpers.MODEL_LOG_FILE)


if __name__ == "__main__":
    train_and_log_models()
