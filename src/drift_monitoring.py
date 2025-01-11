"""
This module detects model drift using Population Stability Index (PSI).
"""

import numpy as np
import pandas as pd
import helpers


def calculate_psi(training_series, new_series, bins=10):
    """
    Calculate the Population Stability Index (PSI) between two data series.

    Args:
        training_series (pd.Series): The original training data.
        new_series (pd.Series): The new data to compare.
        bins (int): Number of bins to divide the data into.

    Returns:
        float: The total PSI value.
    """
    bin_edges = np.linspace(min(training_series), max(training_series), bins + 1)
    training_bin_counts = np.histogram(training_series, bins=bin_edges)[0]
    new_bin_counts = np.histogram(new_series, bins=bin_edges)[0]
    training_bin_counts = np.where(
        training_bin_counts == 0, 0.0001, training_bin_counts
    )
    new_bin_counts = np.where(new_bin_counts == 0, 0.0001, new_bin_counts)
    training_bin_densities = training_bin_counts / sum(training_bin_counts)
    new_bin_densities = new_bin_counts / sum(new_bin_counts)
    psi_values = (new_bin_densities - training_bin_densities) * np.log(
        new_bin_densities / training_bin_densities
    )
    return sum(psi_values)


def merge_datasets(timestamp, processed_path):
    """
    Merge training and test datasets for a given timestamp.

    Args:
        timestamp (str): The timestamp of the dataset.
        processed_path (str): The path to the processed data folder.

    Returns:
        pd.DataFrame: The merged dataset.
    """
    x_train = pd.read_csv(f"{processed_path}/{timestamp}_x_train.csv")
    x_test = pd.read_csv(f"{processed_path}/{timestamp}_x_test.csv")
    return pd.concat([x_train, x_test], ignore_index=True)


def detect_model_drift():
    """
    Detect model drift by comparing training and new data using PSI.

    Prints 'True' if significant drift is detected, otherwise 'False'.
    """
    timestamp_model = helpers.get_last_timestamp(helpers.MODEL_LOG_FILE)

    if not timestamp_model:
        print("True")
        return

    timestamp = helpers.get_last_timestamp(helpers.TIMESTAMP_FILE)
    df_train = merge_datasets(timestamp_model, "../../data/processed")
    df_new = merge_datasets(timestamp, "../../data/processed")

    features_to_check = list(df_train.columns)
    psi_threshold = 0.25
    for feature in features_to_check:
        psi_value = calculate_psi(df_train[feature], df_new[feature])
        print(psi_value)
        if psi_value > psi_threshold:
            print(
                f"Significant drift detected in feature {feature} with PSI: {psi_value}"
            )
            print("True")
            return

    print("False")


if __name__ == "__main__":
    detect_model_drift()
