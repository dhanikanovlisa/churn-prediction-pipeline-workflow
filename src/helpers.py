"""
Helper functions for managing timestamps and model logs.
"""

import os
import json

MODEL_LOG_FILE = "../../data/model_log.json"
TIMESTAMP_FILE = "../../data/timestamps.json"


def save_timestamp(timestamp, file_path):
    """
    Saves a timestamp to a JSON file. If the file already exists, the timestamp is appended.

    Args:
        timestamp (str): The timestamp to save.
        file_path (str): The path to the JSON file where the timestamp will be stored.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                timestamps = json.load(file)
            except json.JSONDecodeError:
                timestamps = []
    else:
        timestamps = []

    timestamps.append({"timestamp": timestamp})

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(timestamps, file, indent=4)


def get_last_timestamp(file_path):
    """
    Retrieves the most recent timestamp from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing timestamps.

    Returns:
        str or None: The most recent timestamp, or None if the file is empty or invalid.
    """
    try:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, "r", encoding="utf-8") as file:
                timestamps = json.load(file)
                if timestamps:
                    last_timestamp = timestamps[-1]["timestamp"]
                    return last_timestamp
        return None
    except (OSError, IOError) as e:
        print(f"File operation error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None
