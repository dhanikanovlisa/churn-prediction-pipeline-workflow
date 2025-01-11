import os
import json

MODEL_LOG_FILE = "../../data/model_log.json"
TIMESTAMP_FILE = "../../data/timestamps.json"

def save_timestamp(timestamp, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            try:
                timestamps = json.load(file)
            except json.JSONDecodeError:
                timestamps = []  
    else:
        timestamps = []

    timestamps.append({'timestamp': timestamp})

    with open(file_path, 'w') as file:
        json.dump(timestamps, file, indent=4)

def get_last_timestamp(file_path):
    try:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, 'r') as file:
                timestamps = json.load(file)
                if timestamps:
                    last_timestamp = timestamps[-1]['timestamp']
                    return last_timestamp
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON from file.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None