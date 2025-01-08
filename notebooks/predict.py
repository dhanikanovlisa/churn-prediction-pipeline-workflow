import pandas as pd
import requests
import json

# Load your CSV data into a DataFrame
data = pd.read_csv('data/processed/data_2 copy.csv')

# Convert the DataFrame to a JSON string in the 'split' orientation
json_data = data.to_json(orient='split')

# Wrap the JSON string in a dictionary under the 'dataframe_split' key
request_data = json.dumps({"dataframe_split": json.loads(json_data)})

# Prepare the headers
headers = {'Content-Type': 'application/json'}

# URL where the MLflow model is being served
url = 'http://127.0.0.1:1234/invocations'

# Send the request
response = requests.post(url, headers=headers, data=request_data)
print(response.text)