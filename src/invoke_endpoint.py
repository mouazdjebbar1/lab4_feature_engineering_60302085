import requests
import json
import pandas as pd

# ===== FILL THESE =====
endpoint_url = "https://amazon-review-endpoint.qatarcentral.inference.ml.azure.com/score"
api_key = ""

# Load dataset
df = pd.read_parquet("data.parquet")

# Take small sample
sample = df.head(5).copy()

# Keep only numeric columns (same as training idea)
sample = sample.select_dtypes(include=["number"]).fillna(0)

# Convert to JSON
data = sample.to_dict(orient="records")

payload = {
    "data": data
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

response = requests.post(endpoint_url, headers=headers, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.text)