import json
import os
import joblib
import pandas as pd

model = None


def build_features(df):
    excluded_exact = {
        "asin", "reviewerID",
        "overall", "overall_x", "overall_y",
        "label"
    }

    feature_cols = []

    for col in df.columns:
        lower = col.lower()

        if col in excluded_exact:
            continue
        if "overall" in lower:
            continue
        if any(x in lower for x in ["reviewtext", "summary", "title", "time", "helpful"]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise RuntimeError("No usable features found.")

    return df[feature_cols].copy().fillna(0)


def init():
    global model
    model_dir = os.environ.get("AZUREML_MODEL_DIR")

    model_path = os.path.join(model_dir, "model_output", "model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pkl not found at: {model_path}")

    model = joblib.load(model_path)


def run(raw_data):
    try:
        data = json.loads(raw_data)
        rows = data["data"] if isinstance(data, dict) and "data" in data else data
        df = pd.DataFrame(rows)
        X = build_features(df)
        preds = model.predict(X)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}