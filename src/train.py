import argparse
import os
import time

import azureml.mlflow
import mlflow
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# --------------------------------------------------
# Arguments
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Path does not exist: {path} (did you even pass the right input?)"
        )
    return pd.read_parquet(path)


# --------------------------------------------------
# Labels
# --------------------------------------------------
def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing. You had one job.")

    df = df.copy()
    df["label"] = (df["overall"] >= 4).astype(int)
    return df


# --------------------------------------------------
# Features
# --------------------------------------------------
def build_features(df):
    excluded_exact = {
        "asin", "reviewerID",
        "overall", "overall_x", "overall_y",
        "label"
    }

    feature_cols = []

    for col in df.columns:
        lower = col.lower()

        # remove exact columns
        if col in excluded_exact:
            continue

        # remove anything leaking target
        if "overall" in lower:
            continue

        # remove text columns
        if any(x in lower for x in ["reviewtext", "summary", "title", "time", "helpful"]):
            continue

        # keep only numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise RuntimeError("No usable features found.")

    print("FINAL FEATURES USED:", feature_cols[:20], "...")

    X = df[feature_cols].copy().fillna(0)
    return X


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
def evaluate(model, X, y, split):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, probs)

    mlflow.log_metric(f"{split}_accuracy", acc)
    mlflow.log_metric(f"{split}_precision", precision)
    mlflow.log_metric(f"{split}_recall", recall)
    mlflow.log_metric(f"{split}_f1", f1)
    mlflow.log_metric(f"{split}_auc", auc)

    print(f"{split} accuracy: {acc:.4f}")
    print(f"{split} precision: {precision:.4f}")
    print(f"{split} recall: {recall:.4f}")
    print(f"{split} f1: {f1:.4f}")
    print(f"{split} auc: {auc:.4f}")


def main():
    args = parse_args()
    start_time = time.time()

    print("Loading data...")
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)
    print("TRAIN COLUMNS:")
    print(train_df.columns.tolist())

    print("Creating labels...")
    train_df = create_labels(train_df)
    val_df = create_labels(val_df)
    test_df = create_labels(test_df)

    print("Building features...")
    X_train = build_features(train_df)
    y_train = train_df["label"]

    X_val = build_features(val_df)
    y_val = val_df["label"]

    X_test = build_features(test_df)
    y_test = test_df["label"]

    if len(X_train) == 0:
        raise RuntimeError("Training data is empty. That’s concerning.")

    print("Logging parameters...")
    mlflow.log_param("model_name", "LogisticRegression")
    mlflow.log_param("label_rule", "overall >= 4")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("num_features", X_train.shape[1])

    print("Training model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")

    print("Saving model...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)
    print(f"training runtime seconds: {runtime:.2f}")

    print("Done.")


if __name__ == "__main__":
    main()