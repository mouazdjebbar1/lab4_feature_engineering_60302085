import argparse
import os
import joblib
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--text_col", type=str, default="reviewText")
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)
    text = df[args.text_col].fillna("").astype(str)

    vec_path = os.path.join(args.model_dir, "tfidf_vectorizer.joblib")
    vec = joblib.load(vec_path)

    X = vec.transform(text)
    feature_names = vec.get_feature_names_out()

    tfidf_df = pd.DataFrame(X.toarray(), columns=[f"tfidf_{c}" for c in feature_names])

    # Keep keys so you can merge later
    keep_cols = []
    for c in ["asin", "reviewerID"]:
        if c in df.columns:
            keep_cols.append(c)
    out_df = pd.concat([df[keep_cols].reset_index(drop=True), tfidf_df], axis=1)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "data.parquet")
    out_df.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(out_df), "Cols:", len(out_df.columns))


if __name__ == "__main__":
    main()