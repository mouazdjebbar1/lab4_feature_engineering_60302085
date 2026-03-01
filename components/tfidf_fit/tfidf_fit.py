import argparse
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--text_col", type=str, default="reviewText")
    p.add_argument("--max_features", type=int, default=5000)
    p.add_argument("--ngram_max", type=int, default=1)  # 1 = unigrams, 2 = uni+bi
    p.add_argument("--model_out", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.data)

    text = df[args.text_col].fillna("").astype(str)

    vec = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max)
    )
    vec.fit(text)

    os.makedirs(args.model_out, exist_ok=True)
    out_path = os.path.join(args.model_out, "tfidf_vectorizer.joblib")
    joblib.dump(vec, out_path)

    print("Saved vectorizer:", out_path)
    print("Vocab size:", len(vec.vocabulary_))


if __name__ == "__main__":
    main()