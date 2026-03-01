import argparse
import os
import pandas as pd
import gc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, required=True)
    p.add_argument("--length", type=str, required=True)
    p.add_argument("--sentiment", type=str, required=True)
    p.add_argument("--tfidf", type=str, required=True)
    p.add_argument("--sbert", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--key1", type=str, default="asin")
    p.add_argument("--key2", type=str, default="reviewerID")
    return p.parse_args()


def load_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def reduce_memory(df, keys):
    """
    Convert float64 → float32 to reduce memory.
    """
    for col in df.columns:
        if col not in keys and df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
    return df


def main():
    args = parse_args()
    keys = [args.key1, args.key2]

    print("Loading base...")
    base = load_df(args.base)

    keep_cols = keys.copy()
    for c in ["overall", "review_year", "brand", "price"]:
        if c in base.columns:
            keep_cols.append(c)

    merged = base[keep_cols].copy()
    del base
    gc.collect()

    print("Loading length...")
    length = load_df(args.length)
    length = reduce_memory(length, keys)
    merged = merged.merge(length, on=keys, how="inner")
    del length
    gc.collect()

    print("Loading sentiment...")
    sentiment = load_df(args.sentiment)
    sentiment = reduce_memory(sentiment, keys)
    merged = merged.merge(sentiment, on=keys, how="inner")
    del sentiment
    gc.collect()

    print("Loading tfidf...")
    tfidf = load_df(args.tfidf)
    tfidf = reduce_memory(tfidf, keys)
    merged = merged.merge(tfidf, on=keys, how="inner")
    del tfidf
    gc.collect()

    print("Loading sbert...")
    sbert = load_df(args.sbert)
    sbert = reduce_memory(sbert, keys)
    merged = merged.merge(sbert, on=keys, how="inner")
    del sbert
    gc.collect()

    print("Final shape:", merged.shape)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "data.parquet")

    merged.to_parquet(out_path, index=False)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()