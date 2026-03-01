import argparse
import os
import re
import pandas as pd


WHITESPACE = re.compile(r"\s+")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--text_col", type=str, default="reviewText")
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    if args.text_col not in df.columns:
        raise ValueError(f"Missing column '{args.text_col}'. Columns: {list(df.columns)}")

    text = df[args.text_col].fillna("").astype(str)

    # characters (simple)
    df["review_length_chars"] = text.str.len()

    # words (split on whitespace after stripping)
    cleaned = text.str.strip().apply(lambda s: WHITESPACE.sub(" ", s))
    df["review_length_words"] = cleaned.apply(lambda s: 0 if s == "" else len(s.split(" ")))

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "data.parquet")
    df.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(df))


if __name__ == "__main__":
    main()