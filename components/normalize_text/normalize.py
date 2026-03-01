import argparse
import os
import re
import pandas as pd


URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
NUM_PATTERN = re.compile(r"\b\d+(\.\d+)?\b")
WHITESPACE_PATTERN = re.compile(r"\s+")

# keep letters/spaces and some simple punctuation if you want.
# But the lab says "removes punctuation", so we remove non-alphanumeric (except whitespace).
PUNCT_PATTERN = re.compile(r"[^a-z0-9\s]+")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Input parquet file or folder")
    parser.add_argument("--text_col", type=str, default="reviewText", help="Text column name")
    parser.add_argument("--min_chars", type=int, default=10, help="Min characters after cleaning")
    parser.add_argument("--out", type=str, required=True, help="Output folder")
    return parser.parse_args()


def read_parquet_any(path: str) -> pd.DataFrame:
    # AzureML uri_folder often mounts as a directory containing parquet parts
    if os.path.isdir(path):
        return pd.read_parquet(path)
    return pd.read_parquet(path)


def normalize_text(s: str) -> str:
    if s is None:
        return ""

    # to string, lowercase
    s = str(s).lower()

    # replace urls
    s = URL_PATTERN.sub(" <url> ", s)

    # replace numbers (optional token)
    s = NUM_PATTERN.sub(" <num> ", s)

    # remove punctuation / special chars
    s = PUNCT_PATTERN.sub(" ", s)

    # collapse whitespace + trim
    s = WHITESPACE_PATTERN.sub(" ", s).strip()

    return s


def main():
    args = parse_args()

    df = read_parquet_any(args.data)

    if args.text_col not in df.columns:
        raise ValueError(
            f"Column '{args.text_col}' not found. Available columns: {list(df.columns)}"
        )

    # normalize
    df[args.text_col] = df[args.text_col].apply(normalize_text)

    # filter out empty or too short
    df = df[df[args.text_col].str.len() >= args.min_chars].copy()

    # write output
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "data.parquet")
    df.to_parquet(out_path, index=False)

    print("Normalized rows:", len(df))
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()