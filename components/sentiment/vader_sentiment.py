import argparse
import os
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--text_col", type=str, default="reviewText")
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def ensure_vader_downloaded():
    # Download VADER lexicon at runtime if missing
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    if args.text_col not in df.columns:
        raise ValueError(f"Missing column '{args.text_col}'. Columns: {list(df.columns)}")

    ensure_vader_downloaded()

    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()

    text = df[args.text_col].fillna("").astype(str)

    # Compute VADER scores per row
    scores = text.apply(lambda s: sia.polarity_scores(s))

    df["sentiment_pos"] = scores.apply(lambda d: float(d["pos"]))
    df["sentiment_neg"] = scores.apply(lambda d: float(d["neg"]))
    df["sentiment_neu"] = scores.apply(lambda d: float(d["neu"]))
    df["sentiment_compound"] = scores.apply(lambda d: float(d["compound"]))

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "data.parquet")
    df.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(df))


if __name__ == "__main__":
    main()