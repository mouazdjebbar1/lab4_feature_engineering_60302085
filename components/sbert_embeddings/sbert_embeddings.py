import argparse
import os
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--text_col", type=str, default="reviewText")
    p.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    if args.text_col not in df.columns:
        raise ValueError(f"Missing column '{args.text_col}'. Columns: {list(df.columns)}")

    # keys for later merge
    keys = []
    for c in ["asin", "reviewerID"]:
        if c in df.columns:
            keys.append(c)

    texts = df[args.text_col].fillna("").astype(str).tolist()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model_name)
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    # build embedding columns
    emb_dim = emb.shape[1]
    emb_cols = [f"sbert_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(emb, columns=emb_cols)

    out_df = pd.concat([df[keys].reset_index(drop=True), emb_df], axis=1)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "data.parquet")
    out_df.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(out_df))
    print("Embedding dim:", emb_dim)


if __name__ == "__main__":
    main()