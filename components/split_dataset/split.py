import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--deploy_ratio", type=float, default=0.10)

    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    total = args.train_ratio + args.val_ratio + args.test_ratio + args.deploy_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, but got {total}")

    df = pd.read_parquet(args.data)

    # First split: deployment set (10%)
    train_val_test_df, deploy_df = train_test_split(
        df,
        test_size=args.deploy_ratio,
        random_state=args.seed,
        shuffle=True
    )

    # Remaining data = 90%
    # We want final ratios:
    # train = 60, val = 15, test = 15
    # inside remaining 90%, that becomes:
    # train = 60/90 = 0.6667
    # val = 15/90 = 0.1667
    # test = 15/90 = 0.1667

    temp_test_ratio = args.test_ratio / (args.train_ratio + args.val_ratio + args.test_ratio)

    train_val_df, test_df = train_test_split(
        train_val_test_df,
        test_size=temp_test_ratio,
        random_state=args.seed,
        shuffle=True
    )

    temp_val_ratio = args.val_ratio / (args.train_ratio + args.val_ratio)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=temp_val_ratio,
        random_state=args.seed,
        shuffle=True
    )

    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)
    os.makedirs(args.deploy_out, exist_ok=True)

    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.val_out, "data.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"), index=False)
    deploy_df.to_parquet(os.path.join(args.deploy_out, "data.parquet"), index=False)

    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))
    print("Deployment rows:", len(deploy_df))


if __name__ == "__main__":
    main()