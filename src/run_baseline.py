import os
import pandas as pd
from .data_prep import read_and_standardize
from .features import add_lag_rolling_feats, make_target, select_features
from .splits import split_by_time
from .baselines import naive_last_value

# Paths (update if needed)
CORE = os.path.join(os.path.dirname(__file__), "..", "data", "metro_core.csv")
HOT  = os.path.join(os.path.dirname(__file__), "..", "data", "metro_hotness.csv")

def main():
    df = read_and_standardize(CORE, HOT)

    # Basic feature set: choose columns that are likely present
    base_cols = [c for c in [
        "hotness_score", "hotness_rank",
        "median_listing_price", "median_days_on_market",
        "active_listing_count", "new_listing_count"
    ] if c in df.columns]

    df = add_lag_rolling_feats(df, base_cols, lags=(1,3,6,12))
    df = make_target(df, current_col="hotness_score", horizon=1)
    df = df.dropna(subset=["target_next"])  # drop last horizon rows per metro

    # Time split
    train, valid, test = split_by_time(df, train_end="2022-12-31", valid_end="2023-12-31")

    # Baseline
    metrics = naive_last_value(valid, test, target_col="target_next", current_col="hotness_score")
    print("Naive baseline:", metrics)

if __name__ == "__main__":
    main()