import pandas as pd
import numpy as np

ID_COL = "cbsa_title"
DATE_RAW = "month_date_yyyymm"
DATE_COL = "date"

def _to_datetime_yyyymm(series):
    s = series.astype(str).str.slice(0, 6)
    return pd.to_datetime(s, format="%Y%m", errors="coerce")

def assert_unique(df, keys):
    dup = df.duplicated(subset=keys, keep=False)
    if dup.any():
        raise ValueError(f"Expected unique rows by {keys}, but found duplicates: {int(dup.sum())}")

def read_and_standardize(core_path, hot_path):
    core = pd.read_csv(core_path)
    hot = pd.read_csv(hot_path)

    # Ensure required columns
    for df in (core, hot):
        if DATE_RAW not in df.columns:
            raise KeyError(f"Missing '{DATE_RAW}' in input data")
        if ID_COL not in df.columns:
            raise KeyError(f"Missing '{ID_COL}' in input data")
        df[DATE_COL] = _to_datetime_yyyymm(df[DATE_RAW])

    # Row-level uniqueness checks on inputs
    assert_unique(core, [ID_COL, DATE_COL])
    assert_unique(hot, [ID_COL, DATE_COL])

    # Join
    df = pd.merge(hot, core.drop(columns=[DATE_RAW]), on=[ID_COL, DATE_COL], how="left", suffixes=("_hot", "_core"))
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    # Basic imputation strategy (conservative): group-wise ffill on numeric cols; then median fill
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df.groupby(ID_COL)[num_cols].ffill()
    for c in num_cols:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)

    return df

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--core", required=True)
    p.add_argument("--hot", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    df = read_and_standardize(args.core, args.hot)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}, shape={df.shape}")