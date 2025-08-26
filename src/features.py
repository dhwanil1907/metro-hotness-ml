import pandas as pd
import numpy as np

ID_COL = "cbsa_title"
DATE_COL = "date"

def add_lag_rolling_feats(df, base_cols, lags=(1,3,6,12)):
    df = df.copy()
    for col in base_cols:
        if col not in df.columns:
            continue
        for L in lags:
            df[f"{col}_lag{L}"] = df.groupby(ID_COL)[col].shift(L)
            df[f"{col}_mom{L}"] = df.groupby(ID_COL)[col].pct_change(L)
            roll = df.groupby(ID_COL)[col].rolling(L, min_periods=1)
            df[f"{col}_rollmean{L}"] = roll.mean().reset_index(level=0, drop=True)
            df[f"{col}_rollstd{L}"] = roll.std().reset_index(level=0, drop=True)
    # Seasonality
    if DATE_COL in df.columns:
        df["month"] = df[DATE_COL].dt.month
        df["year"] = df[DATE_COL].dt.year
    return df

def make_target(df, current_col="hotness_score", horizon=1):
    df = df.copy()
    df["target_next"] = df.groupby(ID_COL)[current_col].shift(-horizon)
    return df

def select_features(df, target_col="target_next", exclude=("cbsa_title","date","month_date_yyyymm")):
    ex = set(exclude) | {target_col}
    feat_cols = [c for c in df.columns if c not in ex and df[c].dtype != "O"]
    return feat_cols