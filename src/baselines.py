import numpy as np
from .metrics import mae, rmse

def naive_last_value(df_valid, df_test, target_col="target_next", current_col="hotness_score"):
    yv = df_valid[target_col].values
    yt = df_test[target_col].values
    pv = df_valid[current_col].values
    pt = df_test[current_col].values
    return {
        "valid_mae": mae(yv, pv),
        "valid_rmse": rmse(yv, pv),
        "test_mae": mae(yt, pt),
        "test_rmse": rmse(yt, pt),
    }