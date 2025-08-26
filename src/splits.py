import pandas as pd

DATE_COL = "date"

def split_by_time(df, train_end="2022-12-31", valid_end="2023-12-31"):
    train_end = pd.to_datetime(train_end)
    valid_end = pd.to_datetime(valid_end)
    trn = df[df[DATE_COL] <= train_end]
    val = df[(df[DATE_COL] > train_end) & (df[DATE_COL] <= valid_end)]
    tst = df[df[DATE_COL] > valid_end]
    return trn, val, tst