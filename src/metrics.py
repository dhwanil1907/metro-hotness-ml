import numpy as np

def mae(y, p):
    return float(np.nanmean(np.abs(y - p)))

def rmse(y, p):
    return float(np.sqrt(np.nanmean((y - p)**2)))