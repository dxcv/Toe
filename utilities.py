import pandas as pd
def EMA(Series, N):
    return pd.Series.ewm(Series, span=N, min_periods=N - 1, adjust=True).mean()


def MACD(Series, short=12, long=26, mid=9):

    DIF = EMA(Series, short) - EMA(Series, long)
    DEA = EMA(DIF, mid)
    MACD = (DIF - DEA) * 2
    return DIF, DEA, MACD
