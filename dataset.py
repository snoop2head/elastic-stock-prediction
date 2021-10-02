# import python modules
import os

# import data wrangling modules
import pandas as pd
import numpy as np

# import visualization modules
from tqdm import tqdm

# import dataloader & stock data downloader
import FinanceDataReader as fdr

# import talib to make stock data derivatives module
from talib import RSI, BBANDS, MACD, ATR


def compute_atr(stock_data):
    """get ATR using talib"""
    df = ATR(stock_data.High, stock_data.Low, stock_data.Close, timeperiod=14)
    return df.sub(df.mean()).div(df.std())


def compute_bb(close):
    """get high and low from Bollinger Bands using talib"""
    high, mid, low = BBANDS(close, timeperiod=20)
    return pd.DataFrame({"bb_high": high, "bb_low": low}, index=close.index)


def compute_macd(close):
    """get macd using talib"""
    macd = MACD(close)[0]
    return (macd - np.mean(macd)) / np.std(macd)


def make_true_df(
    root_path="./data",
    start_date=None,
    cut_off_date=None,
    new_eval_period=False,
):
    """fill out ground truth stock price of the companies for the public submission period"""
    if not new_eval_period:
        sample_name = "sample_submission.csv"  # week 4 submission
        blank_submission = pd.read_csv(os.path.join(root_path, sample_name)).set_index(
            "Day"
        )
        true_df = blank_submission.copy()
        true_df_first_day = true_df.iloc[0].name
        true_df_last_day = true_df.iloc[4].name
    if new_eval_period:
        new_public_name = "new_public.csv"
        df_new_public = pd.read_csv(os.path.join(root_path, new_public_name)).set_index(
            "Day"
        )
        true_df = df_new_public.copy()
        true_df_first_day = true_df.iloc[0].name
        true_df_last_day = true_df.iloc[4].name

    for company_code in tqdm(true_df.columns):
        data_raw = fdr.DataReader(
            company_code, start=start_date, end=cut_off_date
        ).reset_index()
        data_raw = data_raw.drop(columns=["Change"])
        data_raw = data_raw.replace(0, np.nan).ffill()
        data_raw.index = data_raw.Date
        # print(data_raw.head())
        public_true_closes = data_raw.loc[
            true_df_first_day:true_df_last_day
        ].Close.iloc[:]
        # print(public_true_closes)
        true_df.loc[:, company_code] = public_true_closes.to_list() * 2
    return true_df


def make_blank_submission(root_path="./data", new_eval_period=False):
    if not new_eval_period:
        sample_name = "sample_submission.csv"  # week 4 submission
        blank_submission = pd.read_csv(os.path.join(root_path, sample_name)).set_index(
            "Day"
        )
    if new_eval_period:
        new_public_name = "new_public.csv"
        blank_submission = pd.read_csv(
            os.path.join(root_path, new_public_name)
        ).set_index("Day")
    return blank_submission


def make_derivative_data(input_data):
    """
    - make derivative data from the fdr dataset
    - not using external dataset which increases noise that harms the performance
    """
    data = input_data.copy()

    # Korean won volume
    data["KRW_Vol"] = data[["Close", "Volume"]].prod(axis=1)
    data["KRW_Vol_1m"] = data.KRW_Vol.rolling(21).mean()

    # RSI
    data["RSI"] = RSI(data.Close)

    # Bollinger Bands
    data[["bb_high", "bb_low"]] = compute_bb(data.Close)
    data["bb_high"] = data.bb_high.sub(data.Close).div(data.bb_high).apply(np.log1p)
    data["bb_low"] = data.Close.sub(data.bb_low).div(data.Close).apply(np.log1p)

    # ATR
    data["ATR"] = compute_atr(data)

    # MACD
    data["MACD"] = compute_macd(data.Close)

    # Lagged Returns
    lags = [1, 2, 3, 4, 5, 10, 21, 42, 63]
    for lag in lags:
        data[f"return_{lag}d"] = data.Close.pct_change(lag).add(1).pow(1 / lag).sub(1)
    for t in [1, 2, 3, 4, 5]:
        for lag in [1, 5, 10, 21]:
            data[f"return_{lag}d_lag{t}"] = data[f"return_{lag}d"].shift(t * lag)

    # target return
    for t in [1, 2, 3, 4, 5]:
        data[f"target_{t}d"] = data[f"return_{t}d"].shift(-t)

    # volume change
    q = 0.01
    data[data.filter(like="Vol").columns] = (
        data.filter(like="Vol")
        .pct_change()
        .apply(lambda x: x.clip(lower=x.quantile(q), upper=x.quantile(1 - q)))
    )

    # drop original data columns
    data = data.drop(["Date", "Open", "High", "Low", "Close"], axis=1)
    data = data.fillna(method="ffill")
    data = data.fillna(method="bfill")

    return data


def make_external_Data(data, start_date, end_date):
    """
    add external data columns such as WTI, philadelphia price index, NASDAQ, S&P500, etc.
    """

    # fetch economical indices using batch request to fdr
    fred_data_raw = fdr.DataReader(
        ["NASDAQCOM", "ICSA", "UNRATE", "UMCSENT", "HSN1F", "M2", "BAMLH0A0HYM2"],
        start=start_date,
        end=end_date,
        data_source="fred",
    ).reset_index()
    fred_data_raw.rename(
        columns={"DATE": "Date"}, inplace=True
    )  # rename column name DATE to Date
    fred_data_raw = fred_data_raw.replace(0, np.nan).ffill()
    fred_data_raw.index = fred_data_raw.Date

    # Nasdaq index
    data["NASDAQCOM"] = fred_data_raw.NASDAQCOM.pct_change()

    # Initial Claims: https://fred.stlouisfed.org/series/ICSA
    data["ICSA"] = fred_data_raw.ICSA

    # Unrate: https://fred.stlouisfed.org/series/UNRATE
    data["UNRATE"] = fred_data_raw.UNRATE

    # Consumer Sentiment: https://fred.stlouisfed.org/series/UMCSENT
    data["UMCSENT"] = fred_data_raw.UMCSENT

    # New One Family Houses Sold: https://fred.stlouisfed.org/series/HSN1F
    data["HSN1F"] = fred_data_raw.HSN1F

    # M2 money supply: https://fred.stlouisfed.org/series/M2
    data["M2"] = fred_data_raw.M2

    # High Yield Index Option Adjusted Spread: # https://fred.stlouisfed.org/series/BAMLH0A0HYM2
    data["BAMLH0A0HYM2"] = fred_data_raw.BAMLH0A0HYM2

    print(data.info())
    print(data.head())
    # print(data.tail())
    # print(data.summary())
    return data
