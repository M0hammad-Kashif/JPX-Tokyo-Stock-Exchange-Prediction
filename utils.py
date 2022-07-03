from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd


def adjust_price(price):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # transform Date column into datetime
    price.loc[:, "Date"] = pd.to_datetime(price.loc[:, "Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)

        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()

        # generate AdjustedClose
        df.loc[:, "AdClose"] = (
                df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))

        # reverse order
        df = df.sort_values("Date")

        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdClose"] == 0, "AdClose"] = np.nan

        # forward fill AdjustedClose
        df.loc[:, "AdClose"] = df.loc[:, "AdClose"].ffill()
        return df

    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)

    price.set_index("Date", inplace=True)
    return price


def calc_change_rate_base(price, column_name, periods):
    for period in periods:
        price[f"pct_{period}"] = price[column_name].pct_change(period)
    return price


def calc_volatility_base(price, column_name, periods):
    for period in periods:
        price[f"vol_{period}"] = np.log(price[column_name]).diff().rolling(window=period, min_periods=1).std()
    return price


def calc_moving_average_rate_base(price, column_name, periods):
    for period in periods:
        price[f"mov_{period}"] = price[column_name].rolling(window=period, min_periods=1).mean() / price[column_name]
    return price


def get_features_for_predict(price, code):
    """
    Args:
        price (pd.DataFrame): pd.DataFrame include stock_price
        code (int): A local code for a listed company
    Returns:
        feature DataFrame (pd.DataFrame)
    """
    close_col = "AdClose"

    feats = price.loc[price["SecuritiesCode"] == code,
                      ["SecuritiesCode", "High", "Low", close_col]].copy()

    periods = [10, 21, 63]
    feats = calc_change_rate_base(feats, close_col, periods)
    feats = calc_volatility_base(feats, close_col, periods)
    feats = calc_moving_average_rate_base(feats, close_col, periods)

    # Additional features
    feats["HLRolling"] = ((feats["High"] - feats["Low"]) / feats["Low"]).rolling(21).std()

    feats = feats.dropna()

    feats.reset_index(inplace=True)
    feats["month"] = feats["Date"].dt.month
    feats["day"] = feats["Date"].dt.day
    feats["dow"] = feats["Date"].dt.dayofweek
    feats.set_index('Date', inplace=True)

    # filling data for nan and inf
    # feats = feats.fillna(0)
    feats = feats.replace([np.inf, -np.inf], 0)

    # drop AdjustedClose column
    feats = feats.drop([close_col], axis=1)

    return feats


def get_label(price, code):
    """ Labelizer
    Args:
        price (pd.DataFrame): dataframe of stock_price.csv
        code (int): Local Code in the universe
    Returns:
        df (pd.DataFrame): label data
    """
    df = price.loc[price["SecuritiesCode"] == code].copy()
    df.loc[:, "label"] = df["Target"]

    return df.loc[:, ["SecuritiesCode", "label"]]


# split data into TRAIN and TEST
TRAIN_END = "2022-02-25"
# We put a week gap between TRAIN_END and TEST_START
# to avoid leakage of test data information from label
TEST_START = "2022-03-01"


def get_features_and_label(price, codes, features):
    """
    Args:
        price (pd.DataFrame): loaded price data
        codes  (array) : target codes
        features (pd.DataFrame): features
    Returns:
        train_X (pd.DataFrame): training data
        train_y (pd.DataFrame): label for train_X
        test_X (pd.DataFrame): test data
        test_y (pd.DataFrame): label for test_X
    """
    # to store splited data
    trains_X, tests_X = [], []
    trains_y, tests_y = [], []

    # generate feature one by one
    for code in tqdm(codes):

        feats = features[features["SecuritiesCode"] == code].dropna()
        labels = get_label(price, code).dropna()

        if feats.shape[0] > 0 and labels.shape[0] > 0:
            # align label and feature indexes
            labels = labels.loc[labels.index.isin(feats.index)]
            feats = feats.loc[feats.index.isin(labels.index)]

            assert (labels.loc[:, "SecuritiesCode"] == feats.loc[:, "SecuritiesCode"]).all()
            labels = labels.loc[:, "label"]

            # split data into TRAIN and TEST
            _train_X = feats[: TRAIN_END]
            _test_X = feats[TEST_START:]

            _train_y = labels[: TRAIN_END]
            _test_y = labels[TEST_START:]

            assert len(_train_X) == len(_train_y)
            assert len(_test_X) == len(_test_y)

            # store features
            trains_X.append(_train_X)
            tests_X.append(_test_X)
            # store labels
            trains_y.append(_train_y)
            tests_y.append(_test_y)

    # combine features for each codes
    train_X = pd.concat(trains_X)
    test_X = pd.concat(tests_X)
    # combine label for each codes
    train_y = pd.concat(trains_y)
    test_y = pd.concat(tests_y)

    return train_X, train_y, test_X, test_y
