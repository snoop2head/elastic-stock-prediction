# data wrangling tools
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

# import dataloader & stock data downloader
import FinanceDataReader as fdr

# import machine learning modules
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# import machine learning models
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# custom modules
from dataset import make_derivative_data

warnings.filterwarnings("ignore")  # remove warnings from pandas

# set options pandas: display more columns and reduced digits of float numbers
pd.set_option("display.max_columns", 100)
pd.options.display.float_format = "{:.4f}".format


def nmae(true_df, pred_df_input):
    """grading criteria for public leader board"""
    # extract columns from true_df same as pred_df_input
    true_df_copy = true_df.copy()
    true_df_copy = true_df_copy.loc[:, pred_df_input.columns]
    true_df_copy = true_df_copy.iloc[:5]
    pred_df_input = pred_df_input.iloc[:5]
    return (
        (abs(true_df_copy - pred_df_input) / true_df_copy * 100).iloc[:5].values.mean()
    )


def elasticnet_cv_predict(
    pred_df_input,
    model=ElasticNetCV(max_iter=1000000),
    bool_public=True,
    start_date=None,
    recent_known_date="2021-09-24",
    scaler=StandardScaler(),
):
    """predict based on given input"""

    if bool_public:
        # get first five rows of pred_df_input
        pred_df_input = pred_df_input.iloc[:5]
    else:
        # dropping first 5 days of public dates from the submission dataframe
        pred_df_input = pred_df_input.iloc[5:]

    for company_code in tqdm(pred_df_input.columns):
        # clear_output()
        data_raw = fdr.DataReader(
            company_code, start=start_date, end=recent_known_date
        ).reset_index()
        data_raw = data_raw.drop(columns="Change")
        data_raw = data_raw.replace(0, np.nan).ffill()
        data_raw.index = data_raw.Date

        if bool_public:  # if public submission
            # make necessary data for the prediction
            data = make_derivative_data(data_raw)

            # get the last date of the public submission period
            public_last_close = data_raw.loc[recent_known_date].Close
            public = data.loc[[recent_known_date]]

            # loc[] is inclusive for the end slicing date, unlike list slicing
            train_indv_comp = data.loc[:recent_known_date]
            Ys = train_indv_comp.filter(like="target")
            X = train_indv_comp.drop(Ys.columns, axis=1)
            Ys_public = public.filter(like="target")
            X_public = public.drop(Ys_public.columns, axis=1)
            X = scaler.fit_transform(X)
            X_public = scaler.transform(X_public)
            pred_public = []
            public_close = public_last_close
            for y_col in Ys.columns:
                model.fit(X, Ys[y_col])
                r_pred_public = model.predict(X_public)

                public_close = public_close * (1 + r_pred_public[0])

                pred_public.append(public_close)
            # print(pred_df_input.shape)
            # display(pred_df_input)
            pred_df_input.loc[:, company_code] = pred_public

        else:  # if private submission
            data = make_derivative_data(data_raw)
            # display(pred_df_input)
            private_last_close = data_raw.loc[recent_known_date].Close

            # display(data)
            private = data.loc[[recent_known_date]]
            # display(private)
            train_indv_comp = data.loc[:recent_known_date]
            # display(train_indv_comp)

            # make train_indv_comp data
            Ys = train_indv_comp.filter(
                like="target"
            )  # Consisted of mon, tue, wed, thur, fri data columns
            X = train_indv_comp.drop(Ys.columns, axis=1)

            # make private data
            Ys_private = private.filter(like="target")
            X_private = private.drop(Ys_private.columns, axis=1)

            # fit scaler
            X = scaler.fit_transform(X)
            X_private = scaler.transform(X_private)

            pred_private = []
            private_close = private_last_close

            for y_col in Ys.columns:
                model.fit(X, Ys[y_col])
                r_pred_private = model.predict(X_private)
                private_close = private_close * (1 + r_pred_private[0])
                pred_private.append(private_close)
            # print(pred_private)
            pred_df_input.loc[:, company_code] = pred_private
            # display(pred_df_input)
    return pred_df_input
