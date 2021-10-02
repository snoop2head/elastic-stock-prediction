import os

# import customized modules
from dataset import *
from models import *
from visualize import *

# import machine learning models
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV

# Configurations
NUMBER_OF_PREDICTIONS = 20  # int: number of sampled companies, None: all 376 companies
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data")
PREDICT_PATH = os.path.join(ROOT_PATH, "predict")
BOOL_NEW_EVAL_PERIOD = True  # False: public leaderboard period, True: new_public period
BOOL_PUBLIC = True  # True: evaluate and assess, False: make private submission
CUT_OFF_DATE = "20210927"  # date to cut before the private leaderboard period


def main():
    """
    main function for both public leaderboard evaluation and private leaderboard prediction
    """

    print("Get ground truth data for 376 listed companies")
    true_df = make_true_df(
        DATA_PATH,
        start_date=None,
        cut_off_date=CUT_OFF_DATE,
        new_eval_period=BOOL_NEW_EVAL_PERIOD,
    )

    # get blank dataframe for predictions
    blank_submission = make_blank_submission(
        DATA_PATH, new_eval_period=BOOL_NEW_EVAL_PERIOD
    )

    # sample number of companies for prediction
    if NUMBER_OF_PREDICTIONS:
        model = ElasticNet()  # purpose of speeding up the process
        model_name = "ElasticNet"
        blank_submission = blank_submission.sample(
            NUMBER_OF_PREDICTIONS, axis=1, random_state=616
        )
    else:
        model = ElasticNetCV(
            max_iter=1000000
        )  # estimated time for training: 3500+ seconds (1 hour)
        model_name = "ElasticNetCV"

    print("make predictions")
    pred_df = elasticnet_cv_predict(
        pred_df_input=blank_submission,
        model=model,
        bool_public=BOOL_PUBLIC,
        start_date=None,
        recent_known_date=CUT_OFF_DATE,
        scaler=StandardScaler(),
    )

    if BOOL_PUBLIC:  # evaluation only
        print(
            f"{model_name} is evaluated with {NUMBER_OF_PREDICTIONS} sampled companies"
        )
        if not BOOL_NEW_EVAL_PERIOD:
            print(
                f"Public Evaluation period: {blank_submission.iloc[0].name} ~ {blank_submission.iloc[4].name}"
            )
        elif BOOL_NEW_EVAL_PERIOD:
            print(
                f"Arbitrary Evaluation period: {blank_submission.iloc[0].name} ~ {blank_submission.iloc[4].name}"
            )
        print(nmae(true_df, pred_df))
    elif not BOOL_PUBLIC:  # save file without evaluation
        file_name = os.path.join(PREDICT_PATH, f"{model_name}_private.csv")
        pred_df.to_csv(file_name, index=False)
        print("Private prediction results are saved at {file_name}")


if __name__ == "__main__":
    main()
