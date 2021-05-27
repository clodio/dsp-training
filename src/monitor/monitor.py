from datetime import datetime
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from utils import load_pandas_df_from_s3, upload_pandas_df_to_s3
from constants import files
from constants import models as m
from constants import columns as c


def monitor_input_drift(test_file_path, input_stats_history_path):
    new_loans = load_pandas_df_from_s3(files.S3_BUCKET, test_file_path)

    logging.info("Append new stats to history and upload to s3.")
    datetime_now = datetime.now()
    mean_income = new_loans[c.Loans.ApplicantIncome].mean()
    logging.info(f"Mean income is {int(mean_income)}")
    mean_income_df = pd.DataFrame.from_records([{"timestamp": datetime_now, "mean_income": mean_income}])
    try:
        input_stats_history = load_pandas_df_from_s3(files.S3_BUCKET, input_stats_history_path)
    except:
        logging.info("Input stats history dataframe doesn’t exist yet, creating it now.")
        input_stats_history = None

    input_stats_history = pd.concat([input_stats_history, mean_income_df])

    upload_pandas_df_to_s3(input_stats_history, files.S3_BUCKET, input_stats_history_path)

    if len(input_stats_history) >= m.INPUT_DRIFT_SPAN_IN_DAYS:
        lin_reg = LinearRegression()

        # Fit linear regression on last X days
        lin_reg.fit(np.array(range(m.INPUT_DRIFT_SPAN_IN_DAYS)).reshape(-1, 1),
                    input_stats_history.tail(m.INPUT_DRIFT_SPAN_IN_DAYS)["mean_income"].values)

        if abs(lin_reg.coef_) > m.DRIFT_THRESHOLD:
            msg = f"Incomes are evolving very quickly, model might become irrelevant soon !"
            logging.warning(msg)
            raise Exception(msg)


def monitor_loans_ratio(prediction_file_path, prediction_history_path, metrics_history_path):
    new_predictions = load_pandas_df_from_s3(files.S3_BUCKET, prediction_file_path)

    logging.info("Append new predictions to history and upload to s3.")
    datetime_now = datetime.now()
    new_predictions["timestamp"] = datetime_now
    try:
        prediction_history_history = load_pandas_df_from_s3(files.S3_BUCKET, prediction_history_path)
    except:
        logging.info("Prediction history dataframe doesn’t exist yet, creating it now.")
        prediction_history_history = None

    new_prediction_history_history = pd.concat([prediction_history_history, new_predictions])
    upload_pandas_df_to_s3(new_prediction_history_history, files.S3_BUCKET, prediction_history_path)

    logging.info("Compute new metrics and upload to s3.")
    accepted_loans_ratio = new_predictions["prediction"].value_counts()["Y"] / len(new_predictions)
    logging.info(f"Accepted loans ratio: {accepted_loans_ratio:.5f}")
    new_metric_df = pd.DataFrame.from_records([{"date": datetime_now, "accepted_loans_ratio": accepted_loans_ratio}])

    try:
        metrics_history = load_pandas_df_from_s3(files.S3_BUCKET, metrics_history_path)
    except:
        logging.info("Metrics history dataframe doesn’t exist yet, creating it now.")
        metrics_history = None

    new_metrics_history = pd.concat([metrics_history, new_metric_df])
    upload_pandas_df_to_s3(new_metrics_history, files.S3_BUCKET, metrics_history_path)

    # Alert if necessary
    if accepted_loans_ratio < 0.4:
        msg = f"Accepted loans ratio has dropped below 0.4 ! Currently at {accepted_loans_ratio:.2f}"
        logging.warning(msg)
        raise Exception(msg)
