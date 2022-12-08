import pandas as pd
import logging
import mlflow

import src.constants.columns as c
import src.constants.models as m
from sklearn.metrics import f1_score


def evaluate(prediction_file_path):
    """
    Evaluate the performance of the model on test data.

    :param prediction_file_path: path to prediction data.

    :return: None
    """
    prediction_df = pd.read_csv(prediction_file_path)

    logging.info(f"Evaluating {m.LOGISTIC_REG_MODEL_NAME}")

    y_test = prediction_df[c.Loans.target()].values
    y_pred = prediction_df["prediction"].values

    score = round(f1_score(y_test, y_pred, pos_label="Y"), 2)
    # TODO 6 : Logging du f1 score comme métrique
    mlflow.log_metric( 'f1', score)

    logging.info(f"F1 score for model {m.LOGISTIC_REG_MODEL_NAME} is {score}")
