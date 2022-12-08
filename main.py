import os
from datetime import datetime
import logging
import mlflow
from mlflow import log_metric, log_param, log_artifacts
from pathlib import Path

from src.utils import download_file_from_url
from src.preprocess.preprocess import preprocess, load_and_split_data
from src.logistic_reg.logistic_reg_train import logistic_reg_train
from src.evaluation.evaluate import evaluate
from src.predict.predict import predict
import src.constants.files as files
import src.constants.models as m


def main():
    """
    Launch all project steps.

    :return:
    """
    # ne marche pas avec mon proxy --> desctivé copié en local
    # download_file_from_url(files.LOANS_DATA_URL, files.LOANS)

    today_str = str(datetime.date(datetime.now()))

    mlflow.set_experiment(files.MLFLOW_EXPERIMENT_NAME)

    # TODO 1 : Créez un experiment MlFlow avec la fonction set_experiment et un nom MLFLOW_EXPERIMENT_NAME que vous trouverez dans
    #  src/constants/files.py puis lancez un run MlFlow dont le nom est la date du jour en remplaçant
    #  NotImplementedError par la méthode MlFlow appropriée.
    #  N’oubliez pas d’importer le package et de l’ajouter dans les requirements.

    with mlflow.start_run(run_name=today_str):
        logging.info("*********** 1/5 Loading and splitting data ***********")
        load_and_split_data(raw_data_path=files.LOANS,
                            training_file_path=files.TRAIN,
                            test_file_path=files.TEST)

        logging.info("*********** 2/5 Preprocessing ***********")
        preprocess(training_file_path=files.TRAIN,
                   preprocessed_train_path=files.PREPROCESSED_TRAIN,
                   preprocessing_pipeline_name=files.PREPROCESSING_PIPELINE)

        logging.info("*********** 3/5 Modeling ***********")
        logistic_reg_train(preprocessed_train_path=files.PREPROCESSED_TRAIN,
                           logistic_reg_model_name=m.LOGISTIC_REG_MODEL_NAME)

        logging.info("*********** 4/5 Prediction ***********")
        predict(test_file_path=files.TEST,
                preprocessing_pipeline_name=files.PREPROCESSING_PIPELINE,
                logistic_reg_model_name=m.LOGISTIC_REG_MODEL_NAME,
                prediction_file_path=files.PREDICTIONS_TEST)

        logging.info("*********** 5/5 Evaluation ***********")
        evaluate(prediction_file_path=files.PREDICTIONS_TEST)


if __name__ == "__main__":
    main()
