import os
import pandas as pd
import logging
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

import src.constants.files as files


def predict(test_file_path, preprocessing_pipeline_name, logistic_reg_model_name, prediction_file_path):
    """
    Make prediction on test data with saved model.

    :param test_file_path: path to test data.
    :param preprocessing_pipeline_name: name of the preprocessing pipeline saved with mlflow.
    :param logistic_reg_model_name: name of logistic reg model to be saved with mlflow.
    :param prediction_file_path: path to prediction data.

    :return: None
    """
    test_df = pd.read_csv(test_file_path)

    current_run_id = mlflow.active_run().info.run_id

    logging.info("Loading preprocessing pipeline")
    preprocessing_pipeline, pp_run_id = load_latest_preprocessing_pipeline(
        preprocessing_pipeline_name, files.MLFLOW_EXPERIMENT_NAME)
    if pp_run_id != current_run_id:
        logging.info("Using preprocessing pipeline saved in a previous run")
        mlflow.log_param("preprocessing_pipeline_run_id", pp_run_id)

    preprocessed_test = preprocessing_pipeline.transform(test_df)

    logging.info("Loading trained model")
    logistic_reg = mlflow.sklearn.load_model(
        os.path.join(mlflow.active_run().info.artifact_uri, logistic_reg_model_name))

    logging.info(f"Make predictions with {logistic_reg_model_name}")
    y_pred = logistic_reg.predict(preprocessed_test)

    test_df["prediction"] = y_pred

    logging.info("Saving prediction results")
    test_df.to_csv(prediction_file_path, index=False)
    print("fffffffffffffffffffffffffffffff" + prediction_file_path)


def load_latest_preprocessing_pipeline(preprocessing_pipeline_name, experiment_name):
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    runs = MlflowClient().search_runs(
        experiment_ids=str(experiment_id),
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["tag.start_time DESC"]
    )

    for run in runs:
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri
        try:
            return mlflow.sklearn.load_model(os.path.join(artifact_uri, preprocessing_pipeline_name)), run_id
        except:
            continue

    raise Exception(
        f"Could not find a preprocessing pipeline named {preprocessing_pipeline_name} in current and previous runs"
    )
