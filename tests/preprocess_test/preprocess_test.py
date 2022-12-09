import os
import pandas as pd
import mlflow
from unittest import TestCase
import shutil
import src.predict.predict as predict
import src.constants.files

from src.constants import files
from src.preprocess.preprocess import preprocess


LOCAL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
MLRUNS_PATH = files.create_folder(os.path.join("file:", LOCAL_ROOT, "mlruns_test").replace("C:", ""))


class PreprocessTest(TestCase):
    def setUp(self) -> None:
        # Is executed at the beginning of each test run
        mlflow.set_tracking_uri(MLRUNS_PATH)

    def tearDown(self) -> None:
        # Is executed at the end of each test run
        for i in range(10):
            mlruns_path_i = os.path.join(MLRUNS_PATH, str(i))
            if os.path.exists(mlruns_path_i):
                shutil.rmtree(mlruns_path_i)

    @staticmethod
    def test_preprocess():
        mlflow.set_experiment(files.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run():
            # Given
            # TODO 1 : initialiser la variable preprocessed_train_path avec le chemin vers le fichier result_test.csv
            #  du dossier preprocess_test, et la variable training_file_path avec le chemin vers le fichier
            #  loans_test.csv du même dossier. N’utilisez pas de chemin relatif et pensez à utiliser os.path.join pour
            #  une gestion cross-os des fichiers.
            preprocessed_train_path = os.path.join(LOCAL_ROOT, "result_test.csv")
            print("preprocessed_train_path" + preprocessed_train_path)
            training_file_path = os.path.join(LOCAL_ROOT, "loans_test.csv")
            print("training_file_path" + training_file_path)

            # When
            preprocess(
                training_file_path=training_file_path,
                preprocessed_train_path=preprocessed_train_path,
                preprocessing_pipeline_name=files.PREPROCESSING_PIPELINE
            )

            # Then
            # TODO 2 : charger le fichier expected.csv des résultats attendus.
            expected =  pd.read_csv(os.path.join(LOCAL_ROOT, "expected.csv"))
            # Read result from csv to avoid problems with nan
            result = pd.read_csv(preprocessed_train_path)

            pd.testing.assert_frame_equal(result, expected, check_dtype=False)

            try:
                # TODO 3 : charger le preprocessing pipeline depuis mlflow.
                #  indice : inspirez-vous du code de predict.py
                active_run_id = mlflow.active_run()


                mlflow.sklearn.load_model(os.path.join(active_run_id.info.artifact_uri, files.PREPROCESSING_PIPELINE)), active_run_id
                

            except IOError:
                raise AssertionError("The preprocessing pipeline has not been saved with mlflow")
