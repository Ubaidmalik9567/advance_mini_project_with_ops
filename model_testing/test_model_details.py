# that perform by tester, we just for practicing

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Ubaidmalik9567"
        repo_name = "mini_project_with_ops"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "save_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

    
    # @staticmethod
    # def get_latest_model_version(model_name, stage="Staging"):
    #     client = mlflow.MlflowClient()
    #     latest_version = client.get_latest_versions(model_name, stages=[stage])
    #     return latest_version[0].version if latest_version else None
    
    @staticmethod
    def get_latest_model_version(model_name):
        client = mlflow.MlflowClient()
        # Fetch all versions and sort them by version number
        versions = client.get_latest_versions(model_name)
        versions.sort(key=lambda v: int(v.version), reverse=True)
        return versions[0].version if versions else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)
    

    ''' above code is load model: is 1st model_testing stage which load model sucessfully from model resigtery'''
    ''' above code is model signature: is 2nd model_testing stage which tell expected input which us require or give same expected output which we want'''


if __name__ == "__main__":
    unittest.main()