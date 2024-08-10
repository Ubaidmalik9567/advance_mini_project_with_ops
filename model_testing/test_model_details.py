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

    @staticmethod
    def get_latest_model_version(model_name, stage="Production"):
        client = mlflow.MlflowClient()
        # First try to get the latest version in the "Production" stage
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        
        if not latest_versions:
            # If no versions in "Production", fallback to "None"
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
        
        if latest_versions:
            latest_version = latest_versions[0]
            version = latest_version.version
            run_id = latest_version.run_id  # Get the run_id of the model version
            model_uri = f"models:/{model_name}/{version}"
            
            print(f"Model Name: {model_name}")
            print(f"Latest Model Version: {version}")
            print(f"Model URI: {model_uri}")
            print(f"Run ID: {run_id}")  # Print the run_id
            
            return version
        else:
            print(f"No versions found for model: {model_name}")
            return None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)
    

    ''' above code is load model: is 1st model_testing stage which load model successfully from model registry'''
    ''' above code is model signature: is 2nd model_testing stage which tells expected input which us require or give same expected output which we want'''


if __name__ == "__main__":
    unittest.main()
