import unittest
import mlflow
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

        # Set the model name and stage
        cls.new_model_name = "save_model"
        cls.stage = "Production"  # Change stage if needed

        # Load the model from MLflow model registry
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name, cls.stage)
        if not cls.new_model_version:
            raise Exception(f"No model found in the '{cls.stage}' stage.")
        
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

    @staticmethod
    def get_latest_model_version(model_name, stage="Production"):
        client = mlflow.MlflowClient()
        model_versions = client.search_model_versions(f"name='{model_name}'")
        latest_version_info = next(
            (v for v in model_versions if v.current_stage == stage), None
        )
        return latest_version_info.version if latest_version_info else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

if __name__ == "__main__":
    unittest.main()
