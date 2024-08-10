import unittest
import mlflow
import os

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
        if cls.new_model_version is not None:
            cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
            print(f"Loading model from URI: {cls.new_model_uri}")
            cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
        else:
            raise ValueError(f"No available version for model '{cls.new_model_name}'")

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

if __name__ == "__main__":
    unittest.main()
