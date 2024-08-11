import unittest
import mlflow
import os
import logging
import pickle

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
        cls.model_name = "save_model"
        cls.stage = "Production"  # Change stage if needed

        # Get the latest model version
        cls.run_id = cls.get_latest_model_version(cls.model_name, cls.stage)
        if not cls.run_id:
            raise Exception(f"No model found in the '{cls.stage}' stage.")

        cls.download_path = "artifacts"
        cls.download_artifacts(cls.run_id, cls.download_path)
        cls.new_model = cls.load_model_from_artifacts(cls.download_path)

    @staticmethod
    def get_latest_model_version(model_name, stage="Production"):
        client = mlflow.MlflowClient()
        model_versions = client.search_model_versions(f"name='{model_name}'")
        latest_version_info = next(
            (v for v in model_versions if v.current_stage == stage), None
        )
        return latest_version_info.run_id if latest_version_info else None

    @staticmethod
    def download_artifacts(run_id, download_path):
        client = mlflow.MlflowClient()
        os.makedirs(download_path, exist_ok=True)
        client.download_artifacts(run_id, "", download_path)
        logging.info(f"Artifacts downloaded to: {download_path}")

        # Log all files found in the download path
        for root, dirs, files in os.walk(download_path):
            for file in files:
                logging.info(f"Found file: {os.path.join(root, file)}")

    @staticmethod
    def load_model_from_artifacts(download_path):
        model_pkl_path = None
        for root, dirs, files in os.walk(download_path):
            if 'model.pkl' in files:
                model_pkl_path = os.path.join(root, 'model.pkl')
                break

        if model_pkl_path:
            logging.info(f"Found model.pkl at: {model_pkl_path}")
            with open(model_pkl_path, 'rb') as model_file:
                model = pickle.load(model_file)
            logging.info("Model loaded successfully.")
            return model
        else:
            logging.error("model.pkl not found in downloaded artifacts.")
            return None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

if __name__ == "__main__":
    unittest.main()
