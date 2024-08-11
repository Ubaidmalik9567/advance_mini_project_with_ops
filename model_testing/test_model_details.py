import unittest
import mlflow
import os
import logging
import pickle
import pandas as pd

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

        # Load the vectorizer
        try:
            with open('models/vectorizer.pkl', 'rb') as f:
                cls.vectorizer = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Vectorizer file not found. Ensure 'models/vectorizer.pkl' exists.")

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

    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)

        # Verify the input shape
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Verify the output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)  # Assuming a single output column for binary classification

if __name__ == "__main__":
    unittest.main()
