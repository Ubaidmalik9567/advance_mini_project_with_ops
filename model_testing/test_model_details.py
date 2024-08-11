import unittest
import mlflow
import os
import logging
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        vectorizer_path = os.path.join(cls.download_path, 'vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                cls.vectorizer = pickle.load(f)
        else:
            raise FileNotFoundError("Vectorizer file not found. Ensure 'vectorizer.pkl' exists in the artifacts.")

        # Load the holdout data for performance testing
        holdout_data_path = os.path.join(cls.download_path, 'holdout_data.csv')
        if os.path.exists(holdout_data_path):
            cls.holdout_data = pd.read_csv(holdout_data_path)
        else:
            raise FileNotFoundError("Holdout data file not found. Ensure 'holdout_data.csv' exists in the artifacts.")

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

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')

if __name__ == "__main__":
    unittest.main()
