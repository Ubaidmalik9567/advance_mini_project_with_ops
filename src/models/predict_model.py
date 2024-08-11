import pandas as pd
import sys
import pathlib
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json
import logging
import yaml
import mlflow
import os
import dagshub

# Set up DagsHub credentials for MLflow tracking by using key base authentication
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_split_data(dataset_path: str) -> tuple:
    try:
        logging.info(f"Loading dataset from {dataset_path}.")
        dataset = pd.read_csv(dataset_path)
        xtest = dataset.iloc[:, 0:-1]
        ytest = dataset.iloc[:, -1]
        logging.info("Data loaded and split successfully.")
        return xtest, ytest
    except Exception as e:
        logging.error(f"Error loading or splitting data: {e}")
        raise

def load_save_model(file_path: str):
    try:
        logging.info(f"Loading model from {file_path}.")
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        logging.info("Evaluating model performance.")
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info("Model evaluation completed successfully.")
        return metrics_dict
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving metrics to {file_path}/metrics.json.")
        with open(file_path + "/metrics.json", 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info("Metrics saved successfully.")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise

def save_model_info(run_id, model_path, file_path) -> None: # that info use for model registry
    model_info = {'run_id': run_id, 'model_path': model_path}
    with open(file_path, 'w') as file:
        json.dump(model_info, file, indent=4)

def random_sample_csv(csv_path, num_samples):
    df = pd.read_csv(csv_path)
    # Perform random sampling
    sampled_df = df.sample(n=num_samples, random_state=1)
    return sampled_df

def main():
    
    mlflow.set_experiment("dvc-pipeline")  # Set up MLflow experiment
    with mlflow.start_run(run_name="pred2prod_files-run") as run:  # Start MLflow run
        
        try:
            current_dir = pathlib.Path(__file__)
            home_dir = current_dir.parent.parent.parent

            path = sys.argv[1]
            save_metrics_location = home_dir.as_posix() + "/reports"
            processed_datasets_path = home_dir.as_posix() + path + "/processed_testdata.csv"
            trained_model_path = home_dir.as_posix() + "/models/model.pkl"
            raw_csv_path = home_dir.as_posix() + "/data/raw/raw.csv"

            # Perform random sampling
            sampled_data = random_sample_csv(raw_csv_path, num_samples=200)
            sampled_data_path = home_dir.as_posix() + "/reports/sampled_data.csv"
            sampled_data.to_csv(sampled_data_path, index=False)
            mlflow.log_artifact(sampled_data_path)

            x, y = load_and_split_data(processed_datasets_path)
            model = load_save_model(trained_model_path)

            metrics_dict = evaluate_model(model, x, y)
            save_metrics(metrics_dict, save_metrics_location)

            with open("params.yaml", "r") as file:
                params = yaml.safe_load(file)

            # Log parameters from params.yaml
            for param, value in params.items():
                for key, value in value.items():
                    mlflow.log_param(f'{param}_{key}', value)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics_dict.items():
                mlflow.log_metric(metric_name, metric_value)
                    
            # Log all model parameters to MLflow
            if hasattr(model, 'get_params'):
                model_params = model.get_params()
                for param_name, param_value in model_params.items():
                    mlflow.log_param(param_name, param_value)
            
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact("models/vectorizer.pkl")

            save_model_info(run.info.run_id, "models", 'reports/model_experiment_info.json')  # Save model info
            
            # Log the metrics, model info file to MLflow
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact('reports/model_experiment_info.json')

            logging.info("Main function completed successfully.")
        except Exception as e:
            logging.error(f"Error in main function: {e}")
            raise

if __name__ == "__main__":
    main()
