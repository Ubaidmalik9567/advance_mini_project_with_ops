import os
import mlflow
from mlflow.tracking import MlflowClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_mlflow_tracking():
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

def get_latest_model_version(model_name, stage):
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    latest_version_info = next(
        (v for v in model_versions if v.current_stage == stage), None
    )

    if not latest_version_info:
        raise Exception(f"No model found in the '{stage}' stage.")

    return latest_version_info

def list_artifacts(run_id):
    client = MlflowClient()
    artifacts_info = client.list_artifacts(run_id)
    
    # Print directory structure
    for artifact in artifacts_info:
        logging.info(f"Artifact: {artifact.path}")

def main():
    setup_mlflow_tracking()
    model_name = "save_model"
    stage = "Production"

    try:
        # Get the latest model version information
        latest_version_info = get_latest_model_version(model_name, stage)
        run_id = latest_version_info.run_id
        model_version = latest_version_info.version
        
        logging.info(f"Run ID: {run_id}")
        logging.info(f"Model Version: {model_version}")

        # List artifacts
        list_artifacts(run_id)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
