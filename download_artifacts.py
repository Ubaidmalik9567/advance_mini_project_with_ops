import os
import mlflow
from mlflow.tracking import MlflowClient
import logging
import shutil

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

    return latest_version_info.run_id

def download_specific_artifact(run_id, artifact_path, download_path):
    client = MlflowClient()
    # Create the download path if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # Download only the specified artifact
    artifact_download_path = client.download_artifacts(run_id, artifact_path, download_path)
    
    # Log the downloaded file path
    logging.info(f"Artifact downloaded to: {artifact_download_path}")

    return artifact_download_path

def main():
    setup_mlflow_tracking()
    model_name = "save_model"
    stage = "Production"

    try:
        # Get the latest model version information
        run_id = get_latest_model_version(model_name, stage)

        # Specify the artifact path and download path
        artifact_path = "model.pkl"
        download_path = "artifacts"

        # Download only the model.pkl artifact
        model_pkl_path = download_specific_artifact(run_id, artifact_path, download_path)
        
        # Log the path where model.pkl is saved
        logging.info(f"model.pkl saved at: {model_pkl_path}")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
