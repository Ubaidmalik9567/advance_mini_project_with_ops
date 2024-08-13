import os
import mlflow
from mlflow.tracking import MlflowClient
import logging
import pickle

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

def list_artifacts(run_id):
    client = MlflowClient()
    # List all artifact paths
    artifact_paths = []
    for artifact in client.list_artifacts(run_id):
        artifact_paths.append(artifact.path)
        logging.info(f"Artifact: {artifact.path}")
    
    return artifact_paths

def download_specific_artifact(run_id, artifact_path, download_path):
    client = MlflowClient()
    os.makedirs(download_path, exist_ok=True)
    # Download only the specified artifact
    artifact_download_path = client.download_artifacts(run_id, artifact_path, download_path)
    logging.info(f"Artifact downloaded to: {artifact_download_path}")
    return artifact_download_path

def load_model_from_artifacts(download_path):
    model_pkl_path = None
    vectorizer_pkl_path = None
    for root, dirs, files in os.walk(download_path):
        if 'model.pkl' in files:
            model_pkl_path = os.path.join(root, 'model.pkl')
        if 'vectorizer.pkl' in files:
            vectorizer_pkl_path = os.path.join(root, 'vectorizer.pkl')
        
    if model_pkl_path:
        logging.info(f"Found model.pkl at: {model_pkl_path}")
        with open(model_pkl_path, 'rb') as model_file:
            model = pickle.load(model_file)
        logging.info("Model loaded successfully.")
        return model, vectorizer_pkl_path
    else:
        logging.error("model.pkl not found in downloaded artifacts.")
        return None, vectorizer_pkl_path

def main():
    setup_mlflow_tracking()
    model_name = "save_model"
    stage = "Production"

    try:
        run_id = get_latest_model_version(model_name, stage)
        artifact_paths = list_artifacts(run_id)

        # Check and download specific artifacts
        for artifact in ['model.pkl', 'vectorizer.pkl']:
            if artifact in artifact_paths:
                download_path = "artifacts"
                artifact_download_path = download_specific_artifact(run_id, artifact, download_path)
                logging.info(f"{artifact} saved at: {artifact_download_path}")

        model, vectorizer_pkl_path = load_model_from_artifacts(download_path)
        if vectorizer_pkl_path:
            logging.info(f"Found vectorizer.pkl at: {vectorizer_pkl_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
