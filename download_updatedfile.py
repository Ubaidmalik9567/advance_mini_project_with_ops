import os
import mlflow
from mlflow.tracking import MlflowClient
import logging
import pickle
import dagshub

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_mlflow_tracking():
    dagshub.init(repo_owner='Ubaidmalik9567', repo_name='mini_project_with_ops', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/Ubaidmalik9567/mini_project_with_ops.mlflow")

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
    os.makedirs(download_path, exist_ok=True)
    client.download_artifacts(run_id, artifact_path, download_path)
    logging.info(f"Artifact '{artifact_path}' downloaded to: {download_path}")

def load_model_and_vectorizer(download_path):
    model_pkl_path = os.path.join(download_path, 'model.pkl')
    vectorizer_pkl_path = os.path.join(download_path, 'vectorizer.pkl')
    
    if os.path.exists(model_pkl_path) and os.path.exists(vectorizer_pkl_path):
        logging.info(f"Found model.pkl at: {model_pkl_path}")
        logging.info(f"Found vectorizer.pkl at: {vectorizer_pkl_path}")
        with open(model_pkl_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(vectorizer_pkl_path, 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        logging.info("Model and vectorizer loaded successfully.")
        return model, vectorizer
    else:
        logging.error("Model or vectorizer file not found in downloaded artifacts.")
        return None, None

# Main function to execute the process
def main():
    setup_mlflow_tracking()
    model_name = "save_model"
    stage = "Production"

    try:
        run_id = get_latest_model_version(model_name, stage)
        download_path = "updated_artifacts/model"
        
        # Download only the model and vectorizer files
        download_specific_artifact(run_id, "model/model.pkl", download_path)
        download_specific_artifact(run_id, "vectorizer.pkl", "updated_artifacts")
        
        model, vectorizer = load_model_and_vectorizer(download_path)
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
