import os
import mlflow
from mlflow.tracking import MlflowClient
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
model_name = "save_model"
stage = "Production"  # Change stage to "Production" to get the production model

# Initialize MlflowClient
client = MlflowClient()

# Get the latest model version in the specified stage
model_versions = client.search_model_versions(f"name='{model_name}'")
latest_version_info = next(
    (v for v in model_versions if v.current_stage == stage), None
)

if not latest_version_info:
    raise Exception(f"No model found in the '{stage}' stage.")

# Extract the run ID
run_id = latest_version_info.run_id
logging.info(f"Downloading artifacts for run ID: {run_id}")

# Specify the download path for artifacts
download_path = "artifacts"
os.makedirs(download_path, exist_ok=True)

# Download artifacts
client.download_artifacts(run_id, "", download_path)
logging.info(f"Artifacts downloaded to: {download_path}")

# Locate and load the model.pkl file
model_pkl_path = os.path.join(download_path, 'model.pkl')

if os.path.isfile(model_pkl_path):
    logging.info(f"Found model.pkl at: {model_pkl_path}")
    # Load the model.pkl
    with open(model_pkl_path, 'rb') as model_file:
        model = pickle.load(model_file)
    logging.info("Model loaded successfully.")
else:
    logging.error("model.pkl not found in downloaded artifacts.")
