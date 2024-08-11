import os
import mlflow
import logging

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
stage = "Production"  # Change stage to "Staging" or another stage if needed

# Initialize MlflowClient
client = mlflow.tracking.MlflowClient()

# Get the latest model version in the specified stage
model_versions = client.search_model_versions(f"name='{model_name}'")
latest_version_info = next(
    (v for v in model_versions if v.current_stage == stage), None
)

if not latest_version_info:
    raise Exception(f"No model found in the '{stage}' stage.")

# Construct the model URI
model_version = latest_version_info.version
model_uri = f'models:/{model_name}/{model_version}'

# Load the model
try:
    model = mlflow.pyfunc.load_model(model_uri)
    logging.info(f"Model loaded successfully from URI: {model_uri}")
    print(f"Model loaded successfully from URI: {model_uri}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise
