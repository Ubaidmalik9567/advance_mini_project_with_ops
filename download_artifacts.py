import os
import mlflow
from mlflow.tracking import MlflowClient

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

client = MlflowClient()

# Specify the model name and stage
model_name = "save_model"
stage = "Production"

# Get the latest version in production
latest_version = client.get_latest_versions(model_name, stages=[stage])[0]

# Get the run ID associated with the latest model version
run_id = latest_version.run_id

# Specify the path to download the artifacts
download_path = "artifacts"

# Download all artifacts from the specified run
client.download_artifacts(run_id, "", download_path)

print(f"Artifacts downloaded to: {download_path}")
