import os
import logging
import mlflow
import pytest
import dagshub
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@pytest.fixture(scope="module")
def setup_environment():
    dagshub.init(repo_owner='Ubaidmalik9567', repo_name='mini_project_with_ops', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/Ubaidmalik9567/mini_project_with_ops.mlflow")
    
    model_name = "save_model"
    stage = "Staging"  # Change stage if needed

    # Get the latest model version
    run_id = get_latest_model_version(model_name, stage)
    if not run_id:
        pytest.fail(f"No model found in the '{stage}' stage.")

    # Log details about the model
    logging.info(f"Model Name: {model_name}")
    logging.info(f"Stage: {stage}")
    logging.info(f"Run ID: {run_id}")

    return run_id, stage


def get_latest_model_version(model_name, stage="Production"):
    client = mlflow.MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    latest_version_info = next(
        (v for v in model_versions if v.current_stage == stage), None
    )
    return latest_version_info.run_id if latest_version_info else None


@pytest.mark.usefixtures("setup_environment")
def test_model_logging(setup_environment):
    run_id, stage = setup_environment
    assert run_id, "Run ID should not be None"
    logging.info(f"Successfully fetched model with Run ID: {run_id} in stage: {stage}")
