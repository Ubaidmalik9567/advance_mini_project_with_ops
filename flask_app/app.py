from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import nltk
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Text preprocessing functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in text.split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    return re.sub('[%s]' % re.escape(string.punctuation), ' ', text).strip()

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

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

app = Flask(__name__)

def get_latest_model_run_id(model_name, stage="Production"):
    client = mlflow.MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    latest_version_info = next((v for v in model_versions if v.current_stage == stage), None)
    return latest_version_info.run_id if latest_version_info else None

def download_artifacts(run_id, download_path):
    client = mlflow.MlflowClient()
    os.makedirs(download_path, exist_ok=True)
    client.download_artifacts(run_id, "", download_path)
    logging.info(f"Artifacts downloaded to: {download_path}")

    # Log all files found in the download path
    for root, dirs, files in os.walk(download_path):
        for file in files:
            logging.info(f"Found file: {os.path.join(root, file)}")

def load_model_and_vectorizer():
    model_name = "save_model"
    stage = "Production"
    run_id = get_latest_model_run_id(model_name, stage)
    if not run_id:
        raise Exception(f"No model found in the '{stage}' stage.")

    download_path = "artifacts"
    download_artifacts(run_id, download_path)

    # Load the model
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
    else:
        raise FileNotFoundError("model.pkl not found in downloaded artifacts.")

    # Load the vectorizer
    vectorizer_path = os.path.join(download_path, 'vectorizer.pkl')
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logging.info(f"Vectorizer loaded successfully from {vectorizer_path}")
    else:
        raise FileNotFoundError("Vectorizer file not found. Ensure 'vectorizer.pkl' exists in the artifacts.")

    return model, vectorizer

# Load model and vectorizer at startup
model, vectorizer = load_model_and_vectorizer()

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Clean and preprocess the text
    text = normalize_text(text)

    # Transform text to features
    features = vectorizer.transform([text])

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # Make prediction
    result = model.predict(features_df)

    # Show result
    return render_template('index.html', result=result[0])

if __name__ == "__main__":
    app.run(debug=True)
