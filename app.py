from flask import Flask, render_template_string, request
import mlflow
import pickle
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import dagshub

# nltk.download('stopwords')

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

dagshub.init(repo_owner='Ubaidmalik9567', repo_name='mini_project_with_ops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Ubaidmalik9567/mini_project_with_ops.mlflow")

app = Flask(__name__)

def get_latest_model_run_id(model_name, stage="Production"):
    client = mlflow.MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    latest_version_info = next((v for v in model_versions if v.current_stage == stage), None)
    return latest_version_info.run_id if latest_version_info else None

def load_model_and_vectorizer():
    model_name = "save_model"
    stage = "Production"
    run_id = get_latest_model_run_id(model_name, stage)
    if not run_id:
        raise Exception(f"No model found in the '{stage}' stage.")

    # Load the model directly from MLflow
    model_uri = f"runs:/{run_id}/model/model.pkl"
    model_path = mlflow.artifacts.download_artifacts(model_uri)
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    logging.info("Model loaded successfully.")

    # Load the vectorizer directly from MLflow
    vectorizer_uri = f"runs:/{run_id}/vectorizer.pkl"
    vectorizer_path = mlflow.artifacts.download_artifacts(vectorizer_uri)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    logging.info("Vectorizer loaded successfully.")

    return model, vectorizer

# Load model and vectorizer at startup
model, vectorizer = load_model_and_vectorizer()

# HTML template as a string
html_template = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Sentiment Analysis</title>
    </head>
    <body>
        <h1>Sentiment Analysis</h1>
        <form action="/predict" method="POST">
            <label>Write text:</label><br>
            <textarea name="text" rows="10" cols="40"></textarea><br>
            <input type="submit" value="Predict">
        </form>
        {% if result %}
            <h2>Prediction: {{ result.label }}</h2>
            <p>Probability of Happy: {{ result.probability[1] }}</p>
            <p>Probability of Sad: {{ result.probability[0] }}</p>
        {% endif %}
    </body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(html_template, result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Clean the input text
    text = normalize_text(text)

    # Vectorize the text
    features = vectorizer.transform([text])

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # Predict probabilities and class
    probabilities = model.predict_proba(features_df)[0]
    predicted_class = model.predict(features_df)[0]

    # Determine class labels
    class_labels = ['Sad', 'Happy']
    result = {
        'label': class_labels[int(predicted_class)],
        'probability': probabilities
    }

    # Log predictions and probabilities for debugging
    logging.info(f"Predicted class: {result['label']}")
    logging.info(f"Predicted probabilities: {result['probability']}")

    # Show result
    return render_template_string(html_template, result=result)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
