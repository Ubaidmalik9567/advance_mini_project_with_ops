from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI()

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

# Load model and vectorizer from files
model_path = 'model.pkl'  # Adjust the path as needed
vectorizer_path = 'vectorizer.pkl'  # Adjust the path as needed

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
logging.info("Model loaded successfully.")

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
logging.info("Vectorizer loaded successfully.")

@app.post("/predict")
async def predict(text: str = Form(...)):
    try:
        # Clean and preprocess the input text
        text = normalize_text(text)

        # Vectorize the text
        features = vectorizer.transform([text])

        # Convert sparse matrix to DataFrame
        features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

        # Predict probabilities and class
        probabilities = model.predict_proba(features_df)[0]
        predicted_class = model.predict(features_df)[0]

        # Determine class labels
        class_labels = ['Sad', 'Happy']
        result = {
            'label': class_labels[int(predicted_class)],
            'probability': {
                'Happy': probabilities[1],
                'Sad': probabilities[0]
            }
        }

        return JSONResponse(content=result)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def read_root():
    return {"message": "Working fine"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # use this for running: uvicorn testing_app:app --reload
