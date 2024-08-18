# Use a specific Python version
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the application code, requirements file, and model/vectorizer files into the container
COPY requirements.txt /app/
COPY testing_app.py /app/
COPY models/model.pkl /app/models/model.pkl
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download NLTK data required for text preprocessing
RUN python -m nltk.downloader stopwords wordnet

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "testing_app:app", "--host", "0.0.0.0", "--port", "8000"]
