# Use a specific Python version
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache for dependencies
COPY requirements.txt /app/

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Download NLTK data required for text preprocessing
RUN python -m nltk.downloader stopwords wordnet

# Copy the rest of the application code into the container
COPY testing_app.py /app/
COPY models/model.pkl /app/models/model.pkl
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "testing_app:app", "--host", "0.0.0.0", "--port", "8000"]
