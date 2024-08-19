# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "testing_app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
