# Use the official Python base image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files to disk and to enable unbuffered mode
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/

# Install the Python dependencies
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet

# Copy the rest of the application code to the working directory
COPY testing_fastapi_code.py /app/
# COPY models /app/models

# Expose the port on which the app will run
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "testing_fastapi_code:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# Stage 1: Build Stage
# FROM python:3.10 AS build

# # Set the working directory in the container
# WORKDIR /app

# # Copy the requirements.txt file from the flask_app folder
# COPY requirements.txt /app/

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the application code and model files
# COPY testing_fastapi_code.py /app/
# # COPY model.pkl /app/
# # COPY vectorizer.pkl /app/

# # Download only the necessary NLTK data
# RUN python -m nltk.downloader stopwords wordnet

# # Stage 2: Final Stage
# FROM python:3.10-slim AS final

# WORKDIR /app

# # Copy only the necessary files from the build stage
# COPY --from=build /app /app

# # Expose the application port
# EXPOSE 8000

# # Set the command to run the application
# CMD ["uvicorn", "testing_fastapi_code:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
