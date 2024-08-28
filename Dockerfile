# Use a smaller base image
FROM python:3.10-slim AS builder

# Set the working directory in the container
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev

# Upgrade pip and install the dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader stopwords wordnet

# Clean up unnecessary files and packages to reduce image size
RUN apt-get purge -y --auto-remove gcc libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files into the container
COPY testing_fastapi_code.py /app/
# COPY model.pkl /app/
# COPY vectorizer.pkl /app/

# Expose the port on which the FastAPI application will run
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "testing_fastapi_code:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
