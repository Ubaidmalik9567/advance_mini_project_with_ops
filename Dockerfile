# Use a smaller base image
FROM python:3.10-slim AS builder

# Set the working directory in the container
WORKDIR /app

# Install build dependencies and install requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader stopwords wordnet \
    && apt-get purge -y --auto-remove gcc libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files into the container
COPY testing_fastapi_code.py /app/
# COPY model.pkl /app/
# COPY vectorizer.pkl /app/

# Final stage: use a smaller base image to reduce size further
FROM python:3.10-alpine

# Set the working directory in the container
WORKDIR /app

# Copy installed packages and application code from the builder
COPY --from=builder /app /app

# Expose the port on which the FastAPI application will run
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "testing_fastapi_code:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
