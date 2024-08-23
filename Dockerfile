# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies

RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords

    
# Copy the application code into the container
COPY testing_fastapi_code.py /app/

# Expose the port on which the FastAPI application will run
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "testing_fastapi_code:app", "--host", "0.0.0.0", "--port", "8000"]
