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

# Copy the rest of the application code to the working directory
COPY testing_app.py /app/
COPY models /app/models

# Expose the port on which the app will run
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "testing_app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
