# Use Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install pipenv and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    pip install pipenv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy Pipfile and Pipfile.lock for dependency management
COPY Pipfile Pipfile.lock /app/

# Install dependencies in system environment
RUN pipenv install --deploy --system

# Copy application files
COPY . /app

# Expose the application port
EXPOSE 8000

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
