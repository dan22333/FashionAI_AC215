#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define environment variables
export IMAGE_NAME="backend-app"
export BASE_DIR=$(pwd)

# Convert the relative path to an absolute path for the secrets directory
export GOOGLE_CREDENTIALS_PATH="../../../../secrets"

# Check if the credentials file exists in the directory
if [ ! -f "$GOOGLE_CREDENTIALS_PATH/secret.json" ]; then
    echo "Error: Credentials file not found at $GOOGLE_CREDENTIALS_PATH/secret.json"
    exit 1
fi

# Build the Docker image
docker build -t $IMAGE_NAME -f Dockerfile .

IMAGE_NAME="backend-app" && BASE_DIR=$(pwd) && GOOGLE_CREDENTIALS_PATH=$(realpath "$GOOGLE_CREDENTIALS_PATH") && \
if [ ! -f "$GOOGLE_CREDENTIALS_PATH/secret.json" ]; then echo "Error: Credentials file not found at $GOOGLE_CREDENTIALS_PATH/secret.json" && exit 1; fi && \
docker build -t "$IMAGE_NAME" -f Dockerfile . && \
docker run --rm --name "${IMAGE_NAME}-shell" -ti \
    -v "$BASE_DIR":/app \
    -v "$GOOGLE_CREDENTIALS_PATH":/secrets \
    -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/secret.json \
    -p 8000:8000 "$IMAGE_NAME"
