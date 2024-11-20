#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define environment variables
export IMAGE_NAME="docker-shell"
export BASE_DIR=$(pwd)

# Dynamically convert the relative path to an absolute path for the secrets directory
export GOOGLE_CREDENTIALS_PATH=$(realpath "../../../../secrets")

# Check if the credentials file exists in the directory
if [ ! -f "$GOOGLE_CREDENTIALS_PATH/secret.json" ]; then
    echo "Error: Credentials file not found at $GOOGLE_CREDENTIALS_PATH/secret.json"
    exit 1
fi

# Determine if /bin/bash is passed as an argument
if [ "$1" == "/bin/bash" ]; then
    CMD="/bin/bash"
else
    CMD=""
fi

# Build the Docker image
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the Docker container
docker run --rm --name "${IMAGE_NAME}-shell" -ti \
    -v "$BASE_DIR":/app \
    -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/secret.json \
    "$IMAGE_NAME" $CMD
