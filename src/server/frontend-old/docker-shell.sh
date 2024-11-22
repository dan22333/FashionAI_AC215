#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define environment variables
export IMAGE_NAME="frontend-app"
export BASE_DIR=$(pwd)

# Build the Docker image
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
    -v "$BASE_DIR":/app \
    -p 3000:3000 $IMAGE_NAME