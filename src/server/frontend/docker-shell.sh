#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define environment variables
export IMAGE_NAME="frontend-app"
export BASE_DIR=$(pwd)

# Build the Docker image
docker build -t $IMAGE_NAME -f Dockerfile .

# Determine if /bin/bash is passed as an argument
if [ "$1" == "/bin/bash" ]; then
    CMD="/bin/bash"
else
    CMD=""
fi

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
    -v "$BASE_DIR":/app \
    -p 8080:8080 $IMAGE_NAME $CMD