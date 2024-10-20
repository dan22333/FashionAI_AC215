#!/bin/bash

# Load environment variables from the .env file
export $(grep -v '^#' .env | xargs)

cd ../../
export GOOGLE_APPLICATION_CREDENTIALS=$PATH_TO_SECRET_KEY
pipenv run dvc pull --remote fashion_ai_models
cd src/inference

# Check if the image already exists
if ! docker images $IMAGE_NAME | awk '{ print $1 }' | grep -q $IMAGE_NAME; then
    echo "Image does not exist. Building..."
    docker build -t $IMAGE_NAME .
else
    echo "Image already exists. Skipping build..."
fi

# Run the scraper container and redirect output to a log file
docker run --rm --name $IMAGE_NAME \
    -v $(pwd):/src \
    $IMAGE_NAME
