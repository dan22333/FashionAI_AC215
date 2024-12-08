# Load environment variables from the .env file
export $(grep -v '^#' .env | xargs)

docker build -t $IMAGE_NAME .

# Run the scraper container and redirect output to a log file
docker run --rm --name $IMAGE_NAME \
    -v $(realpath ${SECRETS_PATH}${SECRET_FILE_NAME}):/secrets/$SECRET_FILE_NAME:ro \
    -e GOOGLE_APPLICATION_CREDENTIALS="/secrets/$SECRET_FILE_NAME" \
    -v $(realpath ${SECRETS_PATH}huggingface_key.json):/secrets/huggingface_key.json:ro \
    $IMAGE_NAME

CONTAINER_EXIT_CODE=$?