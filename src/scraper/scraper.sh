#!/bin/bash
TODAY=$(date +'%Y-%m-%d %H:%M:%S')

# Load environment variables from the .env file
export $(grep -v '^#' .env | xargs)

# Check if the image already exists
if ! docker images $IMAGE_NAME | awk '{ print $1 }' | grep -q $IMAGE_NAME; then
    echo "Image does not exist. Building..."
    docker build -t $IMAGE_NAME .
else
    echo "Image already exists. Skipping build..."
fi

# Create temporary directories for scraped data
TEMP_METADATA=$(realpath $SCRAPED_METADATA)_tmp
TEMP_RAW_IMAGES=$(realpath $SCRAPED_RAW_IMAGES)_tmp

# Ensure the temporary directories are empty
rm -rf $TEMP_METADATA
rm -rf $TEMP_RAW_IMAGES
mkdir -p $TEMP_METADATA
mkdir -p $TEMP_RAW_IMAGES

# Run the scraper container and redirect output to a log file
docker run --rm --name $IMAGE_NAME \
    -v $(pwd):/src \
    -v $TEMP_METADATA:$SCRAPED_METADATA_CONTAINER \
    -v $TEMP_RAW_IMAGES:$SCRAPED_RAW_IMAGES_CONTAINER \
    $IMAGE_NAME > scraper.log 2>&1
CONTAINER_EXIT_CODE=$?

# Check if the container ran successfully
if [ $CONTAINER_EXIT_CODE -ne 0 ]; then
    echo "The scraper container encountered an issue. Checking logs..."
    
    # Show the logs (from scraper.log)
    cat scraper.log

    # Cleanup temporary directories
    rm -rf $TEMP_METADATA
    rm -rf $TEMP_RAW_IMAGES

    # Attempt to restore old data from DVC
    echo "Aborting script due to container failure. Restoring old data from DVC..."
    if ! dvc pull scraped_raw_data; then
        echo "Failed to restore old data. Please check DVC remote."
        exit 1
    fi
    exit 1
fi

# If the scraper succeeds, move the new data from the temporary directories to the actual directories
rm -rf $(realpath $SCRAPED_METADATA)/*
rm -rf $(realpath $SCRAPED_RAW_IMAGES)/*
mv $TEMP_METADATA/* $(realpath $SCRAPED_METADATA)
mv $TEMP_RAW_IMAGES/* $(realpath $SCRAPED_RAW_IMAGES)

# Clean up temporary directories
rm -rf $TEMP_METADATA
rm -rf $TEMP_RAW_IMAGES

# Proceed with the rest of the script if no issues
git pull --rebase

# Check if the pull created any conflicts
if [ $? -ne 0 ]; then
    echo "There was a merge conflict. Aborting script."
    exit 1
fi

# Add the scraped data to DVC only after ensuring there are no conflicts
dvc add $(realpath $SCRAPED_RAW_IMAGES)
dvc add $(realpath $SCRAPED_METADATA)

# Push data to DVC remote
dvc push --remote scraped_raw_data

# Commit the DVC changes to Git
git add $(realpath $SCRAPED_RAW_IMAGES).dvc
git add $(realpath $SCRAPED_METADATA).dvc
git add $(realpath $GIT_IGNORE) 

git commit -m "Scraped data for $TODAY"

# Tag the run with the current date and time
git tag run-$(date +'%Y-%m-%d-%H-%M-%S')

# Push the changes to Git
git push origin main
git push origin --tags

if [ $? -ne 0 ]; then
    echo "Failed to push changes to Git. Please resolve conflicts manually."
    exit 1
fi