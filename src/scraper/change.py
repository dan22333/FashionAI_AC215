from google.cloud import storage
from apify_client import ApifyClient
import pandas as pd
from io import StringIO
import requests
from dotenv import load_dotenv
import os
import sys
import aiohttp
import asyncio
from google.cloud import secretmanager
from apify import Actor
from aiohttp import ClientTimeout
# Load the .env file
load_dotenv()

# Initialize configuration variables
metadata_folder = os.getenv('SCRAPED_METADATA')
images_folder = os.getenv('SCRAPED_RAW_IMAGES')
topics_file = os.getenv('TOPICS_FILE')  # File containing topics and URLs
column_topic = os.getenv('COLUMN_TOPIC_NAME')
column_url = os.getenv('COLUMN_URL_NAME')
gcp_bucket_name = os.getenv('GCP_BUCKET_NAME')
max_retries = int(os.getenv('MAX_RETRIES', 3))
scrape_data_option = bool(int(os.getenv('SCRAPE_DATA', 1)))
num_items_to_download = int(os.getenv('MAX_ITEMS'))

# Initialize Apify client with the secret token
apify_client = ApifyClient(os.getenv('APIFY_API_KEY'))

# Initialize GCP Storage client
storage_client = storage.Client()

# Function to fetch metadata from a given URL using Apify
def scrape_metadata(topic_name, url, timestamp):
    print(f"Scraping metadata for topic: {topic_name}")
    metadata_path = f"{metadata_folder}/{topic_name}/{timestamp}"
    os.makedirs(metadata_path, exist_ok=True)

    # Fetch data using Apify Actor
    run_input = {
        "startUrls": [{"url": url}],
        "maxRequestsPerCrawl": num_items_to_download,
        "proxy": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]},
        "maxConcurrency": 10,
    }

    run = apify_client.actor("mKTnbkisJ8BAiIbsP").call(run_input=run_input)
    dataset_id = run["defaultDatasetId"]

    # Fetch the CSV dataset from Apify
    api_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?format=csv"
    response = requests.get(api_url)
    csv_data = StringIO(response.content.decode('utf-8-sig'))
    metadata = pd.read_csv(csv_data)

    metadata_file = f"{metadata_path}/metadata.csv"
    metadata.to_csv(metadata_file, index=False)
    print(f"Metadata saved for topic: {topic_name}")
    return metadata, metadata_file

# Function to asynchronously download images
async def download_images(metadata, topic_name, timestamp):
    print(f"Downloading images for topic: {topic_name}")
    images_path = f"{images_folder}/{topic_name}/{timestamp}"
    os.makedirs(images_path, exist_ok=True)  # Ensure the output folder exists

    bad_urls = []  # List to store information about failed downloads

    # Set up Apify proxy configuration to use residential proxies
    async with Actor:
        proxy_configuration = await Actor.create_proxy_configuration(groups=['RESIDENTIAL'])
        proxy_url = await proxy_configuration.new_url()  # Get the proxy URL

        # Create an aiohttp session with a limited connection pool
        connector = aiohttp.TCPConnector(limit_per_host=30)  # Limit concurrent connections per host
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i, row in metadata.iterrows():
                url = row[column_url]
                if url:
                    # Construct the image name based on the id column
                    image_id = row.get("id", f"{i}")
                    image_name = os.path.join(images_path, f"image_{image_id}.jpg")

                    # Schedule the download task, passing the proxy_url
                    tasks.append(download_image(session, url, image_name, bad_urls, image_id, proxy_url))
                else:
                    print(f"URL missing in row {i + 1}")
                    bad_urls.append({'url': 'Missing', 'id': row.get("id", f"{i}"), 'error': 'No URL provided'})

            # Await the completion of all tasks
            await asyncio.gather(*tasks)

    # Log any bad URLs
    if bad_urls:
        print(f"Bad URLs collected: {len(bad_urls)}")
        bad_urls_df = pd.DataFrame(bad_urls)
        bad_urls_df.to_csv(os.path.join(images_path, "bad_urls.csv"), index=False)
    else:
        print("No bad URLs encountered.")

    print(f"Images downloaded for topic: {topic_name}")
    return images_path


# Helper function for individual image download
async def download_image(session, url, output_path, retries=0, max_retries=3, proxy_url=None, bad_urls=None, id=None):
    # Initialize bad_urls list if not provided
    if bad_urls is None:
        bad_urls = []

    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping download.")
        return True

    # Retry logic
    if retries >= max_retries:
        print(f"Max retries reached for {url}. Skipping.")
        if bad_urls is not None and id is not None:
            bad_urls.append({'url': url, 'id': id, 'error': 'Max retries reached'})
        return False

    try:
        # Fetch the image using the provided proxy
        async with session.get(url, proxy=proxy_url, timeout=ClientTimeout(total=600)) as response:
            if response.status == 200:
                # Save the image
                with open(output_path, 'wb') as f:
                    f.write(await response.read())
                print(f"Photo successfully downloaded as {output_path}")
                return True
            else:
                # Log the failed attempt
                print(f"Failed to download {output_path}. Status code: {response.status}")
                return await download_image(session, url, output_path, retries + 1, max_retries, proxy_url, bad_urls, id)
    except Exception as e:
        # Log exceptions and retry
        print(f"Error downloading {url}: {e}")
        return await download_image(session, url, output_path, retries + 1, max_retries, proxy_url, bad_urls, id)

# Function to upload metadata and images to GCP
def upload_to_gcp(metadata_file, images_path, topic_name, timestamp):
    print(f"Uploading metadata and images for topic: {topic_name}")
    bucket = storage_client.bucket(gcp_bucket_name)

    # Upload metadata
    metadata_gcp_path = f"fashion_ai_data/metadata/{topic_name}/{timestamp}/metadata.csv"
    blob = bucket.blob(metadata_gcp_path)
    blob.upload_from_filename(metadata_file)
    print(f"Uploaded metadata to {metadata_gcp_path}")

    # Upload images
    for image_file in os.listdir(images_path):
        local_image_path = f"{images_path}/{image_file}"
        gcp_image_path = f"fashion_ai_data/scraped_data/{topic_name}/{timestamp}/{image_file}"
        blob = bucket.blob(gcp_image_path)
        blob.upload_from_filename(local_image_path)
        print(f"Uploaded {local_image_path} to {gcp_image_path}")

    print(f"All uploads complete for topic: {topic_name}")

    # Cleanup local files
    os.remove(metadata_file)
    for image_file in os.listdir(images_path):
        os.remove(f"{images_path}/{image_file}")
    os.rmdir(images_path)
    print(f"Local files deleted for topic: {topic_name}")

# Main function to orchestrate scraping and uploading
def main():
    topics_df = pd.read_csv(topics_file)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for _, row in topics_df.iterrows():
        topic_name = row[column_topic]
        url = row[column_url]

        metadata = None
        metadata_file = None

        if scrape_data_option:
            metadata, metadata_file = scrape_metadata(topic_name, url, timestamp)
        else:
            metadata_file = f"{metadata_folder}/{topic_name}/{timestamp}/metadata.csv"
            metadata = pd.read_csv(metadata_file)

        loop = asyncio.get_event_loop()
        images_path = loop.run_until_complete(download_images(metadata, topic_name, timestamp))

        upload_to_gcp(metadata_file, images_path, topic_name, timestamp)

if __name__ == "__main__":
    try:
        main()
        print("All topics processed successfully.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
