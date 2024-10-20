
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
# Load the .env file
load_dotenv()

meta_data_folder = os.getenv('SCRAPED_METADATA')
images_folder = os.getenv('SCRAPED_RAW_IMAGES')

men_file_name = os.getenv('MEN_FILE_NAME')
women_file_name = os.getenv('WOMEN_FILE_NAME')

id_col_name = os.getenv('COLUMN_ID_NAME')
image_url_col = os.getenv('URL_IMAGE')
bad_urls_men_file_name = os.getenv('BAD_URLS_MEN')
bad_urls_women_file_name = os.getenv('BAD_URLS_WOMEN')

# Initialize the ApifyClient with your API token
client = secretmanager.SecretManagerServiceClient()
response = client.access_secret_version(request={"name": os.getenv('APIFY_GCP_SECRET_ACCESS')})
secret_value = response.payload.data.decode("UTF-8")
client = ApifyClient(secret_value)

APIFY_PROXY_URL = f"http://auto:{secret_value}@proxy.apify.com:8000?groups=residential"

def get_items_seed(url):
    # Prepare the Actor input for each page
    run_input = {
        "startUrls": [{"url": url}],
        "maxRequestsPerCrawl": num_of_items/2,
        "proxy": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"],  # This specifies using the residential proxy group
        },
        "maxConcurrency": 10,
    }

    # Run the Actor and wait for it to finish
    run = client.actor("mKTnbkisJ8BAiIbsP").call(run_input=run_input)

    dataset_id = run["defaultDatasetId"]

    # Fetch and print Actor results from the run's dataset (if there are any)
    # for item in client.dataset(run["defaultDatasetId"]).iterate_items():
    #     print(item)
    # Fetch Actor results from the run's dataset
    # all_items.append(item)
    # API URL to get the dataset in CSV format
    api_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?format=csv"

    # Fetch the CSV file
    response = requests.get(api_url)


    # Decode the byte string to UTF-8 and remove BOM
    decoded_data = response.content.decode('utf-8-sig')  # The 'utf-8-sig' will handle the BOM

    # Use StringIO to treat the decoded string as a file-like object
    csv_data = StringIO(decoded_data)
    df = pd.read_csv(csv_data)
    return df

# Function to asynchronously download a single image using Apify proxy
async def download_image(session, url, image_name, bad_urls, id):
    try:
        # Fetch the image using Apify's proxy service
        async with session.get(url) as response:
            if response.status == 200:
                # Save the image
                with open(image_name, 'wb') as f:
                    f.write(await response.read())
                print(f"Photo successfully downloaded as {image_name}")
            else:
                # Log the failed download
                print(f"Failed to download {image_name}. Status code: {response.status}")
                bad_urls.append({'url': url, 'id': id, 'error': f'Failed with status code {response.status}'})
    except Exception as e:
        # Log any exceptions
        print(f"Error downloading {url}: {e}")
        bad_urls.append({'url': url, 'id': id, 'error': str(e)})

# Function to download multiple images asynchronously and return a DataFrame of failed downloads
async def download_images(urls_df, output_folder):
    os.makedirs(output_folder, exist_ok=True)# Ensure the output folder exists
    bad_urls = []  # List to store information about failed downloads

    connector = aiohttp.TCPConnector(limit_per_host=30)  # Limit to 10 concurrent connections per host

    # Create an aiohttp session with a limited connection pool
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, row in urls_df.iterrows():
            url = row.get(image_url_col)
            if url:
                # Construct the image name based on the id column
                image_name = os.path.join(output_folder, f"image_{row.get(id_col_name)}.jpg")
                # Schedule the download task
                tasks.append(download_image(session, url, image_name, bad_urls, row.get(id_col_name)))
            else:
                print(f"URL missing in row {i + 1}")
                bad_urls.append({'url': 'Missing', 'id': row.get(id_col_name), 'error': 'No URL provided'})

        # Await the completion of all tasks
        await asyncio.gather(*tasks)

    # Convert bad_urls list to a Pandas DataFrame and return it
    if bad_urls:
        bad_urls_df = pd.DataFrame(bad_urls)
        print(f"Bad URLs collected: {len(bad_urls_df)}")
        return bad_urls_df
    else:
        return pd.DataFrame(columns=['url', 'id', 'error'])  # Return an empty DataFrame if no errors

if __name__ == '__main__':
    try:
        sys.exit(0)
        # # Base URL for pagination
        # base_url_women = "https://www.farfetch.com/shopping/women/clothing-1/items.aspx?page=1"
        # base_url_men = "https://www.farfetch.com/shopping/men/clothing-2/items.aspx?page=1"
        #
        # # Set number of pages to scrape (can be dynamically determined later)
        # num_of_items = 2
        #
        # df_women = get_items_seed(base_url_women)
        # df_men  = get_items_seed(base_url_men)
        # 
        # print("Metadata files were downloaded")

        # Save the DataFrame to the full path

        #df_women.to_csv(os.path.join(meta_data_folder, women_file_name), index=False)
       # df_men.to_csv(os.path.join(meta_data_folder, men_file_name), index=False)

        print("Metadata files were saved")
        df_women = pd.read_csv(os.path.join(meta_data_folder, women_file_name))
        bad_image_metadata_women = asyncio.run(download_images(df_women, os.path.join(images_folder, os.path.splitext(women_file_name)[0])))
        bad_image_metadata_women.to_csv(os.path.join(meta_data_folder, bad_urls_women_file_name),  index=False)

        bad_image_metadata_men = asyncio.run(download_images(df_men, os.path.join(images_folder, os.path.splitext(men_file_name)[0])))
        bad_image_metadata_men.to_csv(os.path.join(meta_data_folder, bad_urls_men_file_name), index=False)

        print("Images files were saved")
        sys.exit(0)

    except Exception as e:
        # Catch any exception and print the error message
        print(f"Error: {e}", file=sys.stderr)  # Print the error to stderr
        # Exit with an error code 1 to indicate failure
        sys.exit(1)

