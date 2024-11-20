import pandas as pd
from io import StringIO
import requests
from dotenv import load_dotenv
import os
import sys
import aiohttp
import asyncio
from aiohttp import ClientTimeout

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

scrape_data = os.getenv('SCRAP_IMAGES')

num_items_to_download = int(os.getenv('MAX_ITEMS'))

# Function to asynchronously download a single image without a proxy
async def download_image(session, url, image_name, bad_urls, id):
    # Check if the image already exists locally to skip downloading
    if os.path.exists(image_name):
        print(f"{image_name} already exists, skipping download.")
        return

    try:
        # Fetch the image directly without using a proxy
        async with session.get(url, timeout=ClientTimeout(total=600)) as response:
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
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    bad_urls = []  # List to store information about failed downloads

    # Create an aiohttp session with a limited connection pool
    connector = aiohttp.TCPConnector(limit_per_host=30)  # Limit to 30 concurrent connections per host
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
        df_women = pd.read_csv(os.path.join(meta_data_folder, women_file_name))
        bad_image_metadata_women = asyncio.run(download_images(df_women, os.path.join(images_folder, os.path.splitext(women_file_name)[0])))
        print("Images saved for women")
        bad_image_metadata_women.to_csv(os.path.join(meta_data_folder, bad_urls_women_file_name), index=False)

        df_men = pd.read_csv(os.path.join(meta_data_folder, men_file_name))
        bad_image_metadata_men = asyncio.run(download_images(df_men, os.path.join(images_folder, os.path.splitext(men_file_name)[0])))
        print("Images saved for men")
        bad_image_metadata_men.to_csv(os.path.join(meta_data_folder, bad_urls_men_file_name), index=False)

        print("Image files were saved")
        sys.exit(0)

    except Exception as e:
        # Catch any exception and print the error message
        print(f"Error: {e}", file=sys.stderr)  # Print the error to stderr
        # Exit with an error code 1 to indicate failure
        sys.exit(1)
