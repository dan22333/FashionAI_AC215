# import os
# import glob
# from google.cloud import storage
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm  # For a progress bar
# from dotenv import load_dotenv
#
# load_dotenv()
#
# # Configuration for GCP
# gcp_project = "fashion-ai"
# bucket_name = "fashion_ai_data"
# photos_folder = os.path.join("../../data/scraped_raw_images/", "men")  # Path to local photos folder
# upload_bucket = "scrapped_data/men_clothes/"  # GCS folder
#
# def get_existing_files():
#     """Retrieve a list of files already uploaded to GCS."""
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blobs = bucket.list_blobs(prefix=upload_bucket)  # List all files in the upload folder
#     return {os.path.basename(blob.name) for blob in blobs}  # Return filenames as a set
#
# def upload_single_photo(photo_file):
#     """Uploads a single photo to GCS."""
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#
#     filename = os.path.basename(photo_file)
#     destination_blob_name = os.path.join(upload_bucket, filename)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(photo_file)
#     return filename
#
# def upload_photos_to_gcs():
#     print("Uploading photos to GCS")
#     uploaded_files = []  # Track successfully uploaded files
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#
#     # Get files already uploaded
#     existing_files = get_existing_files()
#     photo_files = glob.glob(os.path.join(photos_folder, "*.[jp][pn]g"))
#     files_to_upload = [file for file in photo_files if os.path.basename(file) not in existing_files]
#     print(f"Files to upload: {len(files_to_upload)} / {len(photo_files)}")
#
#     try:
#         with ThreadPoolExecutor(max_workers=10) as executor:
#             for photo_file in tqdm(files_to_upload, total=len(files_to_upload)):
#                 filename = upload_single_photo(photo_file)
#                 uploaded_files.append(filename)  # Track uploaded file
#
#         print("All photos uploaded successfully.")
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         print("Rolling back uploaded files...")
#         # Rollback: Delete all files that were uploaded in this session
#         for filename in uploaded_files:
#             blob = bucket.blob(os.path.join(upload_bucket, filename))
#             blob.delete()
#             print(f"Deleted: {filename}")
#
# if __name__ == "__main__":
#     upload_photos_to_gcs()



import os
import glob
import asyncio
from google.cloud import storage
from tqdm.asyncio import tqdm as async_tqdm  # Async version of tqdm
from dotenv import load_dotenv

load_dotenv()

# Configuration for GCP
gcp_project = "fashion-ai"
bucket_name = "fashion_ai_data"
photos_folder = os.path.join("../../data/scraped_raw_images/", "men")  # Path to local photos folder
upload_bucket = "scrapped_data/men_clothes/"  # GCS folder

async def get_existing_files():
    """Retrieve a list of files already uploaded to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=upload_bucket)  # List all files in the upload folder
    return {os.path.basename(blob.name) for blob in blobs}  # Return filenames as a set

async def upload_single_photo(photo_file):
    """Uploads a single photo to GCS asynchronously."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    filename = os.path.basename(photo_file)
    destination_blob_name = os.path.join(upload_bucket, filename)
    blob = bucket.blob(destination_blob_name)

    # Use asyncio.to_thread to run the blocking operation in a thread
    await asyncio.to_thread(blob.upload_from_filename, photo_file)
    return filename

async def upload_photos_to_gcs():
    """Uploads all photos in the folder to GCS asynchronously."""
    print("Uploading photos to GCS")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Get files already uploaded
    existing_files = await get_existing_files()
    photo_files = glob.glob(os.path.join(photos_folder, "*.[jp][pn]g"))
    files_to_upload = [file for file in photo_files if os.path.basename(file) not in existing_files]
    print(f"Files to upload: {len(files_to_upload)} / {len(photo_files)}")

    try:
        # Asynchronous processing with a progress bar
        await async_tqdm.gather(*(upload_single_photo(photo) for photo in files_to_upload))
        print("All photos uploaded successfully.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(upload_photos_to_gcs())
