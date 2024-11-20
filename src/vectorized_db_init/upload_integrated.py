import os
import pandas as pd
from google.cloud import storage, secretmanager
from pinecone import Pinecone, ServerlessSpec
from PIL import Image
from io import BytesIO
import json
from tqdm import tqdm
from helper_functions import get_clip_vector

# Initialize storage client globally
storage_client = storage.Client("fashion-ai")

# Retrieve Pinecone API key from Google Secret Manager
def get_pinecone_api_key(secret_name):
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(request={"name": secret_name})
    return response.payload.data.decode("UTF-8")

# Initialize Pinecone
def initialize_pinecone(index_name, vector_dim, api_key):
    pc = Pinecone(api_key=api_key)
    # Check if the index exists, create it if not
    if index_name not in [item["name"] for item in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=vector_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

# Load JSON data from a GCP bucket
def load_json_from_bucket(bucket_name, blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return json.loads(blob.download_as_text())

# Load text from a GCP bucket
def load_text_from_bucket(bucket_name, blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_text()

# Parse metadata CSV into pandas DataFrame
def parse_metadata(metadata_text):
    from io import StringIO
    return pd.read_csv(StringIO(metadata_text))

# List subfolders in a GCP bucket path
def list_subfolders(bucket_name, prefix):
    # Debug: Print the bucket and prefix being used
    print(f"Bucket: {bucket_name}, Prefix: {prefix}")

    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    # Debug: Check if any prefixes are returned
    if blobs.prefixes:
        print("Found prefixes:")
        for folder in blobs.prefixes:
            print(folder)
        return [folder for folder in blobs.prefixes]
    else:
        print("No prefixes found. Listing all blobs:")
        for blob in blobs:
            print(blob.name)
        return []

# Process and upload data for a specific topic
def process_and_upload_topic(topic, base_bucket, pinecone_index):
    caption_bucket = base_bucket
    caption_path = f"captioned_data/{topic}/2024-11-18_11-34-54"
    metadata_bucket = base_bucket
    metadata_path = f"metadata/{topic}/2024-11-18_11-34-54"
    image_bucket = f"{base_bucket}/scrapped_data/{topic}"

    # Load caption data
    caption_data = load_json_from_bucket(caption_bucket, caption_path)

    # Load and parse metadata using pandas
    metadata_text = load_text_from_bucket(metadata_bucket, metadata_path)
    metadata_df = parse_metadata(metadata_text)

    # Access the image bucket
    image_bucket_obj = storage_client.bucket(base_bucket)

    # Keep track of uploaded items
    uploaded_items = []

    for caption_entry in tqdm(caption_data):
        image_name = caption_entry["image"]
        caption = caption_entry["caption"]

        # Find metadata entry matching the image
        metadata_entry = metadata_df.loc[metadata_df["source/id"] == image_name]
        if metadata_entry.empty:
            print(f"No metadata found for image: {image_name}")
            continue

        # Extract metadata fields
        brand = metadata_entry.get("brand", "")
        categories = [
            metadata_entry.get(f"categories/{i}", "")
            for i in range(10)
        ]
        image_url = metadata_entry.get("medias/0/url", "")

        # Generate vector for the image
        image_blob = image_bucket_obj.blob(f"scrapped_data/{topic}/{image_name}")
        if not image_blob.exists():
            print(f"Image not found in bucket: {image_name}")
            continue

        # Download and process image
        image_data = image_blob.download_as_bytes()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        vector = get_clip_vector(image, is_image=True)

        # Prepare metadata for Pinecone
        pinecone_metadata = {
            "image_name": image_name,
            "brand": brand,
            "categories": categories,
            "image_url": image_url,
            "caption": caption
        }

        # Upload to Pinecone
        pinecone_index.upsert([
            {
                "id": image_name,
                "values": vector.tolist(),
                "metadata": pinecone_metadata
            }
        ])
        uploaded_items.append(image_name)

    print(f"Uploaded {len(uploaded_items)} items for topic: {topic}")

# Main execution
if __name__ == "__main__":
    # Configuration
    SECRET_NAME = "projects/1087474666309/secrets/pincone/versions/latest"
    INDEX_NAME = "clip-vector-index-test"
    VECTOR_DIM = 512
    BASE_BUCKET = "fashion_ai_data"
    SCRAPPED_DATA_PREFIX = "scrapped_data"

    # Initialize Pinecone
    pinecone_api_key = get_pinecone_api_key(SECRET_NAME)
    pinecone_index = initialize_pinecone(INDEX_NAME, VECTOR_DIM, pinecone_api_key)

    # List all topics (subfolders) in the scrapped_data folder
    topics = list_subfolders(BASE_BUCKET, SCRAPPED_DATA_PREFIX)

    # Process and upload data for each topic
    for topic in topics:
        print(f"Processing topic: {topic}")
        process_and_upload_topic(topic, BASE_BUCKET, pinecone_index)
