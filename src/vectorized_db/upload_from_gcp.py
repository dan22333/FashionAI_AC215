from pinecone import Pinecone, ServerlessSpec
from google.cloud import storage
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os
import argparse
from helper_functions import get_clip_vector
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
response = client.access_secret_version(request={"name":"projects/1087474666309/secrets/pincone/versions/latest"})
secret_value = response.payload.data.decode("UTF-8")

pc = Pinecone(api_key=secret_value)

# Define the index name and vector dimension
index_name = "clip-vector-index"
vector_dim = 512  # CLIP's output vector dimension

# Check if the index exists, create if not
if index_name not in [item["name"] for item in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=vector_dim,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Set up argument parsing
parser = argparse.ArgumentParser(description="Upload photos from a specified GCP bucket.")
parser.add_argument("bucket_name", type=str, help="Your GCP bucket name")

# Parse the arguments
args = parser.parse_args()
bucket_name = args.bucket_name

# Initialize the GCP storage client
storage_client = storage.Client()

# Get the GCP bucket
bucket = storage_client.bucket(bucket_name)

# List all images in the bucket and upload to Pinecone
for blob in tqdm(bucket.list_blobs()):
    if blob.name.lower().endswith((".png", ".jpg", ".jpeg")):
        # Download image content as bytes
        image_data = blob.download_as_bytes()
        image = Image.open(BytesIO(image_data)).convert("RGB")  # Ensure images are in RGB format
        vector = get_clip_vector(image, is_image=True)
        
        # Construct the public URL
        image_url = f"https://storage.googleapis.com/{bucket_name}/{blob.name}"
        
        # Add the vector and URL as metadata
        index.upsert([(blob.name, vector, {"url": image_url})])  # Use blob name as ID and URL as metadata

print("Index populated with image embeddings and URLs.")