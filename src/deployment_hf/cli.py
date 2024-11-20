"""
Module to deploy a Hugging Face model (FashionCLIP) on Vertex AI.

Typical usage example from the command line:
    python cli.py --prepare
    python cli.py --deploy
"""

import os
import base64
import argparse
from glob import glob
import numpy as np
from google.cloud import storage
from google.cloud import aiplatform
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import login, HfApi, delete_file, upload_folder, list_repo_files

# Environment variables
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_MODELS_BUCKET_NAME = os.environ["GCS_MODELS_BUCKET_NAME"]
MODEL_PATH = "finetuned-fashionclip"
ARTIFACT_URI = f"gs://{GCS_MODELS_BUCKET_NAME}"
LOCAL_ARTIFACT_PATH = "./artifacts"

def prepare():
    """
    Downloads all files from a specific folder in a GCP bucket
    and saves them locally, preserving the folder structure.
    """
    storage_client = storage.Client(project=GCP_PROJECT)
    bucket = storage_client.bucket(GCS_MODELS_BUCKET_NAME)

    # List all blobs in the specified folder
    blobs = bucket.list_blobs(prefix=MODEL_PATH)

    for blob in blobs:
        # Remove the folder prefix to get the relative path
        relative_path = blob.name[len(MODEL_PATH) + 1:]

        # Skip folder paths
        if not relative_path:
            continue

        # Define the local file path
        local_file_path = os.path.join(LOCAL_ARTIFACT_PATH, relative_path)

        # Ensure the local directory exists
        local_dir = os.path.dirname(local_file_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # Download the file
        print(f"Downloading {blob.name} to {local_file_path}...")
        blob.download_to_filename(local_file_path)

    print(f"All files from {MODEL_PATH} have been downloaded to {LOCAL_ARTIFACT_PATH}")

def deploy():

    hf_token = os.environ["HUGGINGFACE_KEY"]

    login(token=hf_token)

    # Specify model repository name
    repo_name = os.environ["HF_REPO_NAME"]

    # Initialize API
    api = HfApi()

    # List current files in the repository
    repo_files = list_repo_files(repo_id=repo_name, token=hf_token)

    # Files to preserve
    files_to_preserve = [".gitattributes", "README.md"]

    # Delete files not in the preserve list
    for file_path in repo_files:
        if file_path not in files_to_preserve:
            delete_file(path_in_repo=file_path, repo_id=repo_name, token=hf_token)
            print(f"Deleted: {file_path}")

    # Upload all files from the local folder
    upload_folder(
        folder_path=LOCAL_ARTIFACT_PATH,
        repo_id=repo_name,
        token=hf_token,
        commit_message="Updated upload after cleanup"
    )

    print("Upload complete.")



def main(args=None):
    if args.prepare:
        print("Preparing model...")
        prepare()

    elif args.deploy:
        print("Deploying model...")
        deploy()
    else:
        print("Preparing model...")
        prepare()
        print("Deploying model...")
        deploy()


    # elif args.predict:
    #     print("Predicting using deployed endpoint...")
    #     predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FashionCLIP Deployment CLI")

    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare the FashionCLIP model and processor for Vertex AI.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy the FashionCLIP model to Vertex AI.",
    )
    # parser.add_argument(
    #     "--predict",
    #     action="store_true",
    #     help="Make predictions using the deployed endpoint.",
    # )

    args = parser.parse_args()
    main(args)
