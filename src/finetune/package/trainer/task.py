import os
import argparse
from transformers import CLIPModel, CLIPProcessor

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-dir", dest="model_dir", default="test", type=str, help="Model dir."
)
parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument(
    "--model_name",
    dest="model_name",
    default="fashionclip-tuned",
    type=str,
    help="Model name",
)
parser.add_argument(
    "--train_base",
    dest="train_base",
    default=False,
    action="store_true",
    help="Train base or not",
)
parser.add_argument(
    "--epochs", dest="epochs", default=10, type=int, help="Number of epochs."
)
parser.add_argument(
    "--batch_size", dest="batch_size", default=16, type=int, help="Size of a batch."
)
parser.add_argument(
    "--wandb_key", dest="wandb_key", default="16", type=str, help="WandB API Key"
)
args = parser.parse_args()

print("hello world")
print("Working directory contents:", os.listdir())

# Load the fashion-clip model and processor
print("Loading Fashion-CLIP model and processor...")
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
print("Model and processor loaded successfully.")



from google.cloud import storage

# Save the model locally first
local_output_path = "fashion-clip-tuned"
os.makedirs(local_output_path, exist_ok=True)

print(f"Saving model locally to {local_output_path}...")
model.save_pretrained(local_output_path)
processor.save_pretrained(local_output_path)
print(f"Model saved locally to {local_output_path}.")

# Upload to GCS
def upload_to_gcs(local_path, gcs_path):
    client = storage.Client()
    bucket_name, gcs_folder = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            remote_path = os.path.join(gcs_folder, os.path.relpath(local_file_path, local_path))
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to {gcs_path}/{remote_path}")

print(f"Uploading model to GCS path: {args.model_dir}...")
upload_to_gcs(local_output_path, args.model_dir)
print(f"Model successfully uploaded to {args.model_dir}.")

