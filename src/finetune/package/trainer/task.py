import os
import json
import argparse
from google.cloud import storage
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
import wandb
from google.cloud import secretmanager
from tqdm import tqdm

## uncomment for local testing
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../../../../secrets/secret.json"


def get_secret(secret):
    client = secretmanager.SecretManagerServiceClient()
    print("-----\nFetching Key\n-----")
    response = client.access_secret_version(request={"name": secret})
    secret_value = response.payload.data.decode("UTF-8")
    print("-----\nKey Fetched\n-----")
    return secret_value

# Define dataset class
class FashionDataset(Dataset):
    def __init__(self, json_file, image_dir, transform=None):
        with open(json_file, 'r') as f:
            # Use only the first 100 data points
            self.data = json.load(f)[:100]
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        text = item['caption']
        return image, text


# Function to download images and JSON file from GCS
def download_from_gcs(gcs_json_path, gcs_image_dir, local_json_path, local_image_dir, n_samples=None):
    client = storage.Client()

    # Download JSON file if it doesn't already exist locally
    if not os.path.exists(local_json_path):
        bucket_name, json_blob_path = gcs_json_path.replace(
            "gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        json_blob = bucket.blob(json_blob_path)
        json_blob.download_to_filename(local_json_path)
        print(f"Downloaded {json_blob_path} to {local_json_path}")
    else:
        print(f"JSON file already exists at {local_json_path}, skipping download.")

    # Load JSON and filter the first n items
    with open(local_json_path, 'r') as f:
        if n_samples:
            data = json.load(f)[:n_samples]
        else:
            data = json.load(f)

    # Prepare for image downloads
    bucket_name, image_blob_prefix = gcs_image_dir.replace(
        "gs://", "").split("/", 1)
    blobs = client.list_blobs(bucket_name, prefix=image_blob_prefix)
    os.makedirs(local_image_dir, exist_ok=True)

    # Download images if they do not already exist locally
    print("Downloading images...")
    for blob in blobs:
        for item in data:
            if blob.name.endswith(item['image']):
                local_path = os.path.join(
                    local_image_dir, os.path.basename(blob.name))
                # print(f"local_path: {local_path}")
                if not os.path.exists(local_path):
                    blob.download_to_filename(local_path)
                    # print(f"Downloaded {blob.name} to {local_path}")
                else:
                    # print(f"Image {local_path} already exists, skipping download.")
                    pass



# Function to upload model weights to GCS
def upload_to_gcs(local_path, gcs_path, bucket_name="vertexai_train"):
    client = storage.Client()
    # bucket_name, gcs_folder = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    gcs_path = gcs_path[len(f"gs://{bucket_name}/"):]
    print(f"Uploading model to {gcs_path}...")

    for root, _, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            print(f"local_file_path: {local_file_path}")
            remote_path = os.path.relpath(local_file_path, local_path)
            remote_path = os.path.join(gcs_path, remote_path)
            print(f"remote_path: {remote_path}")
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to {gcs_path}/{remote_path}")

DIR_DICT = {
    "men_accessories": {
        "category": "men_accessories",
        "json_path": "gs://fashion_ai_data/captioned_data/men_accessories/2024-11-18_11-34-54/men_accessories.json",
        "image_dir": "gs://fashion_ai_data/scrapped_data/men_accessories/2024-11-18_11-34-54"
} ,
    
    "men_clothes":{
        "category": "men_clothes",
        "json_path": "gs://fashion_ai_data/captioned_data/men_clothes/2024-11-18_11-34-54/men_clothes.json",
        "image_dir": "gs://fashion_ai_data/scrapped_data/men_clothes/2024-11-18_11-34-54"
} ,
        
    "men_shoes":{   
        "category": "men_shoes",
        "json_path": "gs://fashion_ai_data/captioned_data/men_shoes/2024-11-18_11-34-54/men_shoes.json",
        "image_dir": "gs://fashion_ai_data/scrapped_data/men_shoes/2024-11-18_11-34-54"
} ,
        
    "women_accessories":{
        "category": "women_accessories",
        "json_path": "gs://fashion_ai_data/captioned_data/women_accessories/2024-11-18_11-34-54/women_accessories.json",
        "image_dir": "gs://fashion_ai_data/scrapped_data/women_accessories/2024-11-18_11-34-54"
} ,
            
    "women_clothes":{
        "category": "women_clothes",
        "json_path": "gs://fashion_ai_data/captioned_data/women_clothes/2024-11-18_11-34-54/women_clothes.json",
        "image_dir": "gs://fashion_ai_data/scrapped_data/women_clothes/2024-11-18_11-34-54"
} ,
                    
    "women_shoes":{
        "category": "women_shoes",
        "json_path": "gs://fashion_ai_data/captioned_data/women_shoes/2024-11-18_11-34-54/women_shoes.json",
        "image_dir": "gs://fashion_ai_data/scrapped_data/women_shoes/2024-11-18_11-34-54"
}
}


# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="men_accessories", help="Category of the dataset.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="GCS path to save the fine-tuned model.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float,
                        default=5e-6, help="Learning rate.")
    parser.add_argument("--wandb_key", dest="wandb_key",
                        default="projects/1087474666309/secrets/WandbAPI/versions/latest", type=str, help="WandB API Key Secret Name")
    parser.add_argument("--model_name", type=str, default="patrickjohncyh/fashion-clip",
                        help="Name of the model to load.")
    parser.add_argument("--bucket_name", type=str, default="vertexai_train")
    parser.add_argument("--n_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()



    json_path = DIR_DICT[args.category]["json_path"]
    image_dir = DIR_DICT[args.category]["image_dir"]

    # Local paths for data
    local_json_path = json_path.replace("gs://", "")
    local_image_dir = image_dir.replace("gs://", "")
    local_model_dir = "local_fine_tuned_model"

    # Create dedicated subfolder for the model in the output directory
    os.makedirs("/".join(local_json_path.split("/")[:-1]), exist_ok=True)
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_model_dir, exist_ok=True)

    # Download data from GCS
    download_from_gcs(json_path, image_dir,
                      local_json_path, local_image_dir, args.n_samples)

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    # wandb.login(key=get_secret(args.wandb_key))
    # wandb.init(project=f"fashionclip_{args.category}", config=args)

    dataset = FashionDataset(json_file=local_json_path,
                             image_dir=local_image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model and processor
    processor = CLIPProcessor.from_pretrained(args.model_name)
    processor.image_processor.do_rescale = False  # Avoid double rescaling
    model = CLIPModel.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(
            dataloader), desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (images, texts) in progress_bar:
            images = images.to(device)
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            loss_image = torch.nn.functional.cross_entropy(
                logits_per_image, torch.arange(len(images), device=device))
            loss_text = torch.nn.functional.cross_entropy(
                logits_per_text, torch.arange(len(texts), device=device))
            loss = (loss_image + loss_text) / 2
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
        #     wandb.log(
        #         {"epoch": epoch, "batch_idx": batch_idx, "loss": loss.item()})
        # wandb.log({"epoch": epoch, "average_loss": total_loss / len(dataloader)})

    # Save and upload the model
    model.save_pretrained(local_model_dir)
    processor.save_pretrained(local_model_dir)
    upload_to_gcs(local_model_dir, args.output_dir, args.bucket_name)

    print("Training completed and model uploaded successfully.")


if __name__ == "__main__":
    main()
