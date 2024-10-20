import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Search for similar images based on a text query")
    parser.add_argument('--question', type=str, default="A jacket for causal outings.", help="Text query to search for similar images")
    parser.add_argument('--image_dir', type=str, default="finetune_data/AC215Images4", help="Path to the directory containing images")
    parser.add_argument('--model_dir', type=str, default="models/fine_tuned_fashionclip_bs32_lr5e-06_ep3", help="Path to the directory containing the fine-tuned model")
    parser.add_argument('--processor_dir', type=str, default="models/fine_tuned_fashionclip_bs32_lr5e-06_ep3_processor", help="Path to the directory containing the processor")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for processing images")
    return parser.parse_args()

# Define a custom Dataset for the images
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)  # Apply transformation to convert PIL image to tensor
        return image, image_path

def load_model(model_dir, processor_dir):
    model = CLIPModel.from_pretrained(model_dir)
    processor = CLIPProcessor.from_pretrained(processor_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, processor, device

def search_similar_images(text, image_loader, model, processor, device):
    all_probs = []
    all_image_paths = []
    
    for images, image_paths in tqdm(image_loader):
        # Preprocess inputs (text and images)
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True, do_rescale=False).to(device)

        # Forward pass: get the similarity logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # Image-to-text similarity

        # Convert logits to probabilities (softmax over image logits)
        probs = logits_per_image.softmax(dim=0)  # Apply softmax over the last dimension

        # Store results for this batch
        all_probs.append(probs)
        all_image_paths.extend(image_paths)
    
    # Concatenate probabilities across all batches
    all_probs = torch.cat(all_probs, dim=0)
    
    # Find the most relevant image (best match)
    best_image_idx = all_probs.argmax(dim=0).item()  # Get the index of the best match
    
    best_image_path = all_image_paths[best_image_idx]
    return best_image_path, all_probs

if __name__ == "__main__":
    # Parse arguments
    arguments = parse_args()
    question = arguments.question
    img_path = arguments.image_dir
    model_dir = arguments.model_dir
    processor_dir = arguments.processor_dir
    batch_size = arguments.batch_size
    
    # Load the image paths
    image_paths = [f"{img_path}/{p}" for p in os.listdir(img_path) if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Create the image dataset and dataloader
    image_dataset = ImageDataset(image_paths)
    image_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)

    print("Loading model and processor...\n\n")
    model, processor, device = load_model(model_dir, processor_dir)
    print("Model loaded successfully!\n\n")
    
    # Search for the best matching image
    best_image, similarities = search_similar_images(question, image_loader, model, processor, device)
    
    print(f"Question: {question}")
    print(f"Best matching image: {best_image}")
    # print(f"Similarities: {similarities}")
