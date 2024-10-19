import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
# from fashion_clip import FashionCLIPModel
import os

# Load the fine-tuned FashionCLIP model and processor
model = CLIPModel.from_pretrained("/home/wel019/AC215_FashionAI/src/finetune/fine_tuned_fashionclip")
processor = CLIPProcessor.from_pretrained("/home/wel019/AC215_FashionAI/src/finetune/fine_tuned_fashionclip_processor")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode
model.eval()

# Function to preprocess and load images
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

# Function to ask a question (text) and compare it with an image
def ask_question(text, image_paths):
    # Preprocess the images
    images = [load_and_preprocess_image(image_path) for image_path in image_paths]
    
    # Preprocess inputs (text and images)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True).to(device)

    # Forward pass: get the similarity logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # Similarity score of text vs each image

    # Convert logits to probabilities (softmax over image logits)
    probs = logits_per_image.softmax(dim=1)

    # Find the most relevant image (best match)
    best_image_idx = probs.argmax(dim=0).item()  # Take argmax across all images
    best_image_path = image_paths[best_image_idx]

    return best_image_path, probs


# Example usage
if __name__ == "__main__":
    # Example question and list of images
    question = "What is the best prada dress for wedding?"
    path_pre = "/home/wel019/AC215_FashionAI/src/finetune/finetune_data/images"
    image_paths = [f"{path_pre}/{p}" for p in os.listdir("/home/wel019/AC215_FashionAI/src/finetune/finetune_data/images")]
    # print(image_paths)

    best_image, probabilities = ask_question(question, image_paths)

    print(f"Best matching image: {best_image}")
    print(f"Probabilities: {probabilities}")
