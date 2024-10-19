import os
import json
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AdamW
from fashion_clip import FashionCLIPProcessor, FashionCLIPModel  # Assuming the correct imports for FashionCLIP

# Initialize wandb for tracking experiments
wandb.init(project="fashion-clip-finetuning", config={
    "batch_size": 32,
    "learning_rate": 5e-6,
    "epochs": 3
})

# Custom Dataset for FashionCLIP
class FashionDataset(Dataset):
    def __init__(self, json_file, image_dir, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item['image'])  # Assuming 'image' is a key in the JSON file
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        text = item['caption']  # Assuming 'caption' is a key in the JSON file for the textual label
        return image, text

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image for FashionCLIP
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))  # FashionCLIP normalization
])

# Load dataset and create a DataLoader
train_dataset = FashionDataset(json_file='/home/wel019/AC215_FashionAI/src/finetune/finetune_data/fashion_dataset.json', image_dir='/home/wel019/AC215_FashionAI/src/finetune/finetune_data/images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)

# Load pre-trained FashionCLIP model and processor
model = FashionCLIPModel.from_pretrained("fashion-clip/fashion-clip-base")
processor = FashionCLIPProcessor.from_pretrained("fashion-clip/fashion-clip-base")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer for fine-tuning
optimizer = AdamW(model.parameters(), lr=wandb.config.learning_rate)

# Training loop with wandb integration
for epoch in range(wandb.config.epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, texts) in enumerate(train_loader):
        images = images.to(device)
        
        # Preprocess inputs for FashionCLIP
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        
        # Forward pass
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        
        # FashionCLIP's loss function: Cross-entropy between image and text logits
        loss_image = torch.nn.functional.cross_entropy(logits_per_image, torch.arange(len(images), device=device))
        loss_text = torch.nn.functional.cross_entropy(logits_per_text, torch.arange(len(texts), device=device))
        
        # Combined loss
        loss = (loss_image + loss_text) / 2
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics to wandb every 10 batches
        if batch_idx % 10 == 0:
            wandb.log({"epoch": epoch, "batch_idx": batch_idx, "loss": loss.item()})
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    # Log epoch-level metrics
    average_loss = total_loss / len(train_loader)
    wandb.log({"epoch": epoch, "avg_loss": average_loss})
    print(f"Epoch {epoch}, Average Loss: {average_loss}")

# Save fine-tuned FashionCLIP model
model.save_pretrained("fine_tuned_fashionclip")
processor.save_pretrained("fine_tuned_fashionclip_processor")

# End the wandb run
wandb.finish()
