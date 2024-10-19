# import os
# import json
# import torch
# import wandb
# import argparse
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# from transformers import AdamW, CLIPProcessor, CLIPModel
# # from fashion_clip import FashionCLIPModel

# # Function to parse arguments
# def parse_args():
#     parser = argparse.ArgumentParser(description="Fine-tune FashionCLIP on a custom dataset")
#     parser.add_argument('--json_file', type=str, default='/home/wel019/AC215_FashionAI/src/finetune/finetune_data/final_output.json', help="Path to the dataset JSON file")
#     parser.add_argument('--image_dir', type=str, default='/home/wel019/AC215_FashionAI/src/finetune/finetune_data/AC215Images4', help="Path to the directory containing images")
#     parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
#     parser.add_argument('--learning_rate', type=float, default=5e-6, help="Learning rate for optimizer")
#     parser.add_argument('--epochs', type=int, default=3, help="Number of epochs to train for")
#     parser.add_argument('--project', type=str, default="fashion-clip-finetuning_1500", help="Wandb project name")
#     parser.add_argument('--save_model_name', type=str, default="fine_tuned_fashionclip_1500", help="Directory to save the fine-tuned model")
#     parser.add_argument('--save_processor_name', type=str, default="fine_tuned_fashionclip_processor_1500", help="Directory to save the fine-tuned processor")
#     return parser.parse_args()

# # Main function
# def main():
#     args = parse_args()

#     # Initialize wandb for tracking experiments
#     wandb.init(project=args.project, config={
#         "batch_size": args.batch_size,
#         "learning_rate": args.learning_rate,
#         "epochs": args.epochs
#     })

#     # Custom Dataset for FashionCLIP
#     class FashionDataset(Dataset):
#         def __init__(self, json_file, image_dir, transform=None):
#             with open(json_file, 'r') as f:
#                 self.data = json.load(f)
#             self.image_dir = image_dir
#             self.transform = transform

#         def __len__(self):
#             return len(self.data)

#         def __getitem__(self, idx):
#             item = self.data[idx]
#             image_path = os.path.join(self.image_dir, item['image'])  # Assuming 'image' is a key in the JSON file
#             image = Image.open(image_path).convert('RGB')
            
#             if self.transform:
#                 image = self.transform(image)

#             text = item['caption']  # Assuming 'caption' is a key in the JSON file for the textual label
#             return image, text

#     # Define image transformations
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize the image as required
#         transforms.ToTensor(),  # Convert the image to tensor but without normalization
#     ])

#     # Load dataset and create a DataLoader
#     train_dataset = FashionDataset(json_file=args.json_file, image_dir=args.image_dir, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)

#     # Load pre-trained FashionCLIP model and processor
#     model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
#     processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

#     # Move model to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Define optimizer for fine-tuning
#     optimizer = AdamW(model.parameters(), lr=wandb.config.learning_rate)

#     # Training loop with wandb integration
#     for epoch in range(wandb.config.epochs):
#         model.train()
#         total_loss = 0
        
#         for batch_idx, (images, texts) in enumerate(train_loader):
#             images = images.to(device)
            
#             # Preprocess inputs for FashionCLIP
#             inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77, do_rescale=False).to(device)

            
#             # Forward pass
#             outputs = model(**inputs)
#             logits_per_image = outputs.logits_per_image
#             logits_per_text = outputs.logits_per_text
            
#             # FashionCLIP's loss function: Cross-entropy between image and text logits
#             loss_image = torch.nn.functional.cross_entropy(logits_per_image, torch.arange(len(images), device=device))
#             loss_text = torch.nn.functional.cross_entropy(logits_per_text, torch.arange(len(texts), device=device))
            
#             # Combined loss
#             loss = (loss_image + loss_text) / 2
#             total_loss += loss.item()

#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # Log metrics to wandb every 10 batches
#             if batch_idx % 10 == 0:
#                 wandb.log({"epoch": epoch, "batch_idx": batch_idx, "loss": loss.item()})
#                 print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

#         # Log epoch-level metrics
#         average_loss = total_loss / len(train_loader)
#         wandb.log({"epoch": epoch, "avg_loss": average_loss})
#         print(f"Epoch {epoch}, Average Loss: {average_loss}")

#     # Save fine-tuned FashionCLIP model
#     model.save_pretrained(args.save_model_name)
#     processor.save_pretrained(args.save_processor_name)

#     # End the wandb run
#     wandb.finish()

# if __name__ == "__main__":
#     main()



import os
import json
import torch
import wandb
import argparse
from tqdm import tqdm  # Import tqdm for progress bar
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AdamW, CLIPProcessor, CLIPModel

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune FashionCLIP on a custom dataset")
    parser.add_argument('--json_file', type=str, default='/home/wel019/AC215_FashionAI/src/finetune/finetune_data/final_output.json', help="Path to the dataset JSON file")
    parser.add_argument('--image_dir', type=str, default='/home/wel019/AC215_FashionAI/src/finetune/finetune_data/AC215Images4', help="Path to the directory containing images")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=5e-6, help="Learning rate for optimizer")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs to train for")
    parser.add_argument('--project', type=str, default="fashion-clip-finetuning_1500", help="Wandb project name")
    return parser.parse_args()

# Define the sweep configuration
sweep_config = {
    'method': 'grid',  # You can use 'random' or 'bayes' for random or Bayesian search
    'parameters': {
        'batch_size': {'values': [16, 32, 64]},  # Different batch sizes to try
        'learning_rate': {'values': [5e-5, 5e-6, 1e-6]},  # Learning rates to sweep
        'epochs': {'values': [3, 5]}  # Number of epochs to try
    }
}

def sweep_train():
    args = parse_args()
    
    # Initialize wandb for tracking experiments
    wandb.init(config=args)  # Now sweep config will override the defaults
    config = wandb.config

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
            image_path = os.path.join(self.image_dir, item['image'])
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)

            text = item['caption']
            return image, text

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset and create DataLoader
    train_dataset = FashionDataset(json_file=args.json_file, image_dir=args.image_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Load pre-trained FashionCLIP model and processor
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Define optimizer for fine-tuning
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop with progress bar
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        # Initialize the progress bar for the current epoch
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch_idx, (images, texts) in progress_bar:
            images = images.to(device)
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77, do_rescale=False).to(device)

            # Forward pass
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            
            # Compute loss
            loss_image = torch.nn.functional.cross_entropy(logits_per_image, torch.arange(len(images), device=device))
            loss_text = torch.nn.functional.cross_entropy(logits_per_text, torch.arange(len(texts), device=device))
            loss = (loss_image + loss_text) / 2
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar description with loss
            progress_bar.set_postfix(loss=loss.item())

            # Log loss every 10 batches
            if batch_idx % 10 == 0:
                wandb.log({"epoch": epoch, "batch_idx": batch_idx, "loss": loss.item()})

        average_loss = total_loss / len(train_loader)
        wandb.log({"epoch": epoch, "avg_loss": average_loss})

    # Save the fine-tuned FashionCLIP model and processor with hyperparameters in the filename
    model_save_name = f"fine_tuned_fashionclip_bs{config.batch_size}_lr{config.learning_rate}_ep{config.epochs}"
    model.save_pretrained(model_save_name)
    processor.save_pretrained(f"{model_save_name}_processor")

    wandb.finish()

if __name__ == "__main__":
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="fashion-clip-finetuning")
    
    # Run the sweep agent
    wandb.agent(sweep_id, function=sweep_train)

