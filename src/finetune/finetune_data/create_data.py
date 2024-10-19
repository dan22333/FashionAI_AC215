import os
import json

# Directory where your images are stored
image_dir = "images"

# A sample dictionary mapping image names to captions (replace with your actual captions)
image_captions = {
    "img0.jpg": "This is the best blue shirt ever for wedding!",
    "img1.jpg": "This is the best Prada dress ever for wedding!",
}

for i in range(2, 100):
    caption = image_captions[f"img{i % 2}.jpg"]
    image_captions[f"img{i}.jpg"] = caption

# Create the dataset list
dataset = [{"image": os.path.join(image_dir, img), "caption": caption} for img, caption in image_captions.items()]

# Save as JSON
with open("fashion_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)
