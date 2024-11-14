from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from google.cloud import secretmanager

load_dotenv()

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
client = secretmanager.SecretManagerServiceClient()
response = client.access_secret_version(request={"name": 'projects/1087474666309/secrets/pincone/versions/latest'})
secret_value = response.payload.data.decode("UTF-8")

def get_clip_vector(input_data, is_image=False):
    if is_image:
        inputs = processor(images=input_data, return_tensors="pt", padding=True)
        outputs = model.get_image_features(**inputs)
    else:
        inputs = processor(text=[input_data], return_tensors="pt", padding=True)
        outputs = model.get_text_features(**inputs)
    return outputs.detach().numpy().flatten()

def initialize_pinecone():
    pc = Pinecone(
        api_key=secret_value
    )
    index_name = "clip-vector-index"
    return pc.Index(index_name)
