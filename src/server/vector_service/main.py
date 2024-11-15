from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
import torch

app = FastAPI()

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class VectorRequest(BaseModel):
    text: str

@app.post("/get_vector")
async def get_vector(request: VectorRequest):
    try:
        inputs = processor(text=[request.text], return_tensors="pt", padding=True)
        outputs = model.get_text_features(**inputs)
        vector = outputs.detach().numpy().flatten().tolist()
        return {"vector": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating vector: {str(e)}")
