from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
import torch

app = FastAPI()

# Load CLIP model and processor
model = CLIPModel.from_pretrained("weiyueli7/fashionclip")
processor = CLIPProcessor.from_pretrained("weiyueli7/fashionclip")

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

# Add this block to run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)