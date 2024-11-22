from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
from pinecone import Pinecone
from google.cloud import secretmanager

# Load environment variables
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT_PINECONE = int(os.getenv("APP_PORT_PINECONE", 8002))
PINECONE_SECRET_NAME = os.getenv("PINECONE_SECRET_NAME", "projects/1087474666309/secrets/pincone/versions/latest")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "clip-vector-index-test-prod")

app = FastAPI()

# Initialize Pinecone
client = secretmanager.SecretManagerServiceClient()
response = client.access_secret_version(request={"name": PINECONE_SECRET_NAME})
secret_value = response.payload.data.decode("UTF-8")
pc = Pinecone(api_key=secret_value)
index = pc.Index(PINECONE_INDEX_NAME)

class SearchRequest(BaseModel):
    vector: list
    top_k: int

@app.post("/search")
async def search(request: SearchRequest):
    try:
        results = index.query(
            vector=np.array(request.vector, dtype=np.float32).tolist(),
            top_k=request.top_k,
            include_values=True,
            include_metadata=True
        )
        matches = results.get("matches", [])

        formatted_matches = [
            {
                "rank": idx + 1,
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {})
            }
            for idx, match in enumerate(matches)
        ]
        return formatted_matches

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Pinecone: {str(e)}")

# Add this block to run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT_PINECONE, reload=True)
