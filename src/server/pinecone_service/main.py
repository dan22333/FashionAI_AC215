from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
from pinecone import Pinecone
from google.cloud import secretmanager

app = FastAPI()

# Initialize Pinecone
client = secretmanager.SecretManagerServiceClient()
response = client.access_secret_version(request={"name": 'projects/1087474666309/secrets/pincone/versions/latest'})
secret_value = response.payload.data.decode("UTF-8")
pc = Pinecone(api_key=secret_value)
index = pc.Index("clip-vector-index")

class SearchRequest(BaseModel):
    vector: list
    top_k: int

@app.post("/search")
async def search(request: SearchRequest):
    try:
        results = index.query(vector=np.array(request.vector, dtype=np.float32).tolist(), top_k=request.top_k, include_values=True, include_metadata=True)
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
    uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=True)