from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
from .clip_model import get_clip_vector, initialize_pinecone
import numpy as np

app = FastAPI()

# Initialize Pinecone
index_p = initialize_pinecone()

# Configure CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; you can restrict this to your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods, including POST, GET, OPTIONS
    allow_headers=["*"],  # Allows all headers
)

class SearchQuery(BaseModel):
    queryText: str
    top_k: int = 5  # Default top_k is 5

@app.post("/search")
async def search(query: SearchQuery):
    query_text = query.queryText
    top_k = query.top_k

    try:
        # Get vector for the query_text
        query_vector = get_clip_vector(query_text)
        query_vector = query_vector.astype(np.float32).tolist()

        # Perform the query on Pinecone index
        results = index_p.query(vector=query_vector, top_k=top_k, include_values=True, include_metadata=True)
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
