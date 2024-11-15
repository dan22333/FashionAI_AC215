from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchQuery(BaseModel):
    queryText: str
    top_k: int = 5

@app.post("/search")
async def search(query: SearchQuery):
    query_text = query.queryText
    top_k = query.top_k

    try:
        # Request vector from vector service
        response = requests.post("http://vector_service:8001/get_vector", json={"text": query_text})
        response.raise_for_status()
        query_vector = response.json()["vector"]

        # Request Pinecone service
        pinecone_response = requests.post("http://pinecone_service:8002/search", json={"vector": query_vector, "top_k": top_k})
        pinecone_response.raise_for_status()
        return pinecone_response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
