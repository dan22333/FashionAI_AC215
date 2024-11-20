import httpx
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to limit the origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_SERVICE_HOST = os.getenv("VECTOR_SERVICE_HOST", "vector_service")
PINECONE_SERVICE_HOST = os.getenv("PINECONE_SERVICE_HOST", "pinecone_service")

class SearchQuery(BaseModel):
    queryText: str
    top_k: int = 5

@app.post("/search")
async def search(query: SearchQuery):
    query_text = query.queryText
    top_k = query.top_k

    async with httpx.AsyncClient() as client:
        try:
            vector_service_url = f"http://{VECTOR_SERVICE_HOST}:8001/get_vector"
            response = await client.post(vector_service_url, json={"text": query_text}, timeout=10)
            response.raise_for_status()
            query_vector = response.json().get("vector")
            if not query_vector:
                raise ValueError("No vector returned from vector service.")

            pinecone_service_url = f"http://{PINECONE_SERVICE_HOST}:8002/search"
            pinecone_response = await client.post(
                pinecone_service_url, 
                json={"vector": query_vector, "top_k": top_k}, 
                timeout=10
            )
            pinecone_response.raise_for_status()
            search_results = pinecone_response.json()
            items = [
                {
                    "item_name": result["metadata"].get("name", "Unknown Name"),
                    "item_type": result["metadata"].get("type", "Unknown Name"),
                    "item_url": result["metadata"].get("url", "Unknown URL"),
                    "item_caption": result["metadata"].get("caption", "No caption available"),
                    "item_brand": result["metadata"].get("brand", "Unknown Brand"),
                    "rank": result.get("rank", "N/A"),
                    "score": result.get("score", "N/A"),
                }
                for result in search_results if "metadata" in result
            ]

            return {"description": f"Search results for '{query_text}'", "items": items}

        except httpx.RequestError as req_exc:
            raise HTTPException(status_code=500, detail=f"Request error: {req_exc}")

        except ValueError as val_exc:
            raise HTTPException(status_code=501, detail=f"Value error: {val_exc}")

        except KeyError as key_exc:
            raise HTTPException(status_code=502, detail=f"Key error: {key_exc}")

        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Unexpected error: {e}")

# Add this block to run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)