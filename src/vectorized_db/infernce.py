from pinecone import Pinecone, ServerlessSpec
import numpy as np
from helper_functions import get_clip_vector
pc = Pinecone(api_key="3986d9fc-b6db-4264-a118-54cc2c987d0d")

# Define the index name and vector dimension
index_name = "clip-vector-index"
vector_dim = 512  # CLIP's output vector dimension

# Connect to the index
index = pc.Index(index_name)

def query_pinecone_with_text(query_text, top_k=5):
    # Get the CLIP vector for the query text
    query_vector = get_clip_vector(query_text, is_image=False)

    # Convert the numpy array to float32 and then to a Python list for Pinecone compatibility
    query_vector = query_vector.astype(np.float32).tolist()

    # Perform the query on the Pinecone index
    try:
        results = index.query(vector=query_vector, top_k=top_k, include_values=True, include_metadata=True)
        matches = results["matches"]
    except IndexError:
        # Return an empty list if there are no matches
        matches = []

    return matches


# Example text query
query_text = "A skirt"
top_matches = query_pinecone_with_text(query_text)

# Display results
for match in top_matches:
    print(f"Image ID: {match['id']}, Score: {match['score']}")
