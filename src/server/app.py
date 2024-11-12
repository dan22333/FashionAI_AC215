from flask import Flask, request, jsonify, render_template
import requests
import os
import numpy as np
from dotenv import load_dotenv

from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_vector(input_data, is_image=False):
    if is_image:
        inputs = processor(images=input_data, return_tensors="pt", padding=True)
        outputs = model.get_image_features(**inputs)
    else:
        inputs = processor(text=[input_data], return_tensors="pt", padding=True)
        outputs = model.get_text_features(**inputs)
    return outputs.detach().numpy().flatten()

# Load environment variables
load_dotenv()

app = Flask(__name__)

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="3986d9fc-b6db-4264-a118-54cc2c987d0d")

# Define the index name and vector dimension
index_name = "clip-vector-index"
vector_dim = 512  # CLIP's output vector dimension

# Connect to the index
index_p = pc.Index(index_name)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query_text = data.get("queryText")
    top_k = data.get("top_k", 5)  # Default to 5 if top_k is not provided

    try:
        # Get the vector for the query_text
        query_vector = get_clip_vector(query_text)

        # Convert the numpy array to float32 and then to a Python list for Pinecone compatibility
        query_vector = query_vector.astype(np.float32).tolist()

        # Perform the query on the Pinecone index
        results = index_p.query(vector=query_vector, top_k=top_k, include_values=True, include_metadata=True)
        matches = results.get("matches", [])

        # Format the results for the frontend
        formatted_matches = []
        for idx, match in enumerate(matches):
            formatted_matches.append({
                "rank": idx + 1,
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {})
            })

        # Return a JSON response
        return jsonify(formatted_matches), 200

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return jsonify({"message": "Error querying Pinecone", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
