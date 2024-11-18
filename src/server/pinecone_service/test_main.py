import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

from main import app, SearchRequest

# Create a test client
client = TestClient(app)


# Mock Pinecone and Google Cloud SecretManager
@pytest.fixture
def mock_pinecone():
    with patch("main.Pinecone") as mock_pinecone, \
         patch("main.secretmanager.SecretManagerServiceClient") as mock_secret_manager:

        # Mock the SecretManager client
        mock_secret_instance = mock_secret_manager.return_value
        mock_secret_instance.access_secret_version.return_value.payload.data.decode.return_value = "mocked_pinecone_api_key"

        # Mock Pinecone index
        mock_index = MagicMock()
        mock_index.query.return_value = {
            "matches": [
                {"id": "1", "score": 0.95, "metadata": {"title": "Item 1"}},
                {"id": "2", "score": 0.85, "metadata": {"title": "Item 2"}},
            ]
        }

        mock_pinecone.return_value.Index.return_value = mock_index

        yield mock_index


# Test case: Successful search
def test_search_success(mock_pinecone):
    # Define a test search request
    search_request = {
        "vector": [0.1, 0.2, 0.3],
        "top_k": 2
    }

    # Send the POST request to the /search endpoint
    response = client.post("/search", json=search_request)

    # Assert the response
    assert response.status_code == 200
    assert response.json() == [
        {"rank": 1, "id": "1", "score": 0.95, "metadata": {"title": "Item 1"}},
        {"rank": 2, "id": "2", "score": 0.85, "metadata": {"title": "Item 2"}},
    ]

    # Verify Pinecone was queried with the correct data
    mock_pinecone.query.assert_called_once_with(
        vector=np.array([0.1, 0.2, 0.3], dtype=np.float32).tolist(),
        top_k=2,
        include_values=True,
        include_metadata=True
    )


# Test case: Error querying Pinecone
def test_search_error(mock_pinecone):
    # Simulate an error in Pinecone query
    mock_pinecone.query.side_effect = Exception("Pinecone error")

    # Define a test search request
    search_request = {
        "vector": [0.1, 0.2, 0.3],
        "top_k": 2
    }

    # Send the POST request to the /search endpoint
    response = client.post("/search", json=search_request)

    # Assert the response
    assert response.status_code == 500
    assert "Error querying Pinecone: Pinecone error" in response.json()["detail"]

    # Verify Pinecone was queried before the error
    mock_pinecone.query.assert_called_once()


# Test case: Validation error for invalid input
def test_search_validation_error():
    # Send a request with invalid input (missing top_k)
    invalid_request = {
        "vector": [0.1, 0.2, 0.3]
    }
    response = client.post("/search", json=invalid_request)

    # Assert the response
    assert response.status_code == 422  # Unprocessable Entity
    assert "top_k" in response.json()["detail"][0]["loc"][-1]

    # Send a request with an invalid vector
    invalid_request = {
        "vector": "not_a_list",
        "top_k": 2
    }
    response = client.post("/search", json=invalid_request)

    # Assert the response
    assert response.status_code == 422  # Unprocessable Entity
    assert "vector" in response.json()["detail"][0]["loc"][-1]
