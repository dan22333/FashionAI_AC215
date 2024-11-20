import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

@pytest.fixture(autouse=True)
def mock_pinecone_and_secret_manager(monkeypatch):
    import main
    
    # Mock Google Secret Manager client
    mock_secret_manager_client = MagicMock()
    mock_secret_manager_response = MagicMock()
    mock_secret_manager_response.payload.data.decode.return_value = "mocked-api-key"
    mock_secret_manager_client.access_secret_version.return_value = mock_secret_manager_response
    
    # Mock Pinecone
    mock_pinecone = MagicMock()
    mock_index = MagicMock()
    mock_pinecone.Index.return_value = mock_index
    mock_index.query.return_value = {
        "matches": [
            {"id": "item1", "score": 0.95, "metadata": {"label": "label1"}},
            {"id": "item2", "score": 0.89, "metadata": {"label": "label2"}},
        ]
    }
    
    # Apply mocks
    monkeypatch.setattr(main.secretmanager, "SecretManagerServiceClient", lambda: mock_secret_manager_client)
    monkeypatch.setattr(main, "Pinecone", lambda api_key: mock_pinecone)
    return mock_index

def test_search_success():
    payload = {
        "vector": [0.1, 0.2, 0.3],
        "top_k": 2
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["id"] == "item1"
    assert data[1]["id"] == "item2"


def test_search_invalid_vector():
    payload = {
        "vector": "invalid_vector",  # Invalid type
        "top_k": 2
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 422
    error_message = response.json()["detail"][0]["msg"]
    assert "Input should be a valid list" in error_message  # Match actual error message
