import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

client = TestClient(app)

class MockSecretManagerClient:
    def access_secret_version(self, request):
        if request["name"] == "projects/1087474666309/secrets/pincone/versions/latest":
            return MockSecretManagerResponse()
        raise ValueError("Invalid secret name")

class MockSecretManagerResponse:
    class Payload:
        def __init__(self):
            self.data = b"mocked-api-key"
    def __init__(self):
        self.payload = self.Payload()

class MockPineconeClient:
    class Index:
        def __init__(self, name):
            self.name = name

        def query(self, vector, top_k, include_values, include_metadata):
            if len(vector) != 512:
                raise ValueError("Vector dimension does not match index dimension")
            return {
                "matches": [
                    {"id": "item1", "score": 0.95, "metadata": {"label": "label1"}},
                    {"id": "item2", "score": 0.89, "metadata": {"label": "label2"}}
                ]
            }

    def __init__(self, api_key):
        self.api_key = api_key

@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch):
    monkeypatch.setattr("main.secretmanager.SecretManagerServiceClient", lambda: MockSecretManagerClient())
    monkeypatch.setattr("main.Pinecone", lambda api_key: MockPineconeClient(api_key))

def test_search_success():
    payload = {
        "vector": [0.1] * 512,  # Adjusted to match index dimension
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
        "vector": "invalid_vector",
        "top_k": 2
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 422
    error_message = response.json()["detail"][0]["msg"]
    assert "Input should be a valid list" in error_message

def test_search_internal_error():
    with patch("main.MockPineconeClient.Index.query", side_effect=Exception("Mocked internal error")):
        payload = {
            "vector": [0.1] * 512,
            "top_k": 2
        }
        response = client.post("/search", json=payload)
        assert response.status_code == 500
        assert "Error querying Pinecone: Mocked internal error" in response.json()["detail"]
