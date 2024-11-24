import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from main import app, SearchQuery

client = TestClient(app)


@pytest.fixture
def mock_vector_service():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json = AsyncMock(
            return_value={"vector": [0.1, 0.2, 0.3]}
        )
        yield mock_post


@pytest.fixture
def mock_pinecone_service():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json = AsyncMock(
            return_value=[
                {
                    "metadata": {
                        "image_name": "Test Item",
                        "brand": "Test Brand",
                        "gender": "Unisex",
                        "item_type": "Shirt",
                        "item_sub_type": "Casual",
                        "item_url": "https://example.com/item",
                        "image_url": "https://example.com/image.jpg",
                        "caption": "A stylish shirt",
                    },
                    "rank": 1,
                    "score": 0.95,
                }
            ]
        )
        yield mock_post


def test_search_success(mock_vector_service, mock_pinecone_service):
    """Test successful search endpoint with mocked services."""
    query = {"queryText": "Find a casual shirt", "top_k": 3}

    # Make the request
    response = client.post("/search", json=query)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "description" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["item_name"] == "Test Item"
    assert data["items"][0]["item_brand"] == "Test Brand"
    assert data["items"][0]["item_url"] == "https://example.com/item"


def test_search_vector_service_error(mock_vector_service):
    """Test vector service failure."""
    mock_vector_service.return_value = AsyncMock(
        return_value=AsyncMock(status_code=500, text="Internal Server Error")
    )

    query = {"queryText": "Find a casual shirt", "top_k": 3}
    response = client.post("/search", json=query)

    assert response.status_code == 500
    assert "Request error" in response.json()["detail"]


def test_search_no_vector(mock_vector_service):
    """Test vector service returning no vector."""
    mock_vector_service.return_value = AsyncMock(
        return_value=AsyncMock(
            status_code=200, json=AsyncMock(return_value={"vector": None})
        )
    )

    query = {"queryText": "Find a casual shirt", "top_k": 3}
    response = client.post("/search", json=query)

    assert response.status_code == 500
    assert "No vector returned from vector service." in response.json()["detail"]


def test_search_pinecone_service_error(mock_vector_service, mock_pinecone_service):
    """Test Pinecone service failure."""
    mock_vector_service.return_value = AsyncMock(
        return_value=AsyncMock(
            status_code=200, json=AsyncMock(return_value={"vector": [0.1, 0.2, 0.3]})
        )
    )
    mock_pinecone_service.return_value = AsyncMock(
        return_value=AsyncMock(status_code=500, text="Internal Server Error")
    )

    query = {"queryText": "Find a casual shirt", "top_k": 3}
    response = client.post("/search", json=query)

    assert response.status_code == 500
    assert "Request error" in response.json()["detail"]


def test_search_invalid_payload():
    """Test invalid payload."""
    query = {"queryText": 123, "top_k": "five"}  # Invalid types

    response = client.post("/search", json=query)

    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()


def test_search_empty_results(mock_vector_service, mock_pinecone_service):
    """Test empty results from Pinecone service."""
    mock_vector_service.return_value = AsyncMock(
        return_value=AsyncMock(
            status_code=200, json=AsyncMock(return_value={"vector": [0.1, 0.2, 0.3]})
        )
    )
    mock_pinecone_service.return_value = AsyncMock(
        return_value=AsyncMock(status_code=200, json=AsyncMock(return_value=[]))
    )

    query = {"queryText": "Find a casual shirt", "top_k": 3}
    response = client.post("/search", json=query)

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 0
