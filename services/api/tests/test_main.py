import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "amazon_enabled" in data


def test_predict_endpoint():
    """Test sentiment prediction endpoint."""
    test_text = "I love this amazing product!"
    response = client.post("/predict", json={"text": test_text})
    assert response.status_code == 200
    
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert "tokens" in data
    assert data["label"] in ["positive", "negative", "neutral"]
    assert 0.0 <= data["score"] <= 1.0
    assert isinstance(data["tokens"], list)


def test_predict_empty_text():
    """Test prediction with empty text."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # Validation error


def test_predict_long_text():
    """Test prediction with very long text."""
    long_text = "This is a test. " * 1000  # Very long text
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code == 422  # Should exceed max length


def test_reviews_endpoint_disabled():
    """Test reviews endpoint when Amazon integration is disabled."""
    response = client.get("/reviews/analyze?query=test&limit=10")
    assert response.status_code == 503  # Service unavailable
