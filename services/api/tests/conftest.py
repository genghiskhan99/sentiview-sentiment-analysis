import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_amazon_data():
    """Mock Amazon review data for testing."""
    return [
        {
            "id": "R1234567890",
            "text": "I love this amazing product! Great quality and fast shipping.",
            "rating": 5,
            "created_at": "2024-01-01T12:00:00Z",
            "reviewer_id": "user123",
            "product_id": "B001234567",
            "verified_purchase": True
        },
        {
            "id": "R1234567891", 
            "text": "This is terrible and disappointing. Poor quality.",
            "rating": 1,
            "created_at": "2024-01-01T12:01:00Z",
            "reviewer_id": "user456",
            "product_id": "B001234568",
            "verified_purchase": True
        },
        {
            "id": "R1234567892",
            "text": "It's okay, nothing special really. Average product.",
            "rating": 3,
            "created_at": "2024-01-01T12:02:00Z", 
            "reviewer_id": "user789",
            "product_id": "B001234569",
            "verified_purchase": False
        }
    ]
