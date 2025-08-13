import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_twitter_data():
    """Mock Twitter data for testing."""
    return [
        {
            "id": "1234567890",
            "text": "I love this amazing product! #awesome",
            "created_at": "2024-01-01T12:00:00Z",
            "author_id": "user123",
            "public_metrics": {"like_count": 10, "retweet_count": 5}
        },
        {
            "id": "1234567891", 
            "text": "This is terrible and disappointing",
            "created_at": "2024-01-01T12:01:00Z",
            "author_id": "user456",
            "public_metrics": {"like_count": 2, "retweet_count": 1}
        },
        {
            "id": "1234567892",
            "text": "It's okay, nothing special really",
            "created_at": "2024-01-01T12:02:00Z", 
            "author_id": "user789",
            "public_metrics": {"like_count": 5, "retweet_count": 0}
        }
    ]
