import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestEnhancedAPI:
    """Test enhanced API functionality."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "is_loaded" in data
        assert "use_fallback" in data
    
    def test_batch_prediction_endpoint(self):
        """Test batch prediction endpoint."""
        texts = [
            "I love this!",
            "This is terrible.",
            "It's okay."
        ]
        
        response = client.post("/predict/batch", json=texts)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "count" in data
        assert len(data["results"]) == 3
        
        for result in data["results"]:
            assert "label" in result
            assert "score" in result
            assert "tokens" in result
    
    def test_batch_prediction_validation(self):
        """Test batch prediction validation."""
        # Test empty list
        response = client.post("/predict/batch", json=[])
        assert response.status_code == 400
        
        # Test too many texts
        long_list = ["test"] * 101
        response = client.post("/predict/batch", json=long_list)
        assert response.status_code == 400
    
    def test_stats_endpoint(self):
        """Test stats endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "model_loaded" in data
        assert "settings" in data
    
    def test_enhanced_error_handling(self):
        """Test enhanced error handling."""
        # Test with invalid JSON
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422
        
        # Test with empty text
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 400
