import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.amazon_service import AmazonService
from app.config import settings


class TestAmazonIntegration:
    """Test Amazon integration functionality."""
    
    def test_amazon_service_initialization(self):
        """Test Amazon service initialization."""
        service = AmazonService()
        
        if settings.enable_amazon and settings.amazon_access_key_id:
            assert service.is_authenticated
        else:
            assert not service.is_authenticated
    
    def test_amazon_service_availability(self):
        """Test Amazon service availability check."""
        service = AmazonService()
        
        if not settings.enable_amazon or not settings.amazon_access_key_id:
            assert not service.is_available()
    
    def test_clean_review_text(self):
        """Test review text cleaning."""
        service = AmazonService()
        
        # Test product mention normalization
        text = "This PRODUCT is amazing for the price"
        cleaned = service._clean_review_text(text)
        assert "product" in cleaned.lower()
        
        # Test whitespace normalization
        text = "This  has   excessive    whitespace"
        cleaned = service._clean_review_text(text)
        assert cleaned == "This has excessive whitespace"
    
    def test_extract_review_data(self):
        """Test review data extraction."""
        service = AmazonService()
        
        mock_review = Mock()
        mock_review.id = "R123456789"
        mock_review.text = "This is a test review"
        mock_review.rating = 5
        mock_review.created_at = "2024-01-01T12:00:00Z"
        mock_review.reviewer_id = "user123"
        mock_review.product_id = "B001234567"
        mock_review.verified_purchase = True
        
        data = service._extract_review_data(mock_review)
        
        assert data["id"] == "R123456789"
        assert data["text"] == "This is a test review"
        assert data["reviewer_id"] == "user123"
        assert data["rating"] == 5
    
    @pytest.mark.asyncio
    async def test_search_reviews_not_available(self):
        """Test search reviews when service is not available."""
        service = AmazonService()
        service.client = None
        service.is_authenticated = False
        
        with pytest.raises(Exception, match="Amazon service is not available"):
            await service.search_reviews("test", 10)
    
    @pytest.mark.asyncio
    @patch('app.amazon_service.AmazonService._search_reviews_sync')
    async def test_analyze_reviews_sentiment(self, mock_search):
        """Test review sentiment analysis."""
        service = AmazonService()
        service.client = Mock()
        service.is_authenticated = True
        
        mock_reviews = [
            {
                "id": "R1",
                "text": "I love this product!",
                "rating": 5,
                "created_at": "2024-01-01T12:00:00Z",
                "reviewer_id": "user1"
            },
            {
                "id": "R2", 
                "text": "This is terrible",
                "rating": 1,
                "created_at": "2024-01-01T12:01:00Z",
                "reviewer_id": "user2"
            }
        ]
        
        mock_search.return_value = mock_reviews
        
        service.search_reviews = AsyncMock(return_value=mock_reviews)
        
        analyzed_reviews, summary = await service.analyze_reviews_sentiment("test", 2)
        
        assert len(analyzed_reviews) == 2
        assert "pos" in summary
        assert "neg" in summary
        assert "neu" in summary
        assert summary["pos"] + summary["neg"] + summary["neu"] == 2


class TestAmazonAPI:
    """Test Amazon API endpoints."""
    
    def test_amazon_status_endpoint(self, client):
        """Test Amazon status endpoint."""
        from app.main import app
        from fastapi.testclient import TestClient
        
        test_client = TestClient(app)
        response = test_client.get("/amazon/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "enabled" in data
        assert "available" in data
    
    def test_analyze_reviews_endpoint_disabled(self, client):
        """Test analyze reviews endpoint when disabled."""
        from app.main import app
        from fastapi.testclient import TestClient
        
        test_client = TestClient(app)
        
        if not settings.enable_amazon:
            response = test_client.get("/reviews/analyze?query=test&limit=10")
            assert response.status_code == 503
