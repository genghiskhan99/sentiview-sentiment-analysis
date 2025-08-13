import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.twitter_service import TwitterService
from app.config import settings


class TestTwitterIntegration:
    """Test Twitter integration functionality."""
    
    @patch('app.twitter_service.tweepy.Client')
    def test_twitter_service_initialization(self, mock_client):
        """Test Twitter service initialization."""
        # Mock successful authentication
        mock_instance = Mock()
        mock_instance.get_me.return_value = Mock()
        mock_client.return_value = mock_instance
        
        service = TwitterService()
        service._initialize_client()
        
        if settings.enable_twitter and settings.twitter_bearer_token:
            assert service.is_authenticated
        else:
            assert not service.is_authenticated
    
    def test_twitter_service_availability(self):
        """Test Twitter service availability check."""
        service = TwitterService()
        
        # Should return False if not properly configured
        if not settings.enable_twitter or not settings.twitter_bearer_token:
            assert not service.is_available()
    
    def test_clean_tweet_text(self):
        """Test tweet text cleaning."""
        service = TwitterService()
        
        # Test RT removal
        text = "RT @user This is a retweet"
        cleaned = service._clean_tweet_text(text)
        assert cleaned == "@user This is a retweet"
        
        # Test whitespace normalization
        text = "This  has   excessive    whitespace"
        cleaned = service._clean_tweet_text(text)
        assert cleaned == "This has excessive whitespace"
    
    def test_extract_tweet_data(self):
        """Test tweet data extraction."""
        service = TwitterService()
        
        # Mock tweet object
        mock_tweet = Mock()
        mock_tweet.id = 123456789
        mock_tweet.text = "This is a test tweet"
        mock_tweet.created_at = "2024-01-01T12:00:00Z"
        mock_tweet.author_id = "user123"
        mock_tweet.public_metrics = {"like_count": 10}
        mock_tweet.lang = "en"
        
        data = service._extract_tweet_data(mock_tweet)
        
        assert data["id"] == "123456789"
        assert data["text"] == "This is a test tweet"
        assert data["author_id"] == "user123"
        assert data["lang"] == "en"
    
    @pytest.mark.asyncio
    async def test_search_tweets_not_available(self):
        """Test search tweets when service is not available."""
        service = TwitterService()
        service.client = None
        service.is_authenticated = False
        
        with pytest.raises(Exception, match="Twitter service is not available"):
            await service.search_tweets("test", 10)
    
    @pytest.mark.asyncio
    @patch('app.twitter_service.TwitterService._search_tweets_sync')
    async def test_analyze_tweets_sentiment(self, mock_search):
        """Test tweet sentiment analysis."""
        service = TwitterService()
        service.client = Mock()
        service.is_authenticated = True
        
        # Mock tweet data
        mock_tweets = [
            {
                "id": "1",
                "text": "I love this product!",
                "created_at": "2024-01-01T12:00:00Z",
                "author_id": "user1"
            },
            {
                "id": "2", 
                "text": "This is terrible",
                "created_at": "2024-01-01T12:01:00Z",
                "author_id": "user2"
            }
        ]
        
        mock_search.return_value = mock_tweets
        
        # Mock the search_tweets method
        service.search_tweets = AsyncMock(return_value=mock_tweets)
        
        analyzed_tweets, summary = await service.analyze_tweets_sentiment("test", 2)
        
        assert len(analyzed_tweets) == 2
        assert "pos" in summary
        assert "neg" in summary
        assert "neu" in summary
        assert summary["pos"] + summary["neg"] + summary["neu"] == 2


class TestTwitterAPI:
    """Test Twitter API endpoints."""
    
    def test_twitter_status_endpoint(self, client):
        """Test Twitter status endpoint."""
        from app.main import app
        from fastapi.testclient import TestClient
        
        test_client = TestClient(app)
        response = test_client.get("/twitter/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "enabled" in data
        assert "available" in data
    
    def test_analyze_tweets_endpoint_disabled(self, client):
        """Test analyze tweets endpoint when disabled."""
        from app.main import app
        from fastapi.testclient import TestClient
        
        test_client = TestClient(app)
        
        # If Twitter is disabled, should return 503
        if not settings.enable_twitter:
            response = test_client.get("/tweets/analyze?query=test&limit=10")
            assert response.status_code == 503
