import tweepy
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .config import settings
from .ml_model import sentiment_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwitterService:
    """Twitter API service for fetching and analyzing tweets."""
    
    def __init__(self):
        self.client: Optional[tweepy.Client] = None
        self.is_authenticated = False
        self.rate_limit_remaining = 0
        self.rate_limit_reset = 0
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Twitter API client."""
        try:
            if not settings.enable_twitter:
                logger.info("Twitter integration is disabled")
                return
            
            if not settings.twitter_bearer_token:
                logger.warning("Twitter bearer token not provided")
                return
            
            # Initialize Tweepy client with Bearer Token
            self.client = tweepy.Client(
                bearer_token=settings.twitter_bearer_token,
                wait_on_rate_limit=True
            )
            
            # Test authentication
            try:
                # Make a simple request to test authentication
                self.client.get_me()
                self.is_authenticated = True
                logger.info("Twitter API authentication successful")
            except tweepy.Unauthorized:
                logger.error("Twitter API authentication failed - invalid bearer token")
                self.is_authenticated = False
            except tweepy.TooManyRequests:
                logger.warning("Twitter API rate limit exceeded during authentication test")
                self.is_authenticated = True  # Token is valid, just rate limited
            except Exception as e:
                logger.error(f"Twitter API authentication test failed: {e}")
                self.is_authenticated = False
                
        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")
            self.client = None
            self.is_authenticated = False
    
    def is_available(self) -> bool:
        """Check if Twitter service is available."""
        return (
            settings.enable_twitter and 
            self.client is not None and 
            self.is_authenticated
        )
    
    def _clean_tweet_text(self, text: str) -> str:
        """Clean tweet text for better sentiment analysis."""
        if not text:
            return ""
        
        # Remove RT prefix
        text = text.replace("RT @", "@")
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        return text.strip()
    
    def _extract_tweet_data(self, tweet) -> Dict:
        """Extract relevant data from tweet object."""
        try:
            return {
                "id": str(tweet.id),
                "text": self._clean_tweet_text(tweet.text),
                "created_at": tweet.created_at.isoformat() if tweet.created_at else datetime.now(timezone.utc).isoformat(),
                "author_id": str(tweet.author_id) if tweet.author_id else "unknown",
                "public_metrics": getattr(tweet, 'public_metrics', {}),
                "lang": getattr(tweet, 'lang', 'en')
            }
        except Exception as e:
            logger.error(f"Error extracting tweet data: {e}")
            return {
                "id": "unknown",
                "text": str(tweet) if tweet else "",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "author_id": "unknown",
                "public_metrics": {},
                "lang": "en"
            }
    
    async def search_tweets(self, query: str, limit: int = 25) -> List[Dict]:
        """Search for recent tweets matching the query."""
        if not self.is_available():
            raise Exception("Twitter service is not available")
        
        try:
            logger.info(f"Searching for tweets with query: '{query}', limit: {limit}")
            
            # Use ThreadPoolExecutor to run the synchronous Twitter API call
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                tweets = await loop.run_in_executor(
                    executor,
                    self._search_tweets_sync,
                    query,
                    limit
                )
            
            logger.info(f"Retrieved {len(tweets)} tweets")
            return tweets
            
        except tweepy.TooManyRequests as e:
            logger.error("Twitter API rate limit exceeded")
            raise Exception("Twitter API rate limit exceeded. Please try again later.")
        except tweepy.Unauthorized as e:
            logger.error("Twitter API unauthorized")
            raise Exception("Twitter API authentication failed")
        except tweepy.NotFound as e:
            logger.warning(f"No tweets found for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            raise Exception(f"Failed to search tweets: {str(e)}")
    
    def _search_tweets_sync(self, query: str, limit: int) -> List[Dict]:
        """Synchronous tweet search for use with ThreadPoolExecutor."""
        tweets = []
        
        try:
            # Search for recent tweets
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(limit, 100),  # Twitter API limit
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang'],
                expansions=['author_id']
            )
            
            if response.data:
                for tweet in response.data:
                    # Filter out non-English tweets if needed
                    if hasattr(tweet, 'lang') and tweet.lang not in ['en', 'und']:
                        continue
                    
                    tweet_data = self._extract_tweet_data(tweet)
                    if tweet_data["text"]:  # Only include tweets with text
                        tweets.append(tweet_data)
            
            return tweets
            
        except Exception as e:
            logger.error(f"Synchronous tweet search failed: {e}")
            raise
    
    async def analyze_tweets_sentiment(self, query: str, limit: int = 25) -> Tuple[List[Dict], Dict]:
        """Search tweets and analyze their sentiment."""
        try:
            # Search for tweets
            tweets = await self.search_tweets(query, limit)
            
            if not tweets:
                return [], {"pos": 0, "neu": 0, "neg": 0}
            
            # Analyze sentiment for each tweet
            analyzed_tweets = []
            sentiment_counts = {"pos": 0, "neu": 0, "neg": 0}
            
            logger.info(f"Analyzing sentiment for {len(tweets)} tweets")
            
            for tweet in tweets:
                try:
                    # Get sentiment prediction
                    label, score, tokens = sentiment_model.predict(tweet["text"])
                    
                    # Create analyzed tweet object
                    analyzed_tweet = {
                        "id": tweet["id"],
                        "text": tweet["text"],
                        "label": label,
                        "score": score,
                        "created_at": tweet["created_at"],
                        "author_id": tweet.get("author_id", "unknown"),
                        "public_metrics": tweet.get("public_metrics", {})
                    }
                    
                    analyzed_tweets.append(analyzed_tweet)
                    
                    # Update sentiment counts
                    if label == "positive":
                        sentiment_counts["pos"] += 1
                    elif label == "negative":
                        sentiment_counts["neg"] += 1
                    else:
                        sentiment_counts["neu"] += 1
                        
                except Exception as e:
                    logger.error(f"Error analyzing tweet {tweet.get('id', 'unknown')}: {e}")
                    # Add tweet with neutral sentiment as fallback
                    analyzed_tweets.append({
                        "id": tweet["id"],
                        "text": tweet["text"],
                        "label": "neutral",
                        "score": 0.5,
                        "created_at": tweet["created_at"],
                        "author_id": tweet.get("author_id", "unknown"),
                        "public_metrics": tweet.get("public_metrics", {})
                    })
                    sentiment_counts["neu"] += 1
            
            logger.info(f"Sentiment analysis completed. Results: {sentiment_counts}")
            return analyzed_tweets, sentiment_counts
            
        except Exception as e:
            logger.error(f"Error in tweet sentiment analysis: {e}")
            raise
    
    def get_rate_limit_status(self) -> Dict:
        """Get current rate limit status."""
        if not self.is_available():
            return {"available": False, "reason": "Service not available"}
        
        try:
            # This is a simple check - in production you might want to track this more precisely
            return {
                "available": True,
                "remaining": self.rate_limit_remaining,
                "reset_time": self.rate_limit_reset
            }
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return {"available": False, "reason": str(e)}


# Global Twitter service instance
twitter_service = TwitterService()
