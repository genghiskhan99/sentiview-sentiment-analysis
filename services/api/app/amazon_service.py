import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import time
import asyncio
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

from .config import settings
from .ml_model import sentiment_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AmazonService:
    """Amazon reviews service for fetching and analyzing product reviews."""
    
    def __init__(self):
        self.is_authenticated = True  # Amazon data is publicly available
        self.rate_limit_remaining = 1000
        self.rate_limit_reset = 0
        self._initialize_service()
    
    def _initialize_service(self) -> None:
        """Initialize Amazon reviews service."""
        try:
            if not settings.enable_amazon:
                logger.info("Amazon integration is disabled")
                return
            
            logger.info("Amazon reviews service initialized successfully")
            self.is_authenticated = True
                
        except Exception as e:
            logger.error(f"Failed to initialize Amazon service: {e}")
            self.is_authenticated = False
    
    def is_available(self) -> bool:
        """Check if Amazon service is available."""
        return settings.enable_amazon and self.is_authenticated
    
    def _clean_review_text(self, text: str) -> str:
        """Clean review text for better sentiment analysis."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        return text.strip()
    
    def _extract_review_data(self, review_text: str, index: int) -> Dict:
        """Extract relevant data from review text."""
        try:
            return {
                "id": f"review_{index}",
                "text": self._clean_review_text(review_text),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "author_id": f"user_{index}",
                "rating": None,  # Will be filled if available
                "product_id": "unknown"
            }
        except Exception as e:
            logger.error(f"Error extracting review data: {e}")
            return {
                "id": f"review_{index}",
                "text": str(review_text) if review_text else "",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "author_id": f"user_{index}",
                "rating": None,
                "product_id": "unknown"
            }
    
    async def search_reviews(self, query: str, limit: int = 25) -> List[Dict]:
        """Search for Amazon reviews matching the query."""
        if not self.is_available():
            raise Exception("Amazon service is not available")
        
        try:
            logger.info(f"Searching for Amazon reviews with query: '{query}', limit: {limit}")
            
            # Use ThreadPoolExecutor to run the synchronous data loading
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                reviews = await loop.run_in_executor(
                    executor,
                    self._search_reviews_sync,
                    query,
                    limit
                )
            
            logger.info(f"Retrieved {len(reviews)} reviews")
            return reviews
            
        except Exception as e:
            logger.error(f"Error searching reviews: {e}")
            raise Exception(f"Failed to search reviews: {str(e)}")
    
    def _search_reviews_sync(self, query: str, limit: int) -> List[Dict]:
        """Synchronous review search for use with ThreadPoolExecutor."""
        reviews = []
        
        try:
            # Load Amazon reviews dataset
            dataset_url = "https://drive.google.com/uc?export=download&id=1SERc309kmcEGhsqhuztIE_ZaqJGku-WQ"
            
            try:
                df = pd.read_csv(dataset_url)
                logger.info(f"Loaded Amazon dataset with {len(df)} reviews")
            except Exception as e:
                logger.warning(f"Failed to load remote dataset: {e}")
                # Generate sample reviews as fallback
                sample_reviews = [
                    "Great product, highly recommend!",
                    "Amazing quality and fast shipping",
                    "Love this item, works perfectly",
                    "Excellent value for money",
                    "Outstanding customer service",
                    "Very pleased with this purchase",
                    "Good quality but could be better",
                    "Average product, nothing special",
                    "Not what I expected",
                    "Poor quality, disappointed",
                    "Terrible experience, would not buy again",
                    "Worst purchase ever made"
                ]
                df = pd.DataFrame({
                    'input': sample_reviews * (limit // len(sample_reviews) + 1),
                    'label': ['Positive'] * 6 + ['Neutral'] * 2 + ['Negative'] * 4
                })
            
            # Filter reviews based on query if specified
            if query and query.lower() not in ['all', 'reviews', 'amazon']:
                # Simple text matching for demo purposes
                mask = df['input'].str.contains(query, case=False, na=False)
                filtered_df = df[mask]
                if len(filtered_df) == 0:
                    # If no matches, return random sample
                    filtered_df = df.sample(n=min(limit, len(df)))
            else:
                filtered_df = df.sample(n=min(limit, len(df)))
            
            # Convert to review format
            for idx, (_, row) in enumerate(filtered_df.head(limit).iterrows()):
                review_data = self._extract_review_data(row['input'], idx)
                if review_data["text"]:  # Only include reviews with text
                    reviews.append(review_data)
            
            return reviews
            
        except Exception as e:
            logger.error(f"Synchronous review search failed: {e}")
            raise
    
    async def analyze_reviews_sentiment(self, query: str, limit: int = 25) -> Tuple[List[Dict], Dict]:
        """Search Amazon reviews and analyze their sentiment."""
        try:
            # Search for reviews
            reviews = await self.search_reviews(query, limit)
            
            if not reviews:
                return [], {"pos": 0, "neu": 0, "neg": 0}
            
            # Analyze sentiment for each review
            analyzed_reviews = []
            sentiment_counts = {"pos": 0, "neu": 0, "neg": 0}
            
            logger.info(f"Analyzing sentiment for {len(reviews)} reviews")
            
            for review in reviews:
                try:
                    # Get sentiment prediction
                    label, score, tokens = sentiment_model.predict(review["text"])
                    
                    # Create analyzed review object
                    analyzed_review = {
                        "id": review["id"],
                        "text": review["text"],
                        "label": label,
                        "score": score,
                        "created_at": review["created_at"],
                        "author_id": review.get("author_id", "unknown"),
                        "rating": review.get("rating")
                    }
                    
                    analyzed_reviews.append(analyzed_review)
                    
                    # Update sentiment counts
                    if label == "positive":
                        sentiment_counts["pos"] += 1
                    elif label == "negative":
                        sentiment_counts["neg"] += 1
                    else:
                        sentiment_counts["neu"] += 1
                        
                except Exception as e:
                    logger.error(f"Error analyzing review {review.get('id', 'unknown')}: {e}")
                    # Add review with neutral sentiment as fallback
                    analyzed_reviews.append({
                        "id": review["id"],
                        "text": review["text"],
                        "label": "neutral",
                        "score": 0.5,
                        "created_at": review["created_at"],
                        "author_id": review.get("author_id", "unknown"),
                        "rating": review.get("rating")
                    })
                    sentiment_counts["neu"] += 1
            
            logger.info(f"Sentiment analysis completed. Results: {sentiment_counts}")
            return analyzed_reviews, sentiment_counts
            
        except Exception as e:
            logger.error(f"Error in review sentiment analysis: {e}")
            raise
    
    def get_rate_limit_status(self) -> Dict:
        """Get current rate limit status."""
        if not self.is_available():
            return {"available": False, "reason": "Service not available"}
        
        try:
            return {
                "available": True,
                "remaining": self.rate_limit_remaining,
                "reset_time": self.rate_limit_reset
            }
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return {"available": False, "reason": str(e)}


# Global Amazon service instance
amazon_service = AmazonService()
