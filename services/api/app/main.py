from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import time
import logging
from typing import List
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from .config import settings
from .models import (
    TextInput, 
    SentimentResponse, 
    ReviewAnalysisResponse, 
    HealthResponse,
    TokenWeight,
    ReviewItem,
    ReviewSummary
)
from .ml_model import sentiment_model
from .amazon_service import amazon_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global VADER analyzer
vader_analyzer = None


def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    try:
        # Check if VADER lexicon is available
        nltk.data.find('vader_lexicon')
        logger.info("VADER lexicon already available")
    except LookupError:
        logger.info("Downloading VADER lexicon...")
        nltk.download('vader_lexicon', quiet=True)
    
    try:
        # Check if stopwords are available
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK stopwords already available")
    except LookupError:
        logger.info("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    global vader_analyzer
    
    # Startup
    logger.info("Starting Sentiview API...")
    start_time = time.time()
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Initialize VADER analyzer
    try:
        vader_analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER sentiment analyzer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize VADER analyzer: {e}")
        vader_analyzer = None
    
    # Load the ML model
    model_loaded = sentiment_model.load_model()
    load_time = time.time() - start_time
    
    if model_loaded:
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
    else:
        logger.info(f"Using fallback analyzer (load time: {load_time:.2f} seconds)")
    
    # Log model information
    model_info = sentiment_model.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Check Amazon service status
    if amazon_service.is_available():
        logger.info("Amazon service is available")
    else:
        logger.info("Amazon service is not available")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sentiview API...")


# Create FastAPI app
app = FastAPI(
    title="Sentiview API",
    description="Real-time sentiment analysis API with machine learning and Amazon reviews integration",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sentiview API - Real-time Sentiment Analysis",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "features": {
            "sentiment_analysis": True,
            "vader_fusion": vader_analyzer is not None,
            "amazon_integration": amazon_service.is_available(),
            "batch_processing": True
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint with detailed status."""
    try:
        model_info = sentiment_model.get_model_info()
        
        return HealthResponse(
            status="ok",
            model_loaded=sentiment_model.is_loaded,
            amazon_enabled=amazon_service.is_available()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            amazon_enabled=False
        )


@app.get("/model/info")
async def get_model_info():
    """Get detailed information about the loaded model."""
    try:
        info = sentiment_model.get_model_info()
        info["vader_available"] = vader_analyzer is not None
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(input_data: TextInput):
    """Predict sentiment for given text with VADER fusion and enhanced error handling."""
    try:
        start_time = time.time()
        
        # Validate input
        if not input_data.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Get ML model prediction
        label, p_pos, tokens = sentiment_model.predict(input_data.text)
        
        # Get VADER prediction if available
        p_vader = 0.5  # Default neutral if VADER not available
        if vader_analyzer:
            try:
                vader_scores = vader_analyzer.polarity_scores(input_data.text)
                # Map VADER compound score from [-1, 1] to [0, 1]
                p_vader = (vader_scores['compound'] + 1) / 2
            except Exception as e:
                logger.warning(f"VADER analysis failed: {e}")
                p_vader = 0.5
        
        p_mix = 0.7 * p_pos + 0.3 * p_vader
        
        # Apply neutral band classification
        if p_mix >= 0.60:
            final_label = "positive"
        elif p_mix <= 0.40:
            final_label = "negative"
        else:
            final_label = "neutral"
        
        if final_label == "neutral":
            lower_text = input_data.text.lower().strip()
            
            # Strong single words
            strong_pos_words = {"great", "excellent", "amazing", "fantastic", "awesome", "perfect", "wonderful", "outstanding", "brilliant", "superb", "very pleased", "very good", "very nice", "love it", "like it"}
            strong_neg_words = {"terrible", "awful", "worst", "horrible", "disgusting", "pathetic", "useless", "dreadful", "appalling", "atrocious", "very bad", "hate it"}
            
            # Common positive phrases
            positive_phrases = [
                "i like", "i love", "i recommend", "i will recommend", "i would recommend",
                "highly recommend", "strongly recommend", "really like", "really love",
                "works great", "works well", "love it", "like it", "recommend it", "worth buying", "worth it", "good product",
                "nice product", "happy with", "satisfied with", "pleased with",
                "very pleased", "very happy", "very satisfied", "quite good", "pretty good",
                "quite nice", "pretty nice", "really good", "really nice", "so good",
                "so nice", "really pleased", "quite pleased", "pretty pleased"
            ]
            
            # Common negative phrases  
            negative_phrases = [
                "i hate", "i don't like", "i dislike", "don't recommend", "wouldn't recommend",
                "not recommend", "don't buy", "avoid this", "waste of money", "not worth",
                "not good", "not working", "doesn't work", "poor quality", "bad quality",
                "hate it", "dislike it", "regret buying", "disappointed with", "unhappy with"
            ]
            
            # Check single words first
            if lower_text in strong_pos_words:
                final_label = "positive"
                logger.info(f"Override: '{lower_text}' classified as positive (strong word)")
            elif lower_text in strong_neg_words:
                final_label = "negative"
                logger.info(f"Override: '{lower_text}' classified as negative (strong word)")
            else:
                # Check for positive phrases
                for phrase in positive_phrases:
                    if phrase in lower_text:
                        final_label = "positive"
                        logger.info(f"Override: Text containing '{phrase}' classified as positive")
                        break
                
                # Check for negative phrases (only if not already set to positive)
                if final_label == "neutral":
                    for phrase in negative_phrases:
                        if phrase in lower_text:
                            final_label = "negative"
                            logger.info(f"Override: Text containing '{phrase}' classified as negative")
                            break
            
        # Short text polarity adjustment (only if not already overridden by strong words)
        if final_label == "neutral":
            text_words = input_data.text.strip().split()
            if len(text_words) <= 6:
                if p_mix >= 0.55:  # Lower threshold for short positive texts
                    final_label = "positive"
                elif p_mix <= 0.45:  # Higher threshold for short negative texts
                    final_label = "negative"
        
        # Convert tokens to proper format
        token_weights = [
            TokenWeight(term=token["term"], weight=token["weight"])
            for token in tokens
        ]
        
        prediction_time = time.time() - start_time
        logger.info(f"Prediction completed in {prediction_time:.3f}s for text length: {len(input_data.text)}")
        
        return {
            "label": final_label,
            "score": float(p_mix),
            "tokens": token_weights,
            "probs": {
                "model": float(p_pos),
                "vader": float(p_vader),
                "mix": float(p_mix)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_sentiment_batch(texts: List[str]):
    """Predict sentiment for multiple texts efficiently with VADER fusion."""
    try:
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(texts) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 texts allowed per batch")
        
        start_time = time.time()
        
        # Validate all texts
        for i, text in enumerate(texts):
            if not text.strip():
                raise HTTPException(status_code=400, detail=f"Text at index {i} cannot be empty")
            if len(text) > 5000:
                raise HTTPException(status_code=400, detail=f"Text at index {i} exceeds maximum length")
        
        # Make batch predictions
        results = sentiment_model.predict_batch(texts)
        
        # Format results with VADER fusion
        formatted_results = []
        for i, (label, p_pos, tokens) in enumerate(results):
            # Get VADER prediction for this text
            p_vader = 0.5
            if vader_analyzer:
                try:
                    vader_scores = vader_analyzer.polarity_scores(texts[i])
                    p_vader = (vader_scores['compound'] + 1) / 2
                except Exception as e:
                    logger.warning(f"VADER analysis failed for text {i}: {e}")
            
            p_mix = 0.7 * p_pos + 0.3 * p_vader
            
            # Apply neutral band classification
            if p_mix >= 0.60:
                final_label = "positive"
            elif p_mix <= 0.40:
                final_label = "negative"
            else:
                final_label = "neutral"
            
            if final_label == "neutral":
                lower_text = texts[i].lower().strip()
                
                # Strong single words
                strong_pos_words = {"great", "excellent", "amazing", "fantastic", "awesome", "perfect", "wonderful", "outstanding", "brilliant", "superb", "very pleased", "very good", "very nice", "love it", "like it"}
                strong_neg_words = {"terrible", "awful", "worst", "horrible", "disgusting", "pathetic", "useless", "dreadful", "appalling", "atrocious", "very bad", "hate it"}
                
                # Common positive phrases
                positive_phrases = [
                    "i like", "i love", "i recommend", "i will recommend", "i would recommend",
                    "highly recommend", "strongly recommend", "really like", "really love",
                    "works great", "works well", "love it", "like it", "recommend it", "worth buying", "worth it", "good product",
                    "nice product", "happy with", "satisfied with", "pleased with",
                    "very pleased", "very happy", "very satisfied", "quite good", "pretty good",
                    "quite nice", "pretty nice", "really good", "really nice", "so good",
                    "so nice", "really pleased", "quite pleased", "pretty pleased"
                ]
                
                # Common negative phrases  
                negative_phrases = [
                    "i hate", "i don't like", "i dislike", "don't recommend", "wouldn't recommend",
                    "not recommend", "don't buy", "avoid this", "waste of money", "not worth",
                    "not good", "not working", "doesn't work", "poor quality", "bad quality",
                    "hate it", "dislike it", "regret buying", "disappointed with", "unhappy with"
                ]
                
                # Check single words first
                if lower_text in strong_pos_words:
                    final_label = "positive"
                elif lower_text in strong_neg_words:
                    final_label = "negative"
                else:
                    # Check for positive phrases
                    for phrase in positive_phrases:
                        if phrase in lower_text:
                            final_label = "positive"
                            logger.info(f"Override: Text containing '{phrase}' classified as positive")
                            break
                    
                    # Check for negative phrases (only if not already set to positive)
                    if final_label == "neutral":
                        for phrase in negative_phrases:
                            if phrase in lower_text:
                                final_label = "negative"
                                logger.info(f"Override: Text containing '{phrase}' classified as negative")
                                break
            
            # Short text polarity adjustment (only if not already overridden by strong words)
            if final_label == "neutral":
                text_words = texts[i].strip().split()
                if len(text_words) <= 6:
                    if p_mix >= 0.55:  # Lower threshold for short positive texts
                        final_label = "positive"
                    elif p_mix <= 0.45:  # Higher threshold for short negative texts
                        final_label = "negative"
            
            token_weights = [
                TokenWeight(term=token["term"], weight=token["weight"])
                for token in tokens
            ]
            
            formatted_results.append({
                "index": i,
                "label": final_label,
                "score": float(p_mix),
                "tokens": token_weights,
                "probs": {
                    "model": float(p_pos),
                    "vader": float(p_vader),
                    "mix": float(p_mix)
                }
            })
        
        prediction_time = time.time() - start_time
        logger.info(f"Batch prediction completed in {prediction_time:.3f}s for {len(texts)} texts")
        
        return {
            "results": formatted_results,
            "count": len(texts),
            "processing_time": prediction_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/reviews/analyze", response_model=ReviewAnalysisResponse)
async def analyze_reviews(
    query: str = Query(..., description="Search query for Amazon reviews", min_length=1, max_length=500),
    limit: int = Query(25, ge=1, le=100, description="Number of reviews to analyze")
):
    """Analyze sentiment of Amazon product reviews."""
    try:
        # Check if Amazon integration is available
        if not amazon_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Amazon service is not available. Please check the service configuration."
            )
        
        start_time = time.time()
        logger.info(f"Starting Amazon review analysis for query: '{query}' with limit: {limit}")
        
        # Analyze reviews
        analyzed_reviews, sentiment_summary = await amazon_service.analyze_reviews_sentiment(query, limit)
        
        # Convert to response format
        review_items = [
            ReviewItem(
                id=review["id"],
                text=review["text"],
                label=review["label"],
                score=review["score"],
                created_at=review["created_at"]
            )
            for review in analyzed_reviews
        ]
        
        summary = ReviewSummary(
            pos=sentiment_summary["pos"],
            neu=sentiment_summary["neu"],
            neg=sentiment_summary["neg"]
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Amazon review analysis completed in {processing_time:.3f}s for {len(review_items)} reviews")
        
        return ReviewAnalysisResponse(
            items=review_items,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Amazon review analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Amazon review analysis failed: {str(e)}")


@app.get("/amazon/status")
async def get_amazon_status():
    """Get Amazon service status and information."""
    try:
        status = {
            "enabled": settings.enable_amazon,
            "available": amazon_service.is_available(),
            "authenticated": amazon_service.is_authenticated,
        }
        
        if amazon_service.is_available():
            rate_limit = amazon_service.get_rate_limit_status()
            status.update(rate_limit)
        else:
            status["reason"] = "Amazon service not available"
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting Amazon status: {e}")
        return {
            "enabled": settings.enable_amazon,
            "available": False,
            "error": str(e)
        }


# Additional utility endpoints
@app.get("/stats")
async def get_api_stats():
    """Get API usage statistics."""
    return {
        "model_type": "fallback" if sentiment_model.use_fallback else "trained",
        "model_loaded": sentiment_model.is_loaded,
        "vader_available": vader_analyzer is not None,
        "amazon_available": amazon_service.is_available(),
        "settings": {
            "cors_origins": settings.cors_origins,
            "amazon_enabled": settings.enable_amazon,
            "environment": settings.environment
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
