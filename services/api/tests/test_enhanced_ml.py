import pytest
from app.ml_model import SentimentModel
from app.preprocessing import TextPreprocessor, FallbackSentimentAnalyzer


class TestEnhancedML:
    """Test enhanced ML pipeline functionality."""
    
    def test_sentiment_model_initialization(self):
        """Test sentiment model initialization."""
        model = SentimentModel()
        assert model is not None
        assert not model.is_loaded
        assert model.use_fallback
    
    def test_model_info_extraction(self):
        """Test model info extraction."""
        model = SentimentModel()
        model.load_model()  # This will use fallback
        info = model.get_model_info()
        
        assert isinstance(info, dict)
        assert "is_loaded" in info
        assert "use_fallback" in info
        assert "model_path" in info
    
    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        model = SentimentModel()
        model.load_model()
        
        texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special."
        ]
        
        results = model.predict_batch(texts)
        assert len(results) == 3
        
        for label, score, tokens in results:
            assert label in ["positive", "negative", "neutral"]
            assert 0.0 <= score <= 1.0
            assert isinstance(tokens, list)
    
    def test_enhanced_preprocessing(self):
        """Test enhanced preprocessing features."""
        preprocessor = TextPreprocessor()
        
        # Test contraction handling
        text = "I can't believe it's so amazing!"
        result = preprocessor.preprocess(text)
        assert "cannot" in result or "not" in result
        
        # Test negation handling
        text = "I don't like this product"
        result = preprocessor.preprocess(text)
        # Should preserve negation context
        assert len(result) > 0
    
    def test_fallback_analyzer_enhanced(self):
        """Test enhanced fallback analyzer."""
        analyzer = FallbackSentimentAnalyzer()
        
        # Test with complex text
        text = "I absolutely love this product, but the price is too high!"
        label, score, tokens = analyzer.predict(text)
        
        assert label in ["positive", "negative", "neutral"]
        assert 0.0 <= score <= 1.0
        assert len(tokens) > 0
        assert all("term" in token and "weight" in token for token in tokens)
    
    def test_error_handling(self):
        """Test error handling in predictions."""
        model = SentimentModel()
        model.load_model()
        
        # Test with empty text
        label, score, tokens = model.predict("")
        assert label in ["positive", "negative", "neutral"]
        assert 0.0 <= score <= 1.0
        
        # Test with very long text
        long_text = "This is a test. " * 1000
        label, score, tokens = model.predict(long_text)
        assert label in ["positive", "negative", "neutral"]
        assert 0.0 <= score <= 1.0
