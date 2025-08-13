import pytest
from app.preprocessing import TextPreprocessor, FallbackSentimentAnalyzer


def test_text_preprocessor():
    """Test text preprocessing pipeline."""
    preprocessor = TextPreprocessor(use_spacy=False)
    
    # Test basic cleaning
    text = "I LOVE this amazing product! 😊 #awesome @user http://example.com"
    cleaned = preprocessor.clean_text(text)
    assert "http://example.com" not in cleaned
    assert "@user" not in cleaned
    assert "#awesome" not in cleaned
    assert cleaned.islower()
    
    # Test punctuation removal
    text_with_punct = "Don't you think it's great?"
    no_punct = preprocessor.remove_punctuation(text_with_punct)
    assert "not" in no_punct  # Contraction expanded
    
    # Test full preprocessing
    result = preprocessor.preprocess("I really love this amazing product!")
    assert isinstance(result, str)
    assert len(result) > 0


def test_fallback_analyzer():
    """Test fallback sentiment analyzer."""
    analyzer = FallbackSentimentAnalyzer()
    
    # Test positive sentiment
    label, score, tokens = analyzer.predict("I love this amazing product!")
    assert label == "positive"
    assert 0.0 <= score <= 1.0
    assert isinstance(tokens, list)
    
    # Test negative sentiment
    label, score, tokens = analyzer.predict("I hate this terrible product!")
    assert label == "negative"
    assert 0.0 <= score <= 1.0
    
    # Test neutral sentiment
    label, score, tokens = analyzer.predict("This is a product.")
    assert label in ["positive", "negative", "neutral"]
    assert 0.0 <= score <= 1.0
