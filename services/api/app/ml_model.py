import os
import pickle
import joblib
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import logging
import glob
from datetime import datetime

from .preprocessing import TextPreprocessor, FallbackSentimentAnalyzer
from .config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentModel:
    """Enhanced sentiment analysis model wrapper with robust prediction and feature extraction."""
    
    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.preprocessor = TextPreprocessor()
        self.fallback_analyzer = FallbackSentimentAnalyzer()
        self.is_loaded = False
        self.use_fallback = False
        self.model_info: Dict[str, Any] = {}
        self.is_augmented_model = False
        
    def load_model(self) -> bool:
        """Load the trained model from disk with preference for augmented models."""
        try:
            model_path = self._find_best_model()
            
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading model from {model_path}")
                
                if "augmented" in model_path:
                    self._load_augmented_model(model_path)
                else:
                    self.model = joblib.load(model_path)
                
                self.is_loaded = True
                self.use_fallback = False
                
                # Extract model information
                self._extract_model_info()
                logger.info(f"Model loaded successfully: {self.model_info}")
                return True
            else:
                logger.warning(f"No model files found in models directory")
                logger.info("Using fallback TextBlob analyzer")
                self.use_fallback = True
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Using fallback TextBlob analyzer")
            self.use_fallback = True
            return False
    
    def _find_best_model(self) -> Optional[str]:
        """Find the best available model, preferring augmented > Amazon reviews > basic models."""
        models_dir = os.path.dirname(settings.model_path)
        
        # Priority 1: Augmented models (best performance)
        augmented_models = glob.glob(os.path.join(models_dir, "*augmented*.joblib"))
        if augmented_models:
            augmented_models.sort(reverse=True)
            self.is_augmented_model = True
            logger.info(f"Found {len(augmented_models)} augmented models, using newest")
            return augmented_models[0]
        
        # Priority 2: Amazon reviews models (good baseline)
        amazon_models = glob.glob(os.path.join(models_dir, "sentiment_lr_tfidf.pkl"))
        if amazon_models:
            logger.info("Found Amazon reviews model, using it")
            return amazon_models[0]
        
        # Priority 3: Regular models
        regular_models = glob.glob(os.path.join(models_dir, "*.joblib"))
        regular_models = [m for m in regular_models if "augmented" not in m]
        if regular_models:
            regular_models.sort(reverse=True)
            logger.info(f"Found {len(regular_models)} regular models, using newest")
            return regular_models[0]
        
        # Priority 4: Check original model path
        if os.path.exists(settings.model_path):
            return settings.model_path
        
        return None
    
    def _load_augmented_model(self, model_path: str) -> None:
        """Load augmented model and its corresponding vectorizer."""
        try:
            self.model = joblib.load(model_path)
            
            vectorizer_path = model_path.replace("sentiment_model_", "vectorizer_")
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
                logger.info(f"Loaded corresponding vectorizer from {vectorizer_path}")
            else:
                logger.warning(f"Vectorizer not found at {vectorizer_path}")
                
        except Exception as e:
            logger.error(f"Error loading augmented model: {e}")
            raise
    
    def _extract_model_info(self) -> None:
        """Extract information about the loaded model."""
        try:
            self.model_info['is_augmented'] = self.is_augmented_model
            
            if hasattr(self.model, 'named_steps'):
                steps = list(self.model.named_steps.keys())
                self.model_info['pipeline_steps'] = steps
                
                # Get vectorizer info
                if 'tfidf' in self.model.named_steps:
                    vectorizer = self.model.named_steps['tfidf']
                    self.model_info['vocab_size'] = len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else 0
                    self.model_info['max_features'] = getattr(vectorizer, 'max_features', None)
                    self.model_info['ngram_range'] = getattr(vectorizer, 'ngram_range', None)
                
                # Get classifier info
                if 'classifier' in self.model.named_steps:
                    classifier = self.model.named_steps['classifier']
                    self.model_info['classifier_type'] = type(classifier).__name__
                    if hasattr(classifier, 'classes_'):
                        self.model_info['classes'] = classifier.classes_.tolist()
                        self.model_info['n_classes'] = len(classifier.classes_)
            elif self.vectorizer:
                self.model_info['type'] = type(self.model).__name__
                self.model_info['vectorizer_type'] = type(self.vectorizer).__name__
                if hasattr(self.vectorizer, 'vocabulary_'):
                    self.model_info['vocab_size'] = len(self.vectorizer.vocabulary_)
                if hasattr(self.model, 'classes_'):
                    self.model_info['classes'] = self.model.classes_.tolist()
                    self.model_info['n_classes'] = len(self.model.classes_)
            else:
                self.model_info['type'] = type(self.model).__name__
        except Exception as e:
            logger.warning(f"Could not extract model info: {e}")
            self.model_info = {'error': str(e)}
    
    def predict(self, text: str) -> Tuple[str, float, List[dict]]:
        """Predict sentiment for given text with enhanced Amazon reviews model support."""
        if self.use_fallback:
            logger.debug("Using fallback analyzer for prediction")
            return self.fallback_analyzer.predict(text)
        
        if not self.is_loaded:
            logger.error("Model not loaded, falling back to TextBlob")
            return self.fallback_analyzer.predict(text)
        
        try:
            # Preprocess text
            processed_text = self.preprocessor.preprocess(text)
            if not processed_text.strip():
                logger.warning("Text preprocessing resulted in empty string")
                processed_text = text.lower()
            
            # Handle Amazon reviews model (Pipeline format)
            if hasattr(self.model, 'named_steps') and 'tfidf' in self.model.named_steps:
                prediction = self.model.predict([processed_text])[0]
                probabilities = self.model.predict_proba([processed_text])[0]
                
                # Enhanced token extraction for Amazon reviews model
                tokens = self._get_amazon_model_tokens(processed_text, prediction, probabilities)
                
            elif self.vectorizer:
                # Separate vectorizer model
                text_vector = self.vectorizer.transform([processed_text])
                prediction = self.model.predict(text_vector)[0]
                probabilities = self.model.predict_proba(text_vector)[0]
                tokens = self._get_tokens_with_separate_vectorizer(processed_text, prediction, probabilities)
                
            else:
                # Direct model prediction
                prediction = self.model.predict([processed_text])[0]
                probabilities = self.model.predict_proba([processed_text])[0]
                tokens = self._get_important_tokens(processed_text, prediction, probabilities)
            
            # Map prediction to label and get confidence
            label, score = self._map_prediction_to_label(prediction, probabilities)
            
            logger.debug(f"Prediction: {label} (score: {score:.3f}) for text: '{text[:50]}...'")
            return label, float(score), tokens
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            logger.info("Falling back to TextBlob analyzer")
            return self.fallback_analyzer.predict(text)

    def _get_amazon_model_tokens(self, processed_text: str, prediction: str, probabilities: np.ndarray) -> List[dict]:
        """Enhanced token extraction specifically for Amazon reviews TF-IDF model."""
        try:
            vectorizer = self.model.named_steps['tfidf']
            classifier = self.model.named_steps['clf']
            
            # Transform text to get feature vector
            text_vector = vectorizer.transform([processed_text])
            
            # Get feature names and coefficients
            feature_names = vectorizer.get_feature_names_out()
            
            # Get coefficients for the predicted class
            if hasattr(classifier, 'classes_'):
                class_idx = list(classifier.classes_).index(prediction) if prediction in classifier.classes_ else 0
                coefficients = classifier.coef_[class_idx] if len(classifier.coef_.shape) > 1 else classifier.coef_[0]
            else:
                coefficients = classifier.coef_[0]
            
            # Calculate feature contributions
            text_features = text_vector.toarray()[0]
            non_zero_indices = np.where(text_features > 0)[0]
            
            token_weights = []
            for idx in non_zero_indices:
                if idx < len(feature_names) and idx < len(coefficients):
                    # Feature importance = coefficient * tf-idf weight
                    importance = float(coefficients[idx] * text_features[idx])
                    token_weights.append({
                        "term": str(feature_names[idx]),
                        "weight": round(importance, 4)
                    })
            
            # Sort by absolute importance and return top 10
            token_weights.sort(key=lambda x: abs(x["weight"]), reverse=True)
            return token_weights[:10] if token_weights else self._get_simple_tokens(processed_text)
            
        except Exception as e:
            logger.error(f"Error extracting Amazon model tokens: {e}")
            return self._get_simple_tokens(processed_text)
    
    def _map_prediction_to_label(self, prediction: int, probabilities: np.ndarray) -> Tuple[str, float]:
        """Map model prediction to sentiment label with confidence score."""
        try:
            # Get the confidence score (max probability)
            score = float(np.max(probabilities))
            
            if self.is_augmented_model:
                # Augmented models are better at detecting positive sentiment
                # Use slightly different thresholds
                if hasattr(self.model, 'classes_'):
                    classes = self.model.classes_
                elif hasattr(self.model, 'named_steps') and hasattr(self.model.named_steps.get('classifier'), 'classes_'):
                    classes = self.model.named_steps['classifier'].classes_
                else:
                    classes = ['negative', 'neutral', 'positive']
                
                if len(classes) == 3:
                    # Three-class: use prediction directly
                    class_names = ['negative', 'neutral', 'positive']
                    if prediction < len(class_names):
                        label = class_names[prediction]
                    else:
                        label = "neutral"
                else:
                    # Binary or other
                    label_map = {0: "negative", 1: "positive"}
                    label = label_map.get(prediction, "neutral")
            else:
                # Original mapping logic
                if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                    classifier = self.model.named_steps['classifier']
                    if hasattr(classifier, 'classes_'):
                        classes = classifier.classes_
                        
                        if len(classes) == 2:
                            label = "positive" if prediction == 1 else "negative"
                        elif len(classes) == 3:
                            class_to_label = {0: "negative", 1: "neutral", 2: "positive"}
                            label = class_to_label.get(prediction, "neutral")
                        else:
                            label = "neutral"
                    else:
                        label_map = {0: "negative", 1: "neutral", 2: "positive"}
                        label = label_map.get(prediction, "neutral")
                else:
                    label_map = {0: "negative", 1: "neutral", 2: "positive"}
                    label = label_map.get(prediction, "neutral")
            
            return label, score
            
        except Exception as e:
            logger.error(f"Error mapping prediction: {e}")
            return "neutral", 0.5
    
    def _get_important_tokens(self, processed_text: str, prediction: int, probabilities: np.ndarray) -> List[dict]:
        """Extract important tokens and their weights with enhanced feature extraction."""
        try:
            if self.vectorizer:
                return self._get_tokens_with_separate_vectorizer(processed_text, prediction, probabilities)
            
            # Get the vectorizer and classifier from the pipeline
            if not hasattr(self.model, 'named_steps'):
                return self._get_simple_tokens(processed_text)
            
            vectorizer = self.model.named_steps.get('tfidf')
            classifier = self.model.named_steps.get('classifier')
            
            if vectorizer is None or classifier is None:
                return self._get_simple_tokens(processed_text)
            
            # Transform text to get feature vector
            text_vector = vectorizer.transform([processed_text])
            
            # Get feature names
            if hasattr(vectorizer, 'get_feature_names_out'):
                feature_names = vectorizer.get_feature_names_out()
            elif hasattr(vectorizer, 'get_feature_names'):
                feature_names = vectorizer.get_feature_names()
            else:
                return self._get_simple_tokens(processed_text)
            
            # Get coefficients for feature importance
            if hasattr(classifier, 'coef_'):
                coefficients = self._get_coefficients_for_prediction(classifier, prediction)
            else:
                return self._get_simple_tokens(processed_text)
            
            # Get non-zero features (tokens present in text)
            text_features = text_vector.toarray()[0]
            non_zero_indices = np.where(text_features > 0)[0]
            
            # Create token weights
            token_weights = []
            for idx in non_zero_indices:
                if idx < len(feature_names) and idx < len(coefficients):
                    # Calculate importance as coefficient * tf-idf weight
                    importance = float(coefficients[idx] * text_features[idx])
                    token_weights.append({
                        "term": str(feature_names[idx]),
                        "weight": importance
                    })
            
            # Sort by absolute weight and return top tokens
            token_weights.sort(key=lambda x: abs(x["weight"]), reverse=True)
            
            # Ensure we have meaningful tokens
            top_tokens = token_weights[:8]
            if not top_tokens:
                return self._get_simple_tokens(processed_text)
            
            return top_tokens
            
        except Exception as e:
            logger.error(f"Error extracting token weights: {e}")
            return self._get_simple_tokens(processed_text)
    
    def _get_tokens_with_separate_vectorizer(self, processed_text: str, prediction: int, probabilities: np.ndarray) -> List[dict]:
        """Get tokens when using separate vectorizer and model."""
        try:
            # Transform text
            text_vector = self.vectorizer.transform([processed_text])
            
            # Get feature names
            if hasattr(self.vectorizer, 'get_feature_names_out'):
                feature_names = self.vectorizer.get_feature_names_out()
            elif hasattr(self.vectorizer, 'get_feature_names'):
                feature_names = self.vectorizer.get_feature_names()
            else:
                return self._get_simple_tokens(processed_text)
            
            # Get coefficients
            if hasattr(self.model, 'coef_'):
                coefficients = self._get_coefficients_for_prediction(self.model, prediction)
            else:
                return self._get_simple_tokens(processed_text)
            
            # Get non-zero features
            text_features = text_vector.toarray()[0]
            non_zero_indices = np.where(text_features > 0)[0]
            
            # Create token weights
            token_weights = []
            for idx in non_zero_indices:
                if idx < len(feature_names) and idx < len(coefficients):
                    importance = float(coefficients[idx] * text_features[idx])
                    token_weights.append({
                        "term": str(feature_names[idx]),
                        "weight": importance
                    })
            
            # Sort and return top tokens
            token_weights.sort(key=lambda x: abs(x["weight"]), reverse=True)
            return token_weights[:8] if token_weights else self._get_simple_tokens(processed_text)
            
        except Exception as e:
            logger.error(f"Error with separate vectorizer tokens: {e}")
            return self._get_simple_tokens(processed_text)
    
    def _get_coefficients_for_prediction(self, classifier, prediction: int) -> np.ndarray:
        """Get coefficients for the predicted class."""
        try:
            if len(classifier.coef_.shape) > 1:
                # Multi-class classification
                if prediction < classifier.coef_.shape[0]:
                    return classifier.coef_[prediction]
                else:
                    # Use the first class if prediction is out of bounds
                    return classifier.coef_[0]
            else:
                # Binary classification
                return classifier.coef_[0]
        except Exception as e:
            logger.error(f"Error getting coefficients: {e}")
            return np.array([])
    
    def _get_simple_tokens(self, processed_text: str) -> List[dict]:
        """Get simple tokens without weights as fallback."""
        try:
            tokens = processed_text.split()[:6]
            return [{"term": token, "weight": 0.1 * (i + 1)} for i, token in enumerate(tokens)]
        except Exception:
            return [{"term": "unknown", "weight": 0.1}]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "is_loaded": self.is_loaded,
            "use_fallback": self.use_fallback,
            "model_path": settings.model_path,
            "is_augmented": self.is_augmented_model,
            **self.model_info
        }
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float, List[dict]]]:
        """Predict sentiment for multiple texts efficiently."""
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch prediction for text '{text[:50]}...': {e}")
                # Add fallback result
                results.append(("neutral", 0.5, [{"term": "error", "weight": 0.0}]))
        return results


# Global model instance
sentiment_model = SentimentModel()
