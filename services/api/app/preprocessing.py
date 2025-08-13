import re
import string
from typing import List, Optional, Set
import nltk
import spacy
from textblob import TextBlob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load spaCy model (fallback to basic if not available)
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully")
except OSError:
    logger.warning("spaCy model 'en_core_web_sm' not found. Using basic preprocessing.")
    nlp = None

STOP_WORDS = set(stopwords.words('english'))

# Additional stopwords for social media and reviews
REVIEW_STOPWORDS = {
    'product', 'item', 'purchase', 'buy', 'bought', 'order', 'ordered',
    'amazon', 'delivery', 'shipping', 'price', 'cost', 'money', 'worth',
    'like', 'follow', 'followme', 'please', 'thanks', 'thank'
}

STOP_WORDS.update(REVIEW_STOPWORDS)


class TextPreprocessor:
    """Enhanced text preprocessing pipeline for sentiment analysis."""
    
    def __init__(self, use_spacy: bool = True, preserve_negations: bool = True):
        self.use_spacy = use_spacy and nlp is not None
        self.preserve_negations = preserve_negations
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'noone', 'none', 'not',
            'havent', 'hasnt', 'hadnt', 'cant', 'couldnt', 'shouldnt', 'wont',
            'wouldnt', 'dont', 'doesnt', 'didnt', 'isnt', 'arent', 'aint'
        }
        
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better handling of social media content."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs (more comprehensive)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove user mentions and hashtags (keep the text part)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '.', text)
        
        # Remove numbers (optional, might want to keep for some contexts)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def handle_contractions(self, text: str) -> str:
        """Handle contractions and negations more comprehensively."""
        contractions = {
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "won't": "will not",
            "can't": "cannot",
            "shan't": "shall not",
            "y'all": "you all",
            "let's": "let us"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation while preserving important sentiment indicators."""
        # Handle contractions first
        text = self.handle_contractions(text)
        
        # Preserve some punctuation that might be important for sentiment
        # Convert multiple exclamation marks to single
        text = re.sub(r'!+', ' EXCLAMATION ', text)
        text = re.sub(r'\?+', ' QUESTION ', text)
        
        # Remove other punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Restore sentiment indicators
        text = text.replace('EXCLAMATION', '!')
        text = text.replace('QUESTION', '?')
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Enhanced tokenization and lemmatization."""
        if self.use_spacy:
            try:
                doc = nlp(text)
                tokens = []
                for token in doc:
                    if not token.is_space and not token.is_punct:
                        # Use lemma for better normalization
                        lemma = token.lemma_.lower()
                        if lemma != '-PRON-':  # Handle pronouns properly
                            tokens.append(lemma)
                        else:
                            tokens.append(token.text.lower())
                return tokens
            except Exception as e:
                logger.warning(f"spaCy processing failed: {e}, falling back to NLTK")
        
        # Fallback to NLTK tokenization
        try:
            tokens = word_tokenize(text.lower())
            return [token for token in tokens if token.isalpha()]
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {e}, using simple split")
            return text.lower().split()
    
    def handle_negations(self, tokens: List[str]) -> List[str]:
        """Handle negations by marking negated words."""
        if not self.preserve_negations:
            return tokens
        
        negated_tokens = []
        negate = False
        
        for token in tokens:
            if token in self.negation_words:
                negate = True
                negated_tokens.append(token)
            elif negate and token not in STOP_WORDS:
                # Mark negated word
                negated_tokens.append(f"NOT_{token}")
                negate = False
            else:
                negated_tokens.append(token)
                if token in ['.', '!', '?', ',', ';']:
                    negate = False
        
        return negated_tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords while preserving important sentiment words."""
        # Don't remove negation words and sentiment-bearing words
        important_words = self.negation_words.union({
            'very', 'really', 'quite', 'extremely', 'absolutely', 'totally',
            'completely', 'utterly', 'highly', 'incredibly', 'amazingly'
        })
        
        filtered_tokens = []
        for token in tokens:
            # Keep important words, non-stopwords, and words longer than 2 characters
            if (token in important_words or 
                token not in STOP_WORDS or 
                token.startswith('NOT_') or
                len(token) > 2):
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def preprocess(self, text: str) -> str:
        """Complete preprocessing pipeline with enhanced error handling."""
        try:
            if not text or not text.strip():
                return ""
            
            # Clean text
            text = self.clean_text(text)
            if not text:
                return ""
            
            # Remove punctuation (with contraction handling)
            text = self.remove_punctuation(text)
            
            # Tokenize and lemmatize
            tokens = self.tokenize_and_lemmatize(text)
            if not tokens:
                return text.lower()  # Fallback to simple lowercase
            
            # Handle negations
            tokens = self.handle_negations(tokens)
            
            # Remove stopwords
            tokens = self.remove_stopwords(tokens)
            
            # Join back to string
            result = ' '.join(tokens)
            return result if result else text.lower()
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            # Return simple cleaned version as fallback
            return re.sub(r'[^\w\s]', '', text.lower())


class FallbackSentimentAnalyzer:
    """Enhanced fallback sentiment analyzer using TextBlob with better accuracy."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor(use_spacy=False)
    
    def predict(self, text: str) -> tuple[str, float, List[dict]]:
        """Predict sentiment using TextBlob with enhanced logic."""
        try:
            # Use both original and preprocessed text for better accuracy
            blob = TextBlob(text)
            processed_text = self.preprocessor.preprocess(text)
            processed_blob = TextBlob(processed_text) if processed_text else blob
            
            # Get polarity from both versions
            original_polarity = blob.sentiment.polarity
            processed_polarity = processed_blob.sentiment.polarity if processed_text else original_polarity
            
            # Average the polarities for better accuracy
            polarity = (original_polarity + processed_polarity) / 2
            
            # Enhanced mapping with better thresholds
            if polarity > 0.15:
                label = "positive"
                score = min(0.6 + polarity * 0.4, 0.95)
            elif polarity < -0.15:
                label = "negative"
                score = min(0.6 + abs(polarity) * 0.4, 0.95)
            else:
                label = "neutral"
                score = 0.55 + abs(polarity) * 0.1
            
            # Generate enhanced token weights
            tokens = self._extract_meaningful_tokens(text, processed_text, polarity)
            
            return label, score, tokens
            
        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            return "neutral", 0.5, [{"term": "unknown", "weight": 0.0}]
    
    def _extract_meaningful_tokens(self, original_text: str, processed_text: str, overall_polarity: float) -> List[dict]:
        """Extract meaningful tokens with sentiment weights."""
        try:
            # Get tokens from processed text
            tokens = processed_text.split() if processed_text else original_text.lower().split()
            
            # Limit to reasonable number
            tokens = tokens[:8]
            
            token_weights = []
            for token in tokens:
                try:
                    # Get individual word sentiment
                    word_blob = TextBlob(token)
                    word_polarity = word_blob.sentiment.polarity
                    
                    # If word has no sentiment, use context-based weight
                    if abs(word_polarity) < 0.1:
                        word_polarity = overall_polarity * 0.3
                    
                    token_weights.append({
                        "term": token,
                        "weight": float(word_polarity)
                    })
                except Exception:
                    # Fallback weight
                    token_weights.append({
                        "term": token,
                        "weight": overall_polarity * 0.2
                    })
            
            # Sort by absolute weight
            token_weights.sort(key=lambda x: abs(x["weight"]), reverse=True)
            
            # Ensure we have at least one token
            if not token_weights:
                token_weights = [{"term": "text", "weight": overall_polarity}]
            
            return token_weights
            
        except Exception as e:
            logger.error(f"Token extraction failed: {e}")
            return [{"term": "unknown", "weight": 0.0}]
