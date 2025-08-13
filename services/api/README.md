# Sentiview API

FastAPI backend service for sentiment analysis with machine learning capabilities and VADER fusion.

## 🚀 Features

- **Enhanced Sentiment Analysis**: ML model + VADER fusion with confidence scores
- **Multiple Model Support**: Amazon reviews, augmented models, and custom training
- **Twitter Integration**: Real-time Twitter stream analysis with rate limiting
- **Batch Processing**: Efficient batch text analysis
- **Token Analysis**: Feature importance and interpretability
- **RESTful API**: Comprehensive endpoints with OpenAPI documentation

## 🛠️ Installation

### Local Development

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (automatic on first run)
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
\`\`\`

### Docker

\`\`\`bash
# Build image
docker build -t sentiview-api .

# Run container
docker run -p 8000:8000 sentiview-api
\`\`\`

## 🔧 Configuration

### Environment Variables

\`\`\`bash
# Required
ENVIRONMENT=development
LOG_LEVEL=info
MODEL_PATH=./models/sentiment_lr_tfidf.pkl

# Twitter API (Optional)
TWITTER_BEARER_TOKEN=your_bearer_token
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
\`\`\`

### Model Training Options

#### 1. Amazon Reviews Model (Recommended)
\`\`\`bash
# Place Amazon_Unlocked_Mobile.csv in data/ directory
python scripts/train_amazon_reviews.py
\`\`\`
Creates `sentiment_lr_tfidf.pkl` with TF-IDF + LogisticRegression.

#### 2. Augmented Model (Best Performance)
\`\`\`bash
# Train with positive augmentation data
python scripts/train_with_augmentation.py
\`\`\`
Creates augmented models with improved positive sentiment detection.

#### 3. Custom Model
\`\`\`bash
# Train with your own data
python scripts/train.py
\`\`\`

**Model Priority**: Augmented > Amazon Reviews > Custom > TextBlob Fallback

## 📖 API Endpoints

### Enhanced Sentiment Analysis

**POST /predict**
\`\`\`json
{
  "text": "I love this product!"
}
\`\`\`

Response with VADER fusion:
\`\`\`json
{
  "label": "positive",
  "score": 0.8750,
  "tokens": [
    {"term": "love", "weight": 0.4521},
    {"term": "product", "weight": 0.2134}
  ],
  "probs": {
    "model": 0.82,
    "vader": 0.95,
    "mix": 0.8750
  }
}
\`\`\`

**POST /predict/batch**
\`\`\`json
{
  "texts": ["Great product!", "Terrible service", "It's okay"]
}
\`\`\`

### Twitter Analysis

**POST /tweets/analyze**
\`\`\`json
{
  "query": "python programming",
  "count": 50
}
\`\`\`

Response:
\`\`\`json
{
  "query": "python programming",
  "total_analyzed": 50,
  "sentiment_distribution": {
    "positive": 32,
    "negative": 8,
    "neutral": 10
  },
  "average_confidence": 0.78,
  "processing_time": 2.34,
  "tweets": [...]
}
\`\`\`

### Utility Endpoints

- **GET /health** - Health check with model status
- **GET /model/info** - Detailed model information
- **GET /model/stats** - Model performance statistics

## 🧪 Testing

\`\`\`bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test categories
pytest tests/test_ml_model.py -v
pytest tests/test_twitter_integration.py -v
\`\`\`

## 🔍 Model Architecture

### VADER Fusion Algorithm
\`\`\`
1. ML Model Prediction → p_pos (0-1)
2. VADER Compound Score → p_vader (-1 to 1 → 0 to 1)
3. Fusion: p_mix = 0.7 * p_pos + 0.3 * p_vader
4. Classification:
   - p_mix ≥ 0.60 → "positive"
   - p_mix ≤ 0.40 → "negative"
   - else → "neutral"
\`\`\`

### Model Types Supported

| Model Type | Description | Use Case |
|------------|-------------|----------|
| **Augmented** | Enhanced with positive examples | Best overall performance |
| **Amazon Reviews** | TF-IDF + LogisticRegression | Good baseline, fast training |
| **Custom** | User-provided training data | Domain-specific analysis |
| **TextBlob Fallback** | Rule-based sentiment | Always available backup |

## 🚀 Performance Optimizations

- **Model Caching**: In-memory model persistence
- **Batch Processing**: Vectorized operations for multiple texts
- **Async Processing**: Non-blocking I/O operations
- **Feature Extraction**: Efficient token importance calculation
- **Rate Limiting**: Twitter API compliance

## 🛡️ Security & Reliability

- **Input Validation**: Pydantic model validation
- **Error Handling**: Graceful fallbacks at every level
- **CORS Configuration**: Secure cross-origin requests
- **Logging**: Comprehensive request/error tracking
- **Health Monitoring**: Model status and performance metrics

## 📊 Architecture

\`\`\`
services/api/
├── app/
│   ├── main.py              # FastAPI app with VADER fusion
│   ├── ml_model.py          # Enhanced ML pipeline
│   ├── preprocessing.py     # Text preprocessing
│   ├── twitter_service.py   # Twitter integration
│   └── config.py           # Configuration management
├── scripts/
│   ├── train_amazon_reviews.py    # Amazon dataset training
│   ├── train_with_augmentation.py # Augmented model training
│   └── train.py                   # Basic model training
├── tests/                   # Comprehensive test suite
└── models/                  # Trained model storage
\`\`\`

## 🤝 Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Write tests for new features
4. Update documentation
5. Test with multiple model types

## 📄 API Documentation

Interactive API documentation available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔧 Troubleshooting

### Common Issues

**Model Not Loading**
- Check `MODEL_PATH` environment variable
- Ensure model file exists in `models/` directory
- Check logs for specific error messages

**Twitter API Errors**
- Verify Twitter API credentials
- Check rate limiting status
- Ensure bearer token is valid

**NLTK Data Missing**
- Run: `python -c "import nltk; nltk.download('vader_lexicon')"`
- Check internet connection for downloads
- Verify NLTK data directory permissions
