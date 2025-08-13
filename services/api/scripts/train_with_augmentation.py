import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
from preprocessing import TextPreprocessor

def fetch_augmentation_data():
    """Fetch the positive sentiment augmentation dataset."""
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/sentiview_positive_augmentation-p97ZxfaOsW99rQ7E3WhuEbFrCH8LNb.csv"
    
    print("Fetching positive augmentation dataset...")
    response = requests.get(url)
    
    if response.status_code == 200:
        df = pd.read_csv(pd.io.common.StringIO(response.text))
        print(f"Successfully loaded {len(df)} augmentation samples")
        return df
    else:
        print(f"Failed to fetch augmentation data. Status code: {response.status_code}")
        return None

def create_base_training_data():
    """Create base training data with balanced sentiment examples."""
    # Base training examples - you can expand this with your existing dataset
    base_data = [
        # Positive examples
        ("I love this product! It's amazing!", "positive"),
        ("Great quality and fast shipping", "positive"),
        ("Excellent customer service", "positive"),
        ("This is fantastic, highly recommend", "positive"),
        ("Perfect! Exactly what I needed", "positive"),
        ("Outstanding performance", "positive"),
        ("Very satisfied with my purchase", "positive"),
        ("Brilliant design and functionality", "positive"),
        
        # Negative examples
        ("This is terrible, waste of money", "negative"),
        ("Poor quality, broke immediately", "negative"),
        ("Worst customer service ever", "negative"),
        ("Complete disappointment", "negative"),
        ("Don't buy this, it's awful", "negative"),
        ("Horrible experience", "negative"),
        ("Defective product, very upset", "negative"),
        ("Regret buying this", "negative"),
        
        # Neutral examples
        ("The product is okay", "neutral"),
        ("It works as expected", "neutral"),
        ("Average quality for the price", "neutral"),
        ("Nothing special but functional", "neutral"),
        ("Standard product", "neutral"),
        ("It's fine, not great not bad", "neutral"),
        ("Decent but could be better", "neutral"),
        ("Acceptable quality", "neutral"),
    ]
    
    return pd.DataFrame(base_data, columns=['text', 'label'])

def train_enhanced_model():
    """Train sentiment model with positive augmentation data."""
    
    # Create base training data
    base_df = create_base_training_data()
    print(f"Base training data: {len(base_df)} samples")
    print("Base label distribution:")
    print(base_df['label'].value_counts())
    
    # Fetch augmentation data
    aug_df = fetch_augmentation_data()
    
    if aug_df is not None:
        # Combine datasets
        # Only use training split from augmentation data
        aug_train = aug_df[aug_df['split'] == 'train'].copy()
        combined_df = pd.concat([base_df, aug_train[['text', 'label']]], ignore_index=True)
        
        print(f"\nAfter augmentation: {len(combined_df)} samples")
        print("Combined label distribution:")
        print(combined_df['label'].value_counts())
    else:
        print("Using base data only (augmentation failed)")
        combined_df = base_df
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess texts
    print("\nPreprocessing texts...")
    processed_texts = []
    for text in combined_df['text']:
        processed = preprocessor.preprocess_text(text)
        processed_texts.append(processed)
    
    combined_df['processed_text'] = processed_texts
    
    # Prepare features and labels
    X = combined_df['processed_text']
    y = combined_df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Vectorize text
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    print("Training model...")
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train_vec, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    train_score = model.score(X_train_vec, y_train)
    test_score = model.score(X_test_vec, y_test)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Detailed evaluation
    y_pred = model.predict(X_test_vec)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model and vectorizer
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{model_dir}/sentiment_model_augmented_{timestamp}.joblib"
    vectorizer_path = f"{model_dir}/vectorizer_augmented_{timestamp}.joblib"
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    # Test with sample predictions
    print("\nSample predictions:")
    test_texts = [
        "This is great!",
        "Not bad, could be better",
        "Terrible product, don't buy",
        "Love it! Perfect quality",
        "It's okay, nothing special"
    ]
    
    for text in test_texts:
        processed = preprocessor.preprocess_text(text)
        vec = vectorizer.transform([processed])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        
        # Get probabilities for each class
        classes = model.classes_
        prob_dict = dict(zip(classes, proba))
        
        print(f"Text: '{text}' -> {pred} (probs: {prob_dict})")
    
    return model, vectorizer

if __name__ == "__main__":
    model, vectorizer = train_enhanced_model()
