#!/usr/bin/env python3
"""
Training script for sentiment analysis model.
Downloads IMDB dataset and trains TF-IDF + LogisticRegression model.
"""

import os
import argparse
import joblib
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import requests
import tarfile
import pandas as pd


def download_imdb_dataset(data_dir: str = "./data"):
    """Download and extract IMDB movie reviews dataset."""
    os.makedirs(data_dir, exist_ok=True)
    
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    
    if not os.path.exists(tar_path):
        print("Downloading IMDB dataset...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download completed.")
    
    # Extract dataset
    extract_dir = os.path.join(data_dir, "aclImdb")
    if not os.path.exists(extract_dir):
        print("Extracting dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        print("Extraction completed.")
    
    return extract_dir


def load_imdb_data(data_dir: str):
    """Load IMDB dataset from extracted files."""
    texts = []
    labels = []
    
    # Load training data
    for sentiment in ['pos', 'neg']:
        sentiment_dir = os.path.join(data_dir, 'train', sentiment)
        label = 1 if sentiment == 'pos' else 0
        
        for filename in os.listdir(sentiment_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(sentiment_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label)
    
    return texts, labels


def create_simple_dataset():
    """Create a simple dataset for testing when IMDB is not available."""
    print("Creating simple test dataset...")
    
    positive_texts = [
        "I love this movie! It's absolutely amazing and wonderful.",
        "This is the best film I've ever seen. Fantastic acting and great story.",
        "Excellent movie with outstanding performances. Highly recommended!",
        "Amazing cinematography and brilliant direction. A masterpiece!",
        "I enjoyed every minute of this film. Truly spectacular and engaging.",
    ] * 100  # Repeat to have more samples
    
    negative_texts = [
        "This movie is terrible and boring. I hate it completely.",
        "Worst film ever made. Awful acting and horrible story.",
        "I regret watching this movie. It's absolutely dreadful and disappointing.",
        "Terrible direction and poor performances. A complete waste of time.",
        "This film is boring and uninteresting. I couldn't wait for it to end.",
    ] * 100  # Repeat to have more samples
    
    neutral_texts = [
        "This movie is okay. Nothing special but not terrible either.",
        "An average film with decent acting. It's watchable but forgettable.",
        "The movie has its moments but overall it's just mediocre.",
        "It's an ordinary film. Some parts are good, others not so much.",
        "This movie is fine. Not great, not bad, just somewhere in between.",
    ] * 100  # Repeat to have more samples
    
    texts = positive_texts + negative_texts + neutral_texts
    labels = [2] * len(positive_texts) + [0] * len(negative_texts) + [1] * len(neutral_texts)
    
    return texts, labels


def train_model(texts, labels, use_cv=False):
    """Train TF-IDF + LogisticRegression model."""
    print(f"Training on {len(texts)} samples...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        ))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation if requested
    if use_cv:
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(pipeline, texts, labels, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument("--use-imdb", action="store_true", help="Use IMDB dataset (requires download)")
    parser.add_argument("--cv", action="store_true", help="Use cross-validation")
    parser.add_argument("--output", default="./models/sentiment_lr_tfidf.pkl", help="Output model path")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load data
    if args.use_imdb:
        try:
            data_dir = download_imdb_dataset()
            texts, labels = load_imdb_data(data_dir)
        except Exception as e:
            print(f"Error loading IMDB dataset: {e}")
            print("Falling back to simple dataset...")
            texts, labels = create_simple_dataset()
    else:
        texts, labels = create_simple_dataset()
    
    # Train model
    model = train_model(texts, labels, use_cv=args.cv)
    
    # Save model
    print(f"Saving model to {args.output}")
    joblib.dump(model, args.output)
    print("Model saved successfully!")
    
    # Test the saved model
    print("\nTesting saved model...")
    loaded_model = joblib.load(args.output)
    test_texts = [
        "I love this amazing product!",
        "This is terrible and awful.",
        "It's okay, nothing special."
    ]
    
    for text in test_texts:
        prediction = loaded_model.predict([text])[0]
        probabilities = loaded_model.predict_proba([text])[0]
        print(f"Text: '{text}'")
        print(f"Prediction: {prediction}, Probabilities: {probabilities}")
        print()


if __name__ == "__main__":
    main()
