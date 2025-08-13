import os, pandas as pd, joblib, pathlib, requests, re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier

DATASET_URL = "https://drive.google.com/uc?export=download&id=1SERc309kmcEGhsqhuztIE_ZaqJGku-WQ"
MODEL_PATH_LR = "./models/sentiment_lr_tfidf_gdrive.pkl"
MODEL_PATH_SVM = "./models/sentiment_svm_tfidf_gdrive.pkl"
MODEL_PATH_ENSEMBLE = "./models/sentiment_ensemble_tfidf_gdrive.pkl"
TRAIN_SAMPLE = 50000

def setup_nltk():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

def create_short_positive_examples():
    """Create additional short positive training examples to improve classification"""
    short_positive_examples = [
        "very pleased", "love it", "excellent", "amazing", "fantastic", "awesome",
        "great product", "highly recommend", "perfect", "outstanding", "superb",
        "brilliant", "wonderful", "impressive", "top quality", "five stars",
        "best ever", "so good", "really good", "very good", "extremely good",
        "love this", "adore it", "absolutely love", "totally recommend", 
        "highly satisfied", "very happy", "extremely happy", "thrilled",
        "delighted", "pleased", "satisfied", "impressed", "blown away",
        "exceeded expectations", "worth it", "great value", "good buy",
        "smart purchase", "no regrets", "glad I bought", "perfect choice",
        "exactly what I wanted", "better than expected", "works perfectly",
        "high quality", "well made", "durable", "reliable", "solid",
        "fast delivery", "quick shipping", "arrived quickly", "prompt service",
        "great customer service", "helpful staff", "professional", "courteous"
    ]
    
    # Create DataFrame with these examples
    examples_df = pd.DataFrame({
        'text': short_positive_examples,
        'label': ['positive'] * len(short_positive_examples)
    })
    
    print(f"Created {len(short_positive_examples)} short positive training examples")
    return examples_df

def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation, remove stopwords, lemmatize"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation and numbers
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

def load_data():
    """Download and load the Google Drive dataset"""
    print("Downloading dataset from Google Drive...")
    
    # Download the CSV file
    response = requests.get(DATASET_URL)
    response.raise_for_status()
    
    # Save temporarily
    temp_path = "./data/temp_amazon_reviews.csv"
    pathlib.Path("data").mkdir(exist_ok=True)
    
    with open(temp_path, 'wb') as f:
        f.write(response.content)
    
    # Load and process the data
    df = pd.read_csv(temp_path)
    print(f"Loaded {len(df)} rows from dataset")
    
    if 'input' in df.columns and 'label' in df.columns:
        df = df[["input", "label"]].dropna()
        df = df.rename(columns={"input": "text"})
    else:
        raise ValueError("Expected columns 'input' and 'label' not found in dataset")
    
    df['label'] = df['label'].str.lower()
    df['label'] = df['label'].map({
        'positive': 'positive',
        'negative': 'negative', 
        'neutral': 'neutral'
    })
    
    df = df.dropna(subset=["label"]).drop_duplicates(subset=["text"])
    
    short_examples = create_short_positive_examples()
    df = pd.concat([df, short_examples], ignore_index=True)
    print(f"Added short positive examples. New dataset size: {len(df)}")
    
    print("Preprocessing text data...")
    df["text"] = df["text"].apply(preprocess_text)
    
    # Remove empty texts after preprocessing
    df = df[df["text"].str.len() > 0]
    
    if len(df) > TRAIN_SAMPLE:
        # Ensure we keep all short positive examples
        short_processed = df[df['text'].isin([preprocess_text(ex) for ex in short_examples['text']])]
        remaining_df = df[~df['text'].isin([preprocess_text(ex) for ex in short_examples['text']])]
        
        if len(remaining_df) > TRAIN_SAMPLE - len(short_processed):
            remaining_df = remaining_df.sample(TRAIN_SAMPLE - len(short_processed), random_state=7)
        
        df = pd.concat([remaining_df, short_processed], ignore_index=True)
        print(f"Sampled {TRAIN_SAMPLE} rows for training (including all short positive examples)")
    
    # Clean up temp file
    os.remove(temp_path)
    
    print(f"Final dataset size: {len(df)}")
    print("Label distribution:")
    print(df['label'].value_counts())
    
    return df[["text", "label"]]

def train():
    """Train multiple models with enhanced preprocessing"""
    setup_nltk()
    pathlib.Path("models").mkdir(exist_ok=True)
    
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], 
        test_size=0.2, stratify=data["label"], random_state=42
    )
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), 
        max_features=50000, 
        min_df=3,
        max_df=0.95,
        sublinear_tf=True
    )
    
    print("Training Logistic Regression model...")
    # Logistic Regression Pipeline
    lr_pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", LogisticRegression(
            max_iter=1000, 
            class_weight="balanced",
            C=1.0,
            random_state=42
        ))
    ])
    
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    print("Logistic Regression Results:")
    print(classification_report(y_test, y_pred_lr))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    
    print("\nTraining SVM model...")
    svm_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), 
            max_features=30000, 
            min_df=3,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", SVC(
            kernel='linear',
            class_weight="balanced",
            probability=True,  # Enable probability estimates for confidence
            random_state=42
        ))
    ])
    
    svm_pipe.fit(X_train, y_train)
    y_pred_svm = svm_pipe.predict(X_test)
    print("SVM Results:")
    print(classification_report(y_test, y_pred_svm))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    
    print("\nTraining Ensemble model...")
    ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)),
            ('svm', SVC(kernel='linear', class_weight="balanced", probability=True, random_state=42))
        ],
        voting='soft'  # Use probability-based voting
    )
    
    # Fit ensemble on vectorized data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    ensemble.fit(X_train_vec, y_train)
    y_pred_ensemble = ensemble.predict(X_test_vec)
    print("Ensemble Results:")
    print(classification_report(y_test, y_pred_ensemble))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}")
    
    models_info = {
        'vectorizer': vectorizer,
        'preprocessing': 'lowercase, remove punctuation/numbers, remove stopwords, lemmatize',
        'dataset_source': 'Google Drive Amazon Reviews + Short Positive Examples',
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    # Save individual models
    joblib.dump({**models_info, 'model': lr_pipe}, MODEL_PATH_LR)
    joblib.dump({**models_info, 'model': svm_pipe}, MODEL_PATH_SVM)
    joblib.dump({
        **models_info, 
        'model': ensemble,
        'vectorizer': vectorizer  # Ensemble needs separate vectorizer
    }, MODEL_PATH_ENSEMBLE)
    
    print(f"\nModels saved:")
    print(f"- Logistic Regression: {MODEL_PATH_LR}")
    print(f"- SVM: {MODEL_PATH_SVM}")
    print(f"- Ensemble: {MODEL_PATH_ENSEMBLE}")

if __name__ == "__main__":
    train()
