import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def fetch_and_analyze_augmentation_data():
    """Fetch and analyze the positive sentiment augmentation dataset."""
    
    # Fetch the CSV data
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/sentiview_positive_augmentation-p97ZxfaOsW99rQ7E3WhuEbFrCH8LNb.csv"
    
    print("Fetching augmentation dataset...")
    response = requests.get(url)
    
    if response.status_code == 200:
        # Save locally for analysis
        with open('positive_augmentation.csv', 'wb') as f:
            f.write(response.content)
        
        # Load and analyze
        df = pd.read_csv('positive_augmentation.csv')
        
        print(f"\n=== Positive Augmentation Dataset Analysis ===")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Basic statistics
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        
        print(f"\nSource distribution:")
        print(df['source'].value_counts())
        
        print(f"\nSplit distribution:")
        print(df['split'].value_counts())
        
        # Text length analysis
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        print(f"\nText statistics:")
        print(f"Average text length: {df['text_length'].mean():.1f} characters")
        print(f"Average word count: {df['word_count'].mean():.1f} words")
        print(f"Min/Max text length: {df['text_length'].min()}/{df['text_length'].max()}")
        
        # Sample texts
        print(f"\nSample positive texts:")
        for i, text in enumerate(df['text'].head(5)):
            print(f"{i+1}. {text}")
        
        # Common words analysis
        all_text = ' '.join(df['text'].str.lower())
        words = re.findall(r'\b[a-zA-Z]+\b', all_text)
        common_words = Counter(words).most_common(20)
        
        print(f"\nMost common words:")
        for word, count in common_words:
            print(f"  {word}: {count}")
        
        # Check for emojis and special characters
        emoji_count = sum(1 for text in df['text'] if any(ord(char) > 127 for char in text))
        print(f"\nTexts with emojis/special chars: {emoji_count}/{len(df)} ({emoji_count/len(df)*100:.1f}%)")
        
        return df
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None

if __name__ == "__main__":
    df = fetch_and_analyze_augmentation_data()
