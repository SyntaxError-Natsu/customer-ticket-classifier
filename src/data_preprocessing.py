import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, file_path):
        try:
            df = pd.read_excel(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self, df):
        print("\nDATA EXPLORATION")
        print(f"Dataset shape: {df.shape}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        if 'issue_type' in df.columns:
            print(f"\nIssue type distribution:\n{df['issue_type'].value_counts()}")
        if 'urgency_level' in df.columns:
            print(f"\nUrgency level distribution:\n{df['urgency_level'].value_counts()}")
        if 'product' in df.columns:
            print(f"\nProduct distribution:\n{df['product'].value_counts()}")
        
        print("\nSAMPLE TICKETS")
        for i in range(min(3, len(df))):
            print(f"\nTicket {i+1}:")
            print(f"Text: {str(df.iloc[i]['ticket_text'])[:100]}...")
            if 'issue_type' in df.columns:
                print(f"Issue Type: {df.iloc[i]['issue_type']}")
            if 'urgency_level' in df.columns:
                print(f"Urgency: {df.iloc[i]['urgency_level']}")
            if 'product' in df.columns:
                print(f"Product: {df.iloc[i]['product']}")
        
        return df
    
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s\!\?\.]', ' ', text)
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        if not text:
            return []
        
        try:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) 
                     for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            return tokens
        except:
            words = text.split()
            return [word for word in words if len(word) > 2 and word not in self.stop_words]
    
    def get_sentiment(self, text):
        if not text:
            return 0
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0
    
    def preprocess_data(self, df):
        print("\nPREPROCESSING DATA")
        
        df = df.copy()
        
        df['ticket_text'] = df['ticket_text'].fillna('')
        if 'issue_type' in df.columns:
            df['issue_type'] = df['issue_type'].fillna('Unknown')
        if 'urgency_level' in df.columns:
            df['urgency_level'] = df['urgency_level'].fillna('Medium')
        if 'product' in df.columns:
            df['product'] = df['product'].fillna('Unknown')
        
        print("Cleaning text...")
        df['cleaned_text'] = df['ticket_text'].apply(self.clean_text)
        
        print("Adding text features...")
        df['text_length'] = df['cleaned_text'].apply(len)
        df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
        df['sentiment'] = df['cleaned_text'].apply(self.get_sentiment)
        
        print("Tokenizing...")
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))
        
        df['processed_text'] = df['processed_text'].replace('', 'empty text')
        
        print(f"Preprocessing completed! Final shape: {df.shape}")
        return df
