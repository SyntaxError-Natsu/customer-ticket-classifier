import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def create_text_features(self, df):
        print("Creating TF-IDF features...")
        
        processed_texts = df['processed_text'].fillna('empty text')
        processed_texts = processed_texts.replace('', 'empty text')
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(), 
            columns=[f'tfidf_{name}' for name in feature_names]
        )
        
        print(f"Created {tfidf_df.shape[1]} TF-IDF features")
        return tfidf_df
    
    def create_additional_features(self, df):
        print("Creating additional features...")
        
        features_df = pd.DataFrame()
        
        features_df['text_length'] = df['text_length']
        features_df['word_count'] = df['word_count']
        features_df['sentiment'] = df['sentiment']
        
        complaint_keywords = [
            'broken', 'error', 'problem', 'issue', 'fault', 'defect',
            'late', 'delay', 'slow', 'wrong', 'incorrect', 'missing',
            'not working', 'malfunction', 'stuck', 'damaged', 'failed',
            'crashed', 'freeze', 'hang', 'bug', 'glitch', 'stopped'
        ]
        
        urgency_keywords = {
            'high_urgency': ['urgent', 'asap', 'immediately', 'critical', 'emergency', 'now', 'help', 'please'],
            'medium_urgency': ['soon', 'important', 'needed', 'required', 'should'],
            'low_urgency': ['when possible', 'eventually', 'sometime', 'later', 'whenever']
        }
        
        billing_keywords = ['payment', 'bill', 'charge', 'refund', 'money', 'cost', 'price', 'invoice']
        install_keywords = ['install', 'setup', 'configure', 'connect', 'activation']
        
        features_df['complaint_count'] = df['ticket_text'].apply(
            lambda x: sum(1 for keyword in complaint_keywords if keyword in str(x).lower())
        )
        
        for level, keywords in urgency_keywords.items():
            features_df[level] = df['ticket_text'].apply(
                lambda x: sum(1 for keyword in keywords if keyword in str(x).lower())
            )
        
        features_df['billing_indicators'] = df['ticket_text'].apply(
            lambda x: sum(1 for keyword in billing_keywords if keyword in str(x).lower())
        )
        
        features_df['install_indicators'] = df['ticket_text'].apply(
            lambda x: sum(1 for keyword in install_keywords if keyword in str(x).lower())
        )
        
        features_df['question_marks'] = df['ticket_text'].str.count('\?')
        features_df['exclamation_marks'] = df['ticket_text'].str.count('!')
        
        features_df['caps_ratio'] = df['ticket_text'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
        
        features_df['number_count'] = df['ticket_text'].str.count(r'\d+')
        
        features_df['is_short'] = (df['word_count'] < 10).astype(int)
        features_df['is_long'] = (df['word_count'] > 50).astype(int)
        
        print(f"Created {features_df.shape[1]} additional features")
        return features_df
    
    def encode_labels(self, df):
        print("Encoding labels...")
        
        if 'issue_type' in df.columns:
            self.label_encoders['issue_type'] = LabelEncoder()
            df['issue_type_encoded'] = self.label_encoders['issue_type'].fit_transform(df['issue_type'])
            print(f"Issue types: {self.label_encoders['issue_type'].classes_}")
        
        if 'urgency_level' in df.columns:
            self.label_encoders['urgency_level'] = LabelEncoder()
            df['urgency_level_encoded'] = self.label_encoders['urgency_level'].fit_transform(df['urgency_level'])
            print(f"Urgency levels: {self.label_encoders['urgency_level'].classes_}")
        
        return df
    
    def create_all_features(self, df):
        print("\nFEATURE ENGINEERING")
        
        df = self.encode_labels(df)
        tfidf_features = self.create_text_features(df)
        additional_features = self.create_additional_features(df)
        
        X = pd.concat([tfidf_features, additional_features], axis=1)
        X = X.fillna(0)
        
        numerical_cols = ['text_length', 'word_count', 'sentiment', 'caps_ratio']
        numerical_cols = [col for col in numerical_cols if col in X.columns]
        if numerical_cols:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        print(f"Feature engineering completed! Feature matrix shape: {X.shape}")
        return X, df
    
    def save_feature_objects(self, path='../models/'):
        os.makedirs(path, exist_ok=True)
        
        with open(f'{path}tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        with open(f'{path}label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
            
        with open(f'{path}scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Feature objects saved to {path}")
    
    def load_feature_objects(self, path='../models/'):
        with open(f'{path}tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        with open(f'{path}label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
            
        with open(f'{path}scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
