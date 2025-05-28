import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from entity_extraction import EntityExtractor
import warnings
warnings.filterwarnings('ignore')

class TicketClassifier:
    def __init__(self, models_path='../models/'):
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.entity_extractor = EntityExtractor()
        
        self.load_models(models_path)
    
    def load_models(self, models_path):
        try:
            self.feature_engineer.load_feature_objects(models_path)
            self.model_trainer.load_models(models_path)
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please train models first by running the main analysis notebook")
    
    def preprocess_single_ticket(self, ticket_text):
        df = pd.DataFrame({'ticket_text': [ticket_text]})
        
        df['cleaned_text'] = df['ticket_text'].apply(self.preprocessor.clean_text)
        df['text_length'] = df['cleaned_text'].apply(len)
        df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
        df['sentiment'] = df['cleaned_text'].apply(self.preprocessor.get_sentiment)
        df['tokens'] = df['cleaned_text'].apply(self.preprocessor.tokenize_and_lemmatize)
        df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))
        
        df['processed_text'] = df['processed_text'].replace('', 'empty text')
        
        return df
    
    def create_features_single_ticket(self, df):
        tfidf_features = self.feature_engineer.tfidf_vectorizer.transform(df['processed_text'])
        feature_names = self.feature_engineer.tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(), 
            columns=[f'tfidf_{name}' for name in feature_names]
        )
        
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
        
        X = pd.concat([tfidf_df, features_df], axis=1)
        
        for col in self.feature_engineer.tfidf_vectorizer.get_feature_names_out():
            tfidf_col = f'tfidf_{col}'
            if tfidf_col not in X.columns:
                X[tfidf_col] = 0
        
        X = X.fillna(0)
        
        numerical_cols = ['text_length', 'word_count', 'sentiment', 'caps_ratio']
        numerical_cols = [col for col in numerical_cols if col in X.columns]
        if numerical_cols:
            X[numerical_cols] = self.feature_engineer.scaler.transform(X[numerical_cols])
        
        return X
    
    def predict_ticket(self, ticket_text):
        try:
            df = self.preprocess_single_ticket(ticket_text)
            X = self.create_features_single_ticket(df)
            
            issue_pred_encoded = self.model_trainer.best_models['issue_classifier'].predict(X)[0]
            urgency_pred_encoded = self.model_trainer.best_models['urgency_classifier'].predict(X)[0]
            
            issue_type = self.feature_engineer.label_encoders['issue_type'].inverse_transform([issue_pred_encoded])[0]
            urgency_level = self.feature_engineer.label_encoders['urgency_level'].inverse_transform([urgency_pred_encoded])[0]
            
            entities = self.entity_extractor.extract_all_entities(ticket_text)
            
            issue_proba = self.model_trainer.best_models['issue_classifier'].predict_proba(X)[0]
            urgency_proba = self.model_trainer.best_models['urgency_classifier'].predict_proba(X)[0]
            
            issue_confidence = max(issue_proba)
            urgency_confidence = max(urgency_proba)
            
            result = {
                'predicted_issue_type': issue_type,
                'predicted_urgency_level': urgency_level,
                'issue_confidence': float(issue_confidence),
                'urgency_confidence': float(urgency_confidence),
                'extracted_entities': entities
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'predicted_issue_type': 'Unknown',
                'predicted_urgency_level': 'Medium',
                'issue_confidence': 0.0,
                'urgency_confidence': 0.0,
                'extracted_entities': {}
            }
