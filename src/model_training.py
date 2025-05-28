import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.best_models = {}
        
    def balance_data(self, X, y):
        try:
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            print(f"Data balanced: {len(y)} -> {len(y_balanced)} samples")
            return X_balanced, y_balanced
        except:
            df_combined = pd.concat([X, pd.Series(y, name='target')], axis=1)
            classes = df_combined['target'].unique()
            max_size = df_combined['target'].value_counts().max()
            
            balanced_dfs = []
            for class_label in classes:
                class_df = df_combined[df_combined['target'] == class_label]
                class_df_upsampled = resample(class_df, 
                                             replace=True,
                                             n_samples=max_size,
                                             random_state=42)
                balanced_dfs.append(class_df_upsampled)
            
            balanced_df = pd.concat(balanced_dfs)
            X_balanced = balanced_df.drop('target', axis=1)
            y_balanced = balanced_df['target']
            
            print(f"Data balanced using upsampling: {len(y)} -> {len(y_balanced)} samples")
            return X_balanced, y_balanced
        
    def split_data(self, X, y_issue, y_urgency, test_size=0.2, random_state=42):
        X_train, X_test, y_issue_train, y_issue_test, y_urgency_train, y_urgency_test = train_test_split(
            X, y_issue, y_urgency, test_size=test_size, random_state=random_state, 
            stratify=y_issue if len(np.unique(y_issue)) > 1 else None
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_issue_train, y_issue_test, y_urgency_train, y_urgency_test
    
    def train_issue_classifier(self, X_train, y_train, X_test, y_test):
        print("\nTRAINING ISSUE TYPE CLASSIFIER")
        
        X_train_balanced, y_train_balanced = self.balance_data(X_train, y_train)
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, 
                random_state=42, 
                class_weight='balanced',
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=2000, 
                random_state=42, 
                class_weight='balanced',
                C=0.1,
                solver='liblinear'
            )
        }
        
        best_score = 0
        best_model = None
        best_name = None
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='accuracy')
            print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            model.fit(X_train_balanced, y_train_balanced)
            
            test_score = model.score(X_test, y_test)
            print(f"{name} Test Score: {test_score:.4f}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = model
                best_name = name
        
        print(f"\nBest Issue Classifier: {best_name} (Score: {best_score:.4f})")
        self.best_models['issue_classifier'] = best_model
        
        return best_model
    
    def train_urgency_classifier(self, X_train, y_train, X_test, y_test):
        print("\nTRAINING URGENCY LEVEL CLASSIFIER")
        
        X_train_balanced, y_train_balanced = self.balance_data(X_train, y_train)
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300, 
                random_state=42, 
                class_weight='balanced',
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=3000, 
                random_state=42, 
                class_weight='balanced',
                C=1.0,
                solver='liblinear'
            )
        }
        
        best_score = 0
        best_model = None
        best_name = None
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='accuracy')
            print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            model.fit(X_train_balanced, y_train_balanced)
            
            test_score = model.score(X_test, y_test)
            print(f"{name} Test Score: {test_score:.4f}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = model
                best_name = name
        
        print(f"\nBest Urgency Classifier: {best_name} (Score: {best_score:.4f})")
        self.best_models['urgency_classifier'] = best_model
        
        return best_model
    
    def evaluate_model(self, model, X_test, y_test, label_encoder, model_name):
        print(f"\n{model_name.upper()} EVALUATION")
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        target_names = label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return accuracy, y_pred
    
    def save_models(self, path='../models/'):
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.best_models.items():
            with open(f'{path}{name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        print(f"Models saved to {path}")
    
    def load_models(self, path='../models/'):
        for name in ['issue_classifier', 'urgency_classifier']:
            try:
                with open(f'{path}{name}.pkl', 'rb') as f:
                    self.best_models[name] = pickle.load(f)
            except FileNotFoundError:
                print(f"Model {name} not found at {path}")
        
        print(f"Models loaded from {path}")
