Customer Support Ticket Classifier
A comprehensive machine learning pipeline that classifies customer support tickets by issue type and urgency level, and extracts key entities using traditional NLP and ML techniques.

📋 Project Overview
This project implements a complete ticket classification system that:

Classifies tickets into 7 issue types (Billing Problem, Installation Issue, Product Defect, Account Access, General Inquiry, Wrong Item, Late Delivery)

Predicts urgency levels (Low, Medium, High) based on ticket content

Extracts entities including products, dates, order numbers, and complaint keywords

Provides an interactive web interface for real-time predictions and batch processing

🏗️ Project Structure
```
ticket_classifier/
├── data/
│   └── ai_dev_assignment_tickets_complex_1000.xlsx    # Input dataset
├── models/                                            # Saved models and objects
│   ├── processed_data.csv                            # Cleaned dataset
│   ├── issue_classifier.pkl                         # Trained issue classifier
│   ├── urgency_classifier.pkl                       # Trained urgency classifier
│   ├── tfidf_vectorizer.pkl                         # TF-IDF vectorizer
│   ├── label_encoders.pkl                           # Label encoders
│   └── scaler.pkl                                   # Feature scaler
├── src/
│   ├── data_preprocessing.py                        # Text cleaning and preprocessing
│   ├── feature_engineering.py                      # Feature creation and encoding
│   ├── model_training.py                           # Model training and evaluation
│   ├── entity_extraction.py                        # Rule-based entity extraction
│   ├── ticket_classifier.py                        # Main prediction pipeline
│   └── gradio_app.py                              # Web interface
├── notebooks/
│   └── main_analysis.ipynb                         # Complete pipeline execution
├── requirements.txt                                 # Python dependencies
└── README.md                                       # This documentation
```

🚀 Quick Start Guide
1. Environment Setup
```
# Create project directory
mkdir ticket_classifier
cd ticket_classifier

# Create virtual environment (recommended)
python -m venv ticket_env
# On Windows:
ticket_env\Scripts\activate
# On macOS/Linux:
source ticket_env/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn nltk textblob matplotlib seaborn openpyxl wordcloud plotly gradio imbalanced-learn jupyter
```

2. Download NLTK Data
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

3. Project Setup
Create the folder structure as shown above

Place your Excel file in the data/ folder

Copy all source files to their respective directories

4. Training the Models
Option A: Using Jupyter Notebook (Recommended)
```
cd notebooks
jupyter notebook
# Open main_analysis.ipynb and run all cells
```
Option B: Using Python Scripts
```
cd src
python data_preprocessing.py
python feature_engineering.py
python model_training.py
```

5. Running the Web Interface
```
cd src
python gradio_app.py
```

The interface will be available at http://localhost:7860

🔧 Technical Implementation
Data Preprocessing (data_preprocessing.py)
Key Features:

Text Normalization: Converts to lowercase, removes special characters

Tokenization: Uses NLTK word tokenizer

Lemmatization: Reduces words to base forms

Stopword Removal: Filters common English stopwords

Sentiment Analysis: Calculates polarity using TextBlob

Missing Data Handling: Fills null values appropriately

Design Choices:

Preserved punctuation (!, ?, .) as they indicate urgency/questions

Used lemmatization over stemming for better word representation

Added sentiment as it correlates with urgency levels

Feature Engineering (feature_engineering.py)
Text Features:

TF-IDF Vectorization: 2000 features, 1-2 grams, removes rare/common terms

Text Statistics: Length, word count, sentiment score

Engineered Features:

Complaint Keywords: 22 problem-related terms (broken, error, etc.)

Urgency Indicators: High/medium/low urgency keyword counts

Domain-Specific: Billing and installation keyword indicators

Punctuation Features: Question marks, exclamation marks, caps ratio

Length Categories: Short (<10 words) and long (>50 words) indicators

Feature Selection Justification:

TF-IDF captures semantic content effectively

Keyword features provide domain-specific signals

Punctuation features indicate emotional state and urgency

Length features help distinguish query types

Multi-Task Learning (model_training.py)
Models Tested:

Random Forest Classifier

Logistic Regression

Training Strategy:

Class Balancing: SMOTE oversampling for minority classes

Cross-Validation: 5-fold stratified CV for robust evaluation

Model Selection: Best performing model on test set

Performance Optimization:

Hyperparameter tuning for Random Forest (n_estimators, max_depth)

Class weights for handling imbalanced data

Feature scaling for numerical features

Entity Extraction (entity_extraction.py)
Rule-Based Approach:

Product Detection: Pattern matching against product list

Date Extraction: Multiple regex patterns for various date formats

Order Numbers: Patterns for #12345, order 12345, etc.

Complaint Keywords: Comprehensive problem indicator list

Urgency Indicators: Emergency and priority-related terms

Support Contact: Detection of previous interaction mentions

Pattern Examples:
```
# Date patterns
r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b'  # "03 March"
r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'  # "03/15/2024"

# Order patterns
r'#\d{5,6}'  # "#29224"
r'order\s*#?\s*\d{5,6}'  # "order 29224"
```
Integration (ticket_classifier.py)
Main Pipeline:

Text Preprocessing: Clean and tokenize input

Feature Generation: Create TF-IDF and engineered features

Prediction: Use trained models for classification

Entity Extraction: Apply rule-based extraction

Output Formatting: Return structured JSON result

Error Handling:

Graceful degradation for missing models

Default predictions for edge cases

Comprehensive exception handling

📊 Model Performance
Issue Type Classification
Best Model: Random Forest

Accuracy: 60-80% (varies by class balance)

Cross-Validation: 5-fold stratified

Classes: 7 issue types

Urgency Level Classification
Best Model: Random Forest

Accuracy: 55-75% (varies by class balance)

Cross-Validation: 5-fold stratified

Classes: 3 urgency levels (Low, Medium, High)

Entity Extraction
Approach: Rule-based pattern matching

Precision: High for structured entities (order numbers, dates)

Coverage: Comprehensive keyword lists for domain-specific terms

🎯 Key Design Decisions
1. Traditional ML vs Deep Learning
Choice: Traditional ML (Random Forest, Logistic Regression)
Rationale:

Interpretable results

Faster training and inference

Sufficient performance for the task

Lower computational requirements

2. Separate Models vs Multi-Output
Choice: Separate models for issue type and urgency
Rationale:

Different feature importance for each task

Independent optimization

Better handling of class imbalances

3. Rule-Based vs ML Entity Extraction
Choice: Rule-based pattern matching
Rationale:

High precision for structured data

Interpretable and debuggable

No additional training data required

Easy to extend with new patterns

4. Feature Engineering Strategy
Choice: Combination of TF-IDF and engineered features
Rationale:

TF-IDF captures semantic content

Engineered features add domain knowledge

Balanced approach between automation and expertise

🌐 Gradio Web Interface
Features
Single Ticket Prediction: Real-time classification and entity extraction

Batch Processing: Upload CSV/Excel files for multiple tickets

Confidence Scores: Model certainty indicators

JSON Export: Structured output for integration

Example Tickets: Pre-loaded test cases

Design Principles
Theme Compatibility: Works in both light and dark modes

Responsive Layout: Equal-height columns for proper alignment

Clear Visualization: Color-coded results with high contrast

User-Friendly: Intuitive interface with helpful examples

📈 Evaluation Results
Confusion Matrices
Generated automatically during training showing:

True vs predicted classifications

Class-wise performance

Misclassification patterns

Classification Reports
Detailed metrics including:

Precision, Recall, F1-score per class

Macro and weighted averages

Support (number of samples per class)

Visualizations
Issue type and urgency distributions

Text length and sentiment distributions

Feature importance plots

Word count vs sentiment scatter plots

⚠️ Limitations
Model Limitations
Performance Ceiling: Traditional ML may not capture complex semantic relationships

Class Imbalance: Some issue types have limited training data

Domain Specificity: Trained on specific product/service context

Language Support: English only

Entity Extraction Limitations
Rule Dependency: Relies on predefined patterns

Context Ignorance: May extract irrelevant matches

Product List: Limited to predefined product names

Date Ambiguity: May misinterpret relative dates

System Limitations
Static Models: Require retraining for new patterns

Scalability: Single-threaded processing

Memory Usage: Loads all models in memory

🔮 Future Improvements
Short Term
Enhanced Patterns: More comprehensive regex patterns

Product Expansion: Larger product vocabulary

Performance Tuning: Hyperparameter optimization

Error Analysis: Detailed failure case analysis

Long Term
Deep Learning: BERT/RoBERTa for better text understanding

Named Entity Recognition: Advanced NER models

Active Learning: Continuous improvement with user feedback

Multilingual Support: Support for multiple languages

Real-time Learning: Online model updates

📝 Dependencies
```
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
scikit-learn>=1.3.0    # Machine learning
nltk>=3.8.1            # Natural language processing
textblob>=0.17.1       # Sentiment analysis
matplotlib>=3.7.0      # Plotting
seaborn>=0.12.0        # Statistical visualization
openpyxl>=3.1.0        # Excel file handling
wordcloud>=1.9.0       # Word cloud generation
plotly>=5.15.0         # Interactive plots
gradio>=4.0.0          # Web interface
imbalanced-learn>=0.11.0  # SMOTE sampling
jupyter>=1.0.0         # Notebook environment
```
🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/improvement)

Make changes and test thoroughly

Commit with clear messages (git commit -m 'Add new feature')

Push to branch (git push origin feature/improvement)

Create a Pull Request

📄 License
This project is created for educational purposes as part of an AI internship assignment. Feel free to use and modify for learning and development purposes.

🙏 Acknowledgments
NLTK Team: For comprehensive NLP tools

Scikit-learn Community: For robust ML algorithms

Gradio Team: For intuitive web interface framework

Assignment Provider: For the challenging and educational task
