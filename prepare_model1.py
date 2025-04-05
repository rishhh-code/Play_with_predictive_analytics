import pandas as pd
import numpy as np
import re
import string
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("data/spam_ham_dataset.csv")  # Make sure your file is in 'data/' folder

# Rename and clean columns if needed
if 'label' not in df.columns:
    df.rename(columns={'label': 'label', 'text': 'text'}, inplace=True)

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Apply preprocessing
df['text'] = df['text'].astype(str).apply(preprocess_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Map ham/spam to 0/1

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Pipeline: TF-IDF + Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', MultinomialNB())
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("🔍 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(pipeline, df['text'], df['label'], cv=5)
print("✅ Cross-validation Accuracy:", np.mean(scores))

# Save model
joblib.dump(pipeline, "models/spam_detector.pkl")
print("✅ Model saved as 'models/spam_detector.pkl'")
