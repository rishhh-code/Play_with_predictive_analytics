# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# 2. Load and clean the dataset
dataset = pd.read_csv('data/grade.csv')

# Rename the misformatted column
dataset.rename(columns={'label;;;;': 'label'}, inplace=True)

# Clean the label column
dataset['label'] = dataset['label'].str.replace(';', '', regex=False).astype(int)

# 3. Define feature matrix and target
X = dataset[['grade1', 'grade2']]
y = dataset['label']

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Predict and evaluate
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model Evaluation")
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)
print("Accuracy Score:", accuracy)

# 7. Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/logi.pkl")
print("\n✅ Model saved as 'models/logi.pkl'")

# 8. Save feature names
with open("models/logi.txt", "w") as f:
    for feature in X.columns:
        f.write(feature + "\n")
print("✅ Features saved to 'models/logi.txt'")
