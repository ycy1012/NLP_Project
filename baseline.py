import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Load the dataset as requested
dataset = load_dataset("yelp_review_full")

# 1. Data Extraction
# We convert the Hugging Face dataset objects to lists for Scikit-Learn
print("Extracting data...")
X_train = dataset['train']['text']
y_train = dataset['train']['label']

X_test = dataset['test']['text']
y_test = dataset['test']['label']

# 0 = 1 star, 4 = 5 stars. We will map them for clearer reporting later.
target_names = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']

# 2. Build the Pipeline
# TfidfVectorizer: Converts text to numerical vectors.

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, n_jobs=-1, C=1.0))
])

# 3. Train the Model
print("Training baseline model (this may take a few minutes)...")
pipeline.fit(X_train, y_train)

# 4. Evaluation
print("Evaluating on test set...")
y_pred = pipeline.predict(X_test)

# Calculate Metrics
acc = accuracy_score(y_test, y_pred)
print(f"Baseline Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 5. Confusion Matrix Visualization
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Baseline Model')
plt.show()
