import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

print("Loading and preprocessing data...")
df = pd.read_csv(os.path.join('data', 'spam.csv'))

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

df = df.dropna()
df['label'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
df['message'] = df['Message']

print(f"\nClass distribution:")
print(df['label'].value_counts())
print(f"Ham (not spam): {df[df['label']==0].shape[0]} messages")
print(f"Spam: {df[df['label']==1].shape[0]} messages")

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} messages")
print(f"Test set size: {X_test.shape[0]} messages")

print("\nConverting text to numerical features...")
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, max_features=5000)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_features.shape}")

print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_features, y_train)

y_train_pred = model.predict(X_train_features)
y_test_pred = model.predict(X_test_features)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_precision = precision_score(y_train, y_train_pred)
test_precision = precision_score(y_test, y_test_pred)
train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE METRICS")
print("="*50)
print(f"\n{'Metric':<15} {'Training':<15} {'Test':<15}")
print("-"*45)
print(f"{'Accuracy':<15} {train_accuracy:.4f}{'':8} {test_accuracy:.4f}")
print(f"{'Precision':<15} {train_precision:.4f}{'':8} {test_precision:.4f}")
print(f"{'Recall':<15} {train_recall:.4f}{'':8} {test_recall:.4f}")
print(f"{'F1-Score':<15} {train_f1:.4f}{'':8} {test_f1:.4f}")
print("="*50)

print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

os.makedirs('saved_model', exist_ok=True)
joblib.dump(model, 'saved_model/spam_model.pkl')
joblib.dump(vectorizer, 'saved_model/vectorizer.pkl')

print("\n✅ Model and vectorizer saved successfully in 'saved_model' folder!")

def predict_message(message):
    message_features = vectorizer.transform([message])
    prediction = model.predict(message_features)[0]
    probability = model.predict_proba(message_features)[0]
    
    if prediction == 1:
        result = "SPAM"
        confidence = probability[1] * 100
    else:
        result = "NOT SPAM"
        confidence = probability[0] * 100
    
    return result, confidence

test_message = "Congratulations! You've won a free iPhone. Click here to claim your prize!"
result, confidence = predict_message(test_message)
print(f"\nTest Prediction:")
print(f"Message: {test_message}")
print(f"Result: {result}")
print(f"Confidence: {confidence:.2f}%")