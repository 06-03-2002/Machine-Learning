# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 23:00:56 2025

@author: Asus
"""

# --- Import libraries ---
import numpy as np
import pandas as pd
import string
import os
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import lightgbm as lgb

# --- Download stopwords ---
nltk.download("stopwords")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Define save folder ---
output_folder = r"E:\URP\transportation\Airline Occurences.csv\outputs-lightgbm"
os.makedirs(output_folder, exist_ok=True)

# --- Load dataset ---
df = pd.read_csv(r"E:\URP\transportation\Airline Occurences.csv\Airline Occurences.csv")

# --- Clean text columns ---
df['Report'] = df['Report'].astype(str).str.strip().str.lower()
df['Part Failure'] = df['Part Failure'].astype(str).str.strip().str.lower()
df['Occurence Precautionary Procedures'] = df['Occurence Precautionary Procedures'].astype(str).str.strip().str.lower()

# --- Text preprocessing function ---
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

# --- Apply preprocessing ---
df['Report'] = df['Report'].apply(preprocess_text)
df['Part Failure'] = df['Part Failure'].apply(preprocess_text)

# --- Combine text features ---
df['combined_text'] = df['Report'] + ' ' + df['Part Failure']

# --- Prepare features and labels ---
X = df['combined_text']
y = df['Occurence Precautionary Procedures']

# --- Encode target labels ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# --- TF-IDF Vectorization ---
vectorizer = TfidfVectorizer(max_features=20000)  # limit features for speed/memory
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Train LightGBM classifier ---
model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=len(le.classes_),
    learning_rate=0.1,
    n_estimators=300,
    max_depth=-1,
    random_state=42
)

model.fit(X_train_vec, y_train)

# --- Evaluate model ---
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- Save model, vectorizer, and label encoder ---
joblib.dump(model, os.path.join(output_folder, "precaution_model_lightgbm.pkl"))
joblib.dump(vectorizer, os.path.join(output_folder, "vectorizer.pkl"))
joblib.dump(le, os.path.join(output_folder, "label_encoder.pkl"))

# --- Load new dataset ---
new_df = pd.read_csv(r"E:\URP\transportation\Airline Occurences.csv\airline occurence test1.csv")

# --- Preprocess text ---
new_df['Report'] = new_df['Report'].astype(str).str.strip().str.lower().apply(preprocess_text)
new_df['Part Failure'] = new_df['Part Failure'].astype(str).str.strip().str.lower().apply(preprocess_text)
new_df['combined_text'] = new_df['Report'] + ' ' + new_df['Part Failure']

# --- Load model, vectorizer, and label encoder ---
model = joblib.load(os.path.join(output_folder, "precaution_model_lightgbm.pkl"))
vectorizer = joblib.load(os.path.join(output_folder, "vectorizer.pkl"))
le = joblib.load(os.path.join(output_folder, "label_encoder.pkl"))

# --- Transform and predict ---
X_new = vectorizer.transform(new_df['combined_text'])
y_new_pred = model.predict(X_new)
predicted_labels = le.inverse_transform(y_new_pred)

# --- Save predictions ---
new_df['Predicted Occurence Precautionary Procedures'] = predicted_labels
output_csv_path = os.path.join(output_folder, "predicted_output_lightgbm.csv")
new_df.to_csv(output_csv_path, index=False)

print(f"✅ LightGBM model, label encoder, and CSV saved in: {output_folder}")
