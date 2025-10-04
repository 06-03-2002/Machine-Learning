# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 22:01:25 2025

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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Download stopwords (run once locally, then cached) ---
nltk.download("stopwords")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Define save folder (local PC path for Spyder) ---
output_folder = r"E:\URP\transportation\outputs-pytorch"
os.makedirs(output_folder, exist_ok=True)

# --- Load dataset ---
#df = pd.read_csv(r"E:\URP\transportation\Airline Occurences.csv")
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

# --- TF-IDF Vectorization (sparse) ---
vectorizer = TfidfVectorizer(max_features=20000)
X_train_vec = vectorizer.fit_transform(X_train)   # sparse csr_matrix
X_test_vec = vectorizer.transform(X_test)

# --- SparseDataset to avoid memory blow-up ---
class SparseDataset(Dataset):
    def __init__(self, X_sparse, y_array):
        self.X = X_sparse
        self.y = np.array(y_array, dtype=np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx].toarray().astype(np.float32).ravel()   # only one row to dense
        return torch.from_numpy(row), torch.tensor(self.y[idx], dtype=torch.long)

# --- Create DataLoaders ---
train_dataset = SparseDataset(X_train_vec, y_train)
test_dataset = SparseDataset(X_test_vec, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Define PyTorch Model ---
class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_dim = X_train_vec.shape[1]
hidden_dim = 256
num_classes = len(le.classes_)

model = TextClassifier(input_dim, hidden_dim, num_classes)

# --- Define Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# --- Evaluation ---
model.eval()
all_preds = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        outputs = model(batch_x)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())

print(classification_report(y_test, all_preds, target_names=le.classes_))

# --- Save model, vectorizer, and label encoder ---
torch.save(model.state_dict(), os.path.join(output_folder, "precaution_model_pytorch.pth"))
joblib.dump(vectorizer, os.path.join(output_folder, "vectorizer.pkl"))
joblib.dump(le, os.path.join(output_folder, "label_encoder.pkl"))

# --- Load new dataset ---
#new_df = pd.read_csv(r"E:\URP\transportation\airline occurence test1.csv")
new_df = pd.read_csv(r"E:\URP\transportation\Airline Occurences.csv\airline occurence test1.csv")

new_df['Report'] = new_df['Report'].astype(str).str.strip().str.lower().apply(preprocess_text)
new_df['Part Failure'] = new_df['Part Failure'].astype(str).str.strip().str.lower().apply(preprocess_text)
new_df['combined_text'] = new_df['Report'] + ' ' + new_df['Part Failure']

X_new_vec = vectorizer.transform(new_df['combined_text'])

# --- Predict on new dataset in batches ---
new_dataset = SparseDataset(X_new_vec, np.zeros(X_new_vec.shape[0]))
new_loader = DataLoader(new_dataset, batch_size=64, shuffle=False)

model = TextClassifier(input_dim, hidden_dim, num_classes)
model.load_state_dict(torch.load(os.path.join(output_folder, "precaution_model_pytorch.pth"), map_location="cpu"))
model.eval()

preds = []
with torch.no_grad():
    for batch_x, _ in new_loader:
        outputs = model(batch_x)
        _, batch_preds = torch.max(outputs, 1)
        preds.extend(batch_preds.cpu().numpy())

predicted_labels = le.inverse_transform(np.array(preds))

# --- Save predictions ---
new_df['Predicted Occurence Precautionary Procedures'] = predicted_labels
output_csv_path = os.path.join(output_folder, "predicted_output_pytorch.csv")
new_df.to_csv(output_csv_path, index=False)

print(f"✅ Model, vectorizer, label encoder, and predictions saved in: {output_folder}")
