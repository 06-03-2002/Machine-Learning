# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 21:40:35 2025

@author: Asus
"""

# inference_pytorch_save_csv.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)
import torch
import torch.nn as nn

# -------- USER CONFIG --------

# Path to the test CSV the user will provide
test_path = r"E:/URP/transportation/flights_sample_3m.csv/testing data flight cancellation.csv"

# Directory where training saved artifacts (adjust if needed)
model_dir = r"E:/URP/transportation/flights_sample_3m.csv/pytorch_trained_model"

# Model architecture hyperparameters (must match training)
HIDDEN_UNITS = [128, 64]
DROP_P = 0.2
THRESHOLD = 0.5

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# -----------------------------


# FFNN must match training
class FFNN(nn.Module):
    def __init__(self, n_features, hidden_units):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_units:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(DROP_P))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # binary output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------
# Load artifacts
# ------------------------------
print("Loading artifacts...")
state_dict = torch.load(os.path.join(model_dir, "pytorch_cancel_model_state.pth"), map_location="cpu")
features_list = joblib.load(os.path.join(model_dir, "features_list.pkl"))
encoders = joblib.load(os.path.join(model_dir, "label_encoders.pkl"))
imputer = joblib.load(os.path.join(model_dir, "imputer.pkl"))
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
print("Artifacts loaded.")


# ------------------------------
# Load test CSV
# ------------------------------
test_df = pd.read_csv(test_path, encoding="latin1")
orig_test_df = test_df.copy()  # keep original for saving


# ------------------------------
# Preprocess test set
# ------------------------------
for c in test_df.select_dtypes(include=["object"]).columns:
    test_df[c] = test_df[c].astype(str)

aligned = pd.DataFrame(index=test_df.index)
for feat in features_list:
    if feat in test_df.columns:
        aligned[feat] = test_df[feat]
    else:
        aligned[feat] = np.nan  # fill missing features

# Encode categoricals
if isinstance(encoders, dict):
    for col, le in encoders.items():
        if col in aligned.columns:
            classes = list(le.classes_)
            mapping = {c: i for i, c in enumerate(classes)}
            aligned[col] = aligned[col].map(lambda v: mapping[v] if v in mapping else -1).astype(float)

for col in aligned.select_dtypes(include=["object"]).columns:
    aligned[col] = aligned[col].astype(str).map(lambda v: -1).astype(float)

aligned = aligned.apply(pd.to_numeric, errors="coerce")

# Impute + scale
X_imputed = imputer.transform(aligned)
X_scaled = scaler.transform(X_imputed).astype(np.float32)


# ------------------------------
# Load model and predict
# ------------------------------
n_features = X_scaled.shape[1]
model = FFNN(n_features=n_features, hidden_units=HIDDEN_UNITS).to(device)
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    X_tensor = torch.from_numpy(X_scaled).to(device)
    logits = model(X_tensor).cpu().numpy().reshape(-1)
    probs = 1 / (1 + np.exp(-logits))
    pred_class = (probs >= THRESHOLD).astype(int)


# ------------------------------
# Save results as CSV
# ------------------------------
out_df = orig_test_df.copy()
out_df["PRED_PROB"] = probs
out_df["PRED_CLASS"] = pred_class

# Build output path
test_folder = os.path.dirname(test_path) or "."
test_name_pytorch = os.path.splitext(os.path.basename(test_path))[0]
out_path = os.path.join(test_folder, f"{test_name_pytorch}_with_predictions.csv")

# Save CSV
out_df.to_csv(out_path, index=False)
print(f"\n✅ Predictions with input data saved to:\n{out_path}")


# ------------------------------
# Compute metrics if labels exist
# ------------------------------
if "CANCELLED" in test_df.columns:
    y_true = test_df["CANCELLED"].astype(int).values
    cm = confusion_matrix(y_true, pred_class)
    acc = accuracy_score(y_true, pred_class)
    prec = precision_score(y_true, pred_class, zero_division=0)
    rec = recall_score(y_true, pred_class, zero_division=0)
    f1 = f1_score(y_true, pred_class, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    print("\n--- Test Metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Cancelled (0)", "Cancelled (1)"],
                yticklabels=["Not Cancelled (0)", "Cancelled (1)"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Save metrics to CSV too
    metrics_path = os.path.join(test_folder, f"{test_name_pytorch}_metrics.csv")
    pd.DataFrame([{
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc
    }]).to_csv(metrics_path, index=False)
    print(f"📊 Metrics saved to: {metrics_path}")
else:
    print("\n⚠️ No 'CANCELLED' column found — only predictions saved.")
