# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 19:34:00 2025

@author: Asus
"""

# full_pytorch_pipeline.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------
# Config
# ------------------------------
train_path = r"E:/URP/transportation/flights_sample_3m.csv/Flight cancellation sample- training dataset.csv"
model_folder_name = "pytorch_trained_model"
random_state = 42
test_size = 0.1

# Training hyperparams
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 80
PATIENCE = 8            # early stopping patience on val loss
HIDDEN_UNITS = [128, 64]  # hidden layer sizes
WEIGHT_DECAY = 1e-5

# PDP / plotting
pdp_grid_points = 200
pdp_percentile_range = (1, 99)  # clip extremes

# Reproducibility
np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# Load data
# ------------------------------
data = pd.read_csv(train_path, encoding="latin1")

# ------------------------------
# Encode categoricals
# ------------------------------
encoders = {}
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = data[col].astype(str)
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# ------------------------------
# Features / Target
# ------------------------------
if "CANCELLED" not in data.columns:
    raise ValueError("Target column 'CANCELLED' not found.")

features = data.drop("CANCELLED", axis=1)
target = data["CANCELLED"].astype(int)  # binary labels 0/1

# ------------------------------
# Impute missing values (keep unscaled for plotting)
# ------------------------------
imputer = SimpleImputer(strategy="most_frequent")
X_imputed = imputer.fit_transform(features)   # unscaled imputed matrix
y = target.values.astype(np.float32)          # float32 for PyTorch BCE loss

# ------------------------------
# Scale features
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed).astype(np.float32)

# ------------------------------
# Train / Test split (stratified)
# ------------------------------
x_train, x_test, y_train, y_test, Xtrain_imputed, Xtest_imputed = train_test_split(
    X_scaled, y, X_imputed, test_size=test_size, random_state=random_state, stratify=y
)

# ------------------------------
# Prepare PyTorch datasets / loaders
# ------------------------------
train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.reshape(-1, 1)))
val_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test.reshape(-1, 1)))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ------------------------------
# Define feedforward classifier
# ------------------------------
class FFNN(nn.Module):
    def __init__(self, n_features, hidden_units):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_units:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.2))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # single logit output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits

n_features = x_train.shape[1]
model = FFNN(n_features=n_features, hidden_units=HIDDEN_UNITS).to(device)
print(model)

# ------------------------------
# Loss, optimizer
# ------------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

# ------------------------------
# Training loop with early stopping (monitor val loss)
# ------------------------------
best_val_loss = float("inf")
best_state = None
patience_counter = 0
history = {"train_loss": [], "val_loss": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_loss = np.mean(train_losses)

    # validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_losses.append(loss.item())
    val_loss = np.mean(val_losses)
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    # early stopping
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        best_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best val_loss={best_val_loss:.6f}")
            break

# restore best model
if best_state is not None:
    model.load_state_dict(best_state)

# ------------------------------
# Evaluate model: predictions & metrics
# ------------------------------
model.eval()
with torch.no_grad():
    X_test_tensor = torch.from_numpy(x_test).to(device)
    logits_test = model(X_test_tensor).cpu().numpy().reshape(-1)
    probs_test = 1 / (1 + np.exp(-logits_test))  # sigmoid
    y_pred_class = (probs_test >= 0.5).astype(int)

# Classification metrics
cm = confusion_matrix(y_test, y_pred_class)
acc = accuracy_score(y_test, y_pred_class)
prec = precision_score(y_test, y_pred_class, zero_division=0)
rec = recall_score(y_test, y_pred_class, zero_division=0)
f1 = f1_score(y_test, y_pred_class, zero_division=0)

# ROC AUC
fpr, tpr, _ = roc_curve(y_test, probs_test)
roc_auc = auc(fpr, tpr)

print("\n--- Classification Metrics on Test Set ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

# Plot ROC
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC={roc_auc:.3f})')
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Confusion matrix heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Cancelled (0)", "Cancelled (1)"],
            yticklabels=["Not Cancelled (0)", "Cancelled (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ------------------------------
# PDP-style plots for every feature (model probability vs feature)
# ------------------------------
median_unscaled = np.median(X_imputed, axis=0)

# We'll use the saved scaler to transform grid samples before passing to model.
for feat_idx, feat_name in enumerate(features.columns):
    # grid in unscaled domain
    low, high = np.percentile(X_imputed[:, feat_idx], pdp_percentile_range)
    grid = np.linspace(low, high, pdp_grid_points)

    # build samples where this feature varies and others at median_unscaled
    grid_samples_unscaled = np.tile(median_unscaled, (len(grid), 1))
    grid_samples_unscaled[:, feat_idx] = grid

    # scale
    grid_samples_scaled = scaler.transform(grid_samples_unscaled).astype(np.float32)

    # predict probabilities
    with torch.no_grad():
        grid_tensor = torch.from_numpy(grid_samples_scaled).to(device)
        logits_grid = model(grid_tensor).cpu().numpy().reshape(-1)
        probs_grid = 1 / (1 + np.exp(-logits_grid))

    # scatter of true test points (unscaled) and model predictions on test points
    x_test_feat_unscaled = Xtest_imputed[:, feat_idx]

    plt.figure(figsize=(8, 6))
    plt.scatter(x_test_feat_unscaled, y_test, alpha=0.4, label="True (test samples)")
    plt.plot(grid, probs_grid, color="red", lw=2, label="Model probability (PDP-style)")
    # model preds on test points
    plt.scatter(x_test_feat_unscaled, probs_test, color="green", alpha=0.5, s=20, label="Model prob (test samples)")
    plt.xlabel(f"{feat_name} (imputed, original scale)")
    plt.ylabel("Predicted probability of CANCELLED")
    plt.title(f"Model probability vs {feat_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------
# Bar plots: CANCELLED vs each feature (use unscaled imputed features)
# ------------------------------
plot_df = pd.DataFrame(X_imputed, columns=features.columns)
plot_df["CANCELLED"] = y

for col in features.columns:
    plt.figure(figsize=(8, 5))
    if pd.api.types.is_numeric_dtype(plot_df[col]):
        try:
            bins = pd.qcut(plot_df[col], q=10, duplicates="drop")
        except Exception:
            bins = pd.cut(plot_df[col], bins=10)
        grouped = plot_df.groupby([bins, "CANCELLED"]).size().reset_index(name="count")
        grouped["bin_str"] = grouped.iloc[:, 0].astype(str)
        sns.barplot(data=grouped, x="bin_str", y="count", hue="CANCELLED")
        plt.xlabel(f"{col} ")
        plt.xticks(rotation=45, ha="right")
    else:
        grouped = plot_df.groupby([col, "CANCELLED"]).size().reset_index(name="count")
        sns.barplot(data=grouped, x=col, y="count", hue="CANCELLED")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(col)
    plt.title(f"CANCELLED vs {col}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# ------------------------------
# Save artifacts
# ------------------------------
base_dir = os.path.dirname(train_path)
model_dir = os.path.join(base_dir, model_folder_name)
os.makedirs(model_dir, exist_ok=True)

model_file = os.path.join(model_dir, "pytorch_cancel_model_state.pth")
features_file = os.path.join(model_dir, "features_list.pkl")
encoders_file = os.path.join(model_dir, "label_encoders.pkl")
imputer_file = os.path.join(model_dir, "imputer.pkl")
scaler_file = os.path.join(model_dir, "scaler.pkl")

# Save model state_dict
torch.save(model.state_dict(), model_file)
# Save preprocessing artifacts
joblib.dump(list(features.columns), features_file)
joblib.dump(encoders, encoders_file)
joblib.dump(imputer, imputer_file)
joblib.dump(scaler, scaler_file)

print("\n📁 Saved:")
print(f" - PyTorch model state_dict: {model_file}")
print(f" - Features list: {features_file}")
print(f" - Label encoders: {encoders_file}")
print(f" - Imputer: {imputer_file}")
print(f" - Scaler: {scaler_file}")
