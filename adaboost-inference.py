# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 21:19:24 2025

@author: Asus
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# ------------------------------
# CONFIG
# ------------------------------
test_path = r"E:/URP/transportation/flights_sample_3m.csv/testing data flight cancellation.csv"

# Directory where training saved the models
base_model_dir = os.path.join(
    os.path.dirname(r"E:/URP/transportation/flights_sample_3m.csv/Flight cancellation sample- training dataset.csv"),
    "adaboost_trained_model"
)

# ------------------------------
# Load artifacts
# ------------------------------
ada_model_file = os.path.join(base_model_dir, "adaboost_cancel_model.pkl")
dt_model_file = os.path.join(base_model_dir, "decision_tree_cancel_model.pkl")
features_file = os.path.join(base_model_dir, "features_list.pkl")
encoders_file = os.path.join(base_model_dir, "label_encoders.pkl")
imputer_file = os.path.join(base_model_dir, "imputer.pkl")

# Check existence
for f in [ada_model_file, features_file, encoders_file, imputer_file]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Required file not found: {f}\nMake sure models were saved in: {base_model_dir}")

# Load objects
ada = joblib.load(ada_model_file)
dt = joblib.load(dt_model_file) if os.path.exists(dt_model_file) else None
train_features = joblib.load(features_file)
encoders = joblib.load(encoders_file)
imputer = joblib.load(imputer_file)

print("✅ Loaded models and preprocessing artifacts.")

# ------------------------------
# Load test CSV
# ------------------------------
test_df = pd.read_csv(test_path, encoding="latin1")
output_df = test_df.copy()  # copy for final output

# ------------------------------
# Prepare working dataframe
# ------------------------------
work = test_df.copy()

# Convert object cols to str (consistency)
for col in work.select_dtypes(include=["object"]).columns:
    work[col] = work[col].astype(str)

# ------------------------------
# Encode categorical columns using saved encoders
# ------------------------------
if isinstance(encoders, dict):
    for col, le in encoders.items():
        if col in work.columns:
            classes = list(le.classes_)
            mapping = {c: i for i, c in enumerate(classes)}
            work[col] = work[col].map(lambda v: mapping[v] if v in mapping else -1).astype(float)

# Encode any new object cols not seen in training
for col in work.select_dtypes(include=["object"]).columns:
    uniq = work[col].unique().tolist()
    mapping = {v: i for i, v in enumerate(uniq)}
    work[col] = work[col].map(lambda v: mapping.get(v, -1)).astype(float)

# ------------------------------
# Align with training features
# ------------------------------
aligned = work.reindex(columns=train_features)  # drop extras, add missing as NaN
aligned = aligned.apply(pd.to_numeric, errors="coerce")

# ------------------------------
# Impute missing values
# ------------------------------
X_test_prepared = imputer.transform(aligned)

# ------------------------------
# Predictions
# ------------------------------
# AdaBoost
pred_adb = ada.predict(X_test_prepared)
proba_adb = ada.predict_proba(X_test_prepared)[:, 1] if hasattr(ada, "predict_proba") else np.full(len(pred_adb), np.nan)

output_df["CANCELLED_PREDICTED_ADA"] = pred_adb
output_df["CANCELLED_PROB_ADA"] = proba_adb

# Decision Tree (if available)
if dt is not None:
    pred_dt = dt.predict(X_test_prepared)
    try:
        proba_dt = dt.predict_proba(X_test_prepared)[:, 1]
    except Exception:
        proba_dt = np.full(len(pred_dt), np.nan)
    output_df["CANCELLED_PREDICTED_DT"] = pred_dt
    output_df["CANCELLED_PROB_DT"] = proba_dt

# ------------------------------
# Save predictions
# ------------------------------
save_path = os.path.join(os.path.dirname(test_path), "predicted_output_with_models.csv")
output_df.to_csv(save_path, index=False)

print(f"✅ Predictions saved to: {save_path}")
