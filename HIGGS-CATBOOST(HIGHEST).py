# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 14:05:40 2025

@author: Asus
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
import joblib
import warnings

warnings.filterwarnings("ignore")

# --- Load dataset ---
df = pd.read_csv("E:/LHC-CERN/Z_boson-2.csv")
print("Data loaded successfully.")

# --- Label encode categorical features ---
categorical_features = [
    # none for this dataset
]

# Dictionary to store label encoders for each column
label_encoders = {}

for col in categorical_features + ["class"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for future use

# --- Define features and target ---
X = df.drop("class", axis=1)
y = df["class"]

# --- Optional: Apply MinMaxScaler ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# --- Split train-test ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42
)

# --- Train CatBoost ---
catboost = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=100
)
catboost.fit(X_train, y_train)

# --- Predict ---
y_pred = catboost.predict(X_test)

# --- Evaluate ---
print(classification_report(y_test, y_pred))

# --- Save the trained CatBoost model ---
catboost.save_model("E:/LHC-CERN/catboost_model-higgs-2.cbm")

# --- Save the label encoders ---
joblib.dump(label_encoders, "E:/LHC-CERN/label_encoders-catboost-higgs-2.pkl")

# --- Save the scaler ---
joblib.dump(scaler, "E:/LHC-CERN/minmax_scaler-catboost-higgs-2.pkl")

print("✅ CatBoost model, label encoders, and scaler saved successfully.")

# --- 1. Plot classification report as heatmap ---
import numpy as np

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).iloc[:-1, :-1]  # Exclude avg/total row

plt.figure(figsize=(10, 6))
sns.heatmap(report_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Classification Report Heatmap (CatBoost)")
plt.show()

# --- 2. Scatter plots instead of bar plots ---
df_raw = df.copy()
for col, le in label_encoders.items():
    df_raw[col] = le.inverse_transform(df[col])

features_to_plot = [
    "pt1", "eta1", "phi1", "Q1",
    "pt2", "eta2", "phi2", "Q2"
]

for feature in features_to_plot:
    plt.figure(figsize=(12, 6))
    sns.stripplot(
        x=feature, y="class", data=df_raw,
        jitter=True, palette="Set2", alpha=0.7, size=5
    )
    plt.title(f"class vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
