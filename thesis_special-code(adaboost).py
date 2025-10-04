# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 19:57:27 2025

@author: Asus
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import joblib
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# ===============================
# Settings
# ===============================
data_path = r"E:/URP/thesis/train file-kuznet curve.csv"
models_dir = os.path.join(os.path.dirname(data_path), "models-ada")
os.makedirs(models_dir, exist_ok=True)

# ===============================
# Load dataset
# ===============================
data = pd.read_csv(data_path, encoding='latin1')
data = data.dropna(subset=["gdp"]).reset_index(drop=True)

# ===============================
# Encode categorical variables
# ===============================
categorical_mappings = {}
for column in data.select_dtypes(include=['object']).columns:
    data[column] = data[column].astype(str)
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    categorical_mappings[column] = dict(zip(le.transform(le.classes_), le.classes_))

# ===============================
# Features and target
# ===============================
features = data.drop("gdp", axis=1)
target = data["gdp"].values

# ===============================
# Handle missing values
# ===============================
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)
joblib.dump(imputer, os.path.join(models_dir, "imputer.joblib"))

# ===============================
# Standardize features
# ===============================
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)
joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

# ===============================
# Train-test split
# ===============================
x_train, x_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.1, random_state=42
)

# ===============================
# Non-linear regression using AdaBoost
# ===============================
base_dt = DecisionTreeRegressor(max_depth=4, random_state=42)
adaboost = AdaBoostRegressor(
    estimator=base_dt,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
adaboost.fit(x_train, y_train)
joblib.dump(adaboost, os.path.join(models_dir, "adaboost_model.joblib"))

# Predictions
y_train_pred = adaboost.predict(x_train)
y_test_pred = adaboost.predict(x_test)

# R² scores
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"R² Score (Train): {r2_train:.4f}")
print(f"R² Score (Test): {r2_test:.4f}")
import seaborn as sns

# ===============================
# Plot Actual vs Predicted
# ===============================
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.7, edgecolor="k")

# Add reference line (perfect prediction)
min_val = min(min(y_test), min(y_test_pred))
max_val = max(max(y_test), max(y_test_pred))
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Perfect Fit")

plt.xlabel("Actual GDP")
plt.ylabel("Predicted GDP")
plt.title(f"Actual vs Predicted GDP (R² = {r2_test:.4f})")
plt.legend()
plt.show()

# Model info
print("\nAdaBoost model parameters:")
print(adaboost.get_params())
# Number of fitted estimators (after training)
try:
    print("Number of fitted estimators:", len(adaboost.estimators_))
except Exception:
    print("Number of fitted estimators: (not available)")

# ===============================
# Scatter plots: features vs GDP
# ===============================
for col_index, col_name in enumerate(features.columns):
    plt.figure(figsize=(8,6))
    X_col = features_imputed[:, col_index]
    
    # Scatter all data points
    plt.scatter(X_col, target, alpha=0.6, color="blue", edgecolor="k")
    
    # Optional: prediction curve for this feature
    X_range = np.linspace(X_col.min(), X_col.max(), 200).reshape(-1, 1)
    
    # Keep all features fixed at mean, vary only this column
    X_fixed = np.mean(features_scaled, axis=0).reshape(1, -1).repeat(200, axis=0)
    # Standardize this single column values to match the scaling used in features_scaled
    # (we fit a scaler on the original single-column values to transform X_range to standardized values)
    X_fixed[:, col_index] = StandardScaler().fit(X_col.reshape(-1,1)).transform(X_range).flatten()
    y_range_pred = adaboost.predict(X_fixed)
    
    plt.plot(X_range, y_range_pred, color="red", linewidth=2, label="AdaBoost Fit")
    
    plt.xlabel(col_name)
    plt.ylabel("GDP")
    plt.title(f"GDP vs {col_name} (AdaBoost fit)")
    plt.legend()
    plt.show()
