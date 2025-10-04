# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 22:17:35 2025

@author: Asus
"""

# ===============================
# Imports
# ===============================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import joblib
from catboost import CatBoostRegressor

# ===============================
# Settings
# ===============================
data_path = r"E:/URP/thesis/train file-kuznet curve.csv"
models_dir = os.path.join(os.path.dirname(data_path), "models-catboost")
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
# Non-linear regression using CatBoost
# ===============================
cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    random_seed=42,
    verbose=100
)
cat_model.fit(x_train, y_train)
cat_model.save_model(os.path.join(models_dir, "catboost_model.cbm"))

# ===============================
# Predictions
# ===============================
y_train_pred = cat_model.predict(x_train)
y_test_pred = cat_model.predict(x_test)

# R² scores
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"R² Score (Train): {r2_train:.4f}")
print(f"R² Score (Test): {r2_test:.4f}")

# ===============================
# Plot Actual vs Predicted
# ===============================
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.7, edgecolor="k")

min_val = min(min(y_test), min(y_test_pred))
max_val = max(max(y_test), max(y_test_pred))
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Perfect Fit")

plt.xlabel("Actual GDP")
plt.ylabel("Predicted GDP")
plt.title(f"Actual vs Predicted GDP (R² = {r2_test:.4f})")
plt.legend()
plt.show()

# ===============================
# Model info
# ===============================
print("\nCatBoost model parameters:")
print(cat_model.get_params())
print("Number of trees:", cat_model.tree_count_)

# ===============================
# Scatter plots: features vs GDP
# ===============================
for col_index, col_name in enumerate(features.columns):
    plt.figure(figsize=(8,6))
    X_col = features_imputed[:, col_index]

    # Scatter actual values
    plt.scatter(X_col, target, alpha=0.6, color="blue", edgecolor="k")

    # Prediction curve for this feature
    X_range = np.linspace(X_col.min(), X_col.max(), 200).reshape(-1, 1)
    X_fixed = np.mean(features_scaled, axis=0).reshape(1, -1).repeat(200, axis=0)
    X_fixed[:, col_index] = StandardScaler().fit(X_col.reshape(-1,1)).transform(X_range).flatten()

    y_range_pred = cat_model.predict(X_fixed)

    plt.plot(X_range, y_range_pred, color="red", linewidth=2, label="CatBoost Fit")

    plt.xlabel(col_name)
    plt.ylabel("GDP")
    plt.title(f"GDP vs {col_name} (CatBoost fit)")
    plt.legend()
    plt.show()
