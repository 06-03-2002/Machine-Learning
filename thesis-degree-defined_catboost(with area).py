# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 02:08:49 2025

@author: Asus
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import matplotlib.cm as cm
import joblib

# ===============================
# Paths
# ===============================
data_path = r"E:/URP/thesis/owid-co2-data.csv"
models_dir = os.path.join(os.path.dirname(data_path), "model-catboost2")
os.makedirs(models_dir, exist_ok=True)

# ===============================
# Load dataset
# ===============================
data = pd.read_csv(data_path, encoding='latin1')
data = data.dropna(subset=["gdp"]).reset_index(drop=True)

# Keep country names
country_names = data["country"].values

# Encode categorical features (except country)
for column in data.select_dtypes(include=['object']).columns:
    if column != "country":
        data[column] = data[column].astype(str)
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])

# ===============================
# Features and target
# ===============================
features = data.drop(["gdp", "country"], axis=1)
target = data["gdp"].values

# ===============================
# Handle missing features
# ===============================
imputer = SimpleImputer(strategy="mean")
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
x_train, x_test, y_train, y_test, country_train, country_test = train_test_split(
    features_scaled, target, country_names, test_size=0.1, random_state=42
)

# ===============================
# CatBoost Regression
# ===============================
catboost = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    subsample=0.8,
    random_seed=42,
    verbose=0
)
catboost.fit(x_train, y_train)
joblib.dump(catboost, os.path.join(models_dir, "catboost_model.joblib"))

# Predictions
y_train_pred = catboost.predict(x_train)
y_test_pred = catboost.predict(x_test)

# R² scores
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"R² Score (Train): {r2_train:.4f}")
print(f"R² Score (Test): {r2_test:.4f}")

# ===============================
# Scatter plots: GDP vs features colored by top 10 countries
# ===============================
country_counts = pd.Series(country_names).value_counts()
top_countries = country_counts.head(10).index
colors = cm.get_cmap("tab10", len(top_countries))
country_color_map = {country: colors(i) for i, country in enumerate(top_countries)}

for column_index, column in enumerate(features.columns):
    X_col = features_scaled[:, column_index]
    y = target

    plt.figure(figsize=(8, 6))

    # Scatter points colored by top 10 countries
    for country in top_countries:
        mask = country_names == country
        plt.scatter(
            X_col[mask],
            y[mask],
            label=country,
            alpha=0.7,
            color=country_color_map[country],
            edgecolor="k",
            s=50
        )

    # ----- CatBoost fitted curve -----
    X_mean = np.mean(features_scaled, axis=0)  # baseline (all features at mean)
    X_curve = np.tile(X_mean, (200, 1))        # 200 grid points for smooth line
    X_curve[:, column_index] = np.linspace(min(X_col), max(X_col), 200)

    y_curve = catboost.predict(X_curve)

    plt.plot(np.linspace(min(X_col), max(X_col), 200), y_curve,
             color="red", linewidth=2, label="CatBoost Fit")

    plt.xlabel(column)
    plt.ylabel("GDP")
    plt.title(f"GDP vs {column} (Top 10 countries)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()                     # Convert imputed features (NumPy) back to DataFrame for easier indexing
features_imputed_df = pd.DataFrame(features_imputed, columns=features.columns)

# ----- Plot again with RAW feature values (imputed) -----
for column_index, column in enumerate(features.columns):
    raw_col = features_imputed_df.iloc[:, column_index].values  # imputed raw feature

    # Build curve in raw space → then scale before predicting
    grid_raw = np.linspace(raw_col.min(), raw_col.max(), 200)
    X_mean_raw = features_imputed_df.mean().values              # baseline in raw space
    X_curve_raw = np.tile(X_mean_raw, (200, 1))
    X_curve_raw[:, column_index] = grid_raw

    # Transform raw grid into scaled space for prediction
    X_curve_scaled = scaler.transform(X_curve_raw)
    y_curve_raw = catboost.predict(X_curve_scaled)

    plt.figure(figsize=(8, 6))
    for country in top_countries:
        mask = country_names == country
        plt.scatter(
            raw_col[mask],
            target[mask],
            label=country,
            alpha=0.7,
            color=country_color_map[country],
            edgecolor="k",
            s=50
        )

    plt.plot(grid_raw, y_curve_raw, color="red", linewidth=2, label="catboost Fit (raw, imputed)")
    plt.xlabel(column + " (raw values)")
    plt.ylabel("GDP")
    plt.title(f"GDP vs {column} (Top 10 countries, raw scale, imputed)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

