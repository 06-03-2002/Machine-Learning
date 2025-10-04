# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 00:33:53 2025

@author: Asus
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings

warnings.filterwarnings("ignore")

# --- Load dataset ---
df = pd.read_csv("E:/LHC-CERN/zee-main.csv")
print("Data loaded successfully.")

# --- Label encode categorical features (if any) ---
categorical_features = [
    # none for this dataset
]

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- Define features and target ---
X = df.drop("M", axis=1)   # Features
y = df["M"]                # Target is continuous mass

# --- Scale features ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# --- Split train-test ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42
)

# --- Train CatBoost Regressor ---
catboost_reg = CatBoostRegressor(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=100
)
catboost_reg.fit(X_train, y_train)

# --- Predict ---
y_pred = catboost_reg.predict(X_test)

# --- Evaluate with R² and RMSE ---
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"✅ R² Score: {r2:.4f}")
print(f"✅ RMSE: {rmse:.4f}")

# --- Save the trained CatBoost model ---
catboost_reg.save_model("E:/LHC-CERN/catboost_regressor-higgs-mass.cbm")

# --- Save encoders and scaler ---
joblib.dump(label_encoders, "E:/LHC-CERN/label_encoders-catboost-reg-higgs-mass.pkl")
joblib.dump(scaler, "E:/LHC-CERN/minmax_scaler-catboost-reg-higgs-mass.pkl")

print("✅ CatBoost Regressor, encoders, and scaler saved successfully.")

# --- Plot y_true vs y_pred ---
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("True M")
plt.ylabel("Predicted M")
plt.title(f"True vs Predicted M (CatBoost Regressor)\nR²={r2:.4f}, RMSE={rmse:.4f}")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # diagonal
plt.tight_layout()
plt.show()
# --- Plot target (M) vs selected features ---
features_to_plot = [
    "Run", "Event", "pt1", "eta1", "phi1", "Q1",
    "pt2", "eta2", "phi2", "Q2"
]

for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df[feature], y=df["M"],
        alpha=0.6, s=25, edgecolor=None
    )
    plt.title(f"M vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("M")
    plt.tight_layout()
    plt.show()
