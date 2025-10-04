# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 08:17:50 2025

@author: Asus
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Small epsilon to avoid log(0)
epsilon = 1e-10

# === Load dataset ===
df = pd.read_csv("E:/LHC-CERN/Z_boson-2.csv")
print("✅ Data loaded successfully.")

# Encode target variable
le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])

# Define features and target
X = df.drop("class", axis=1).values
feature_names = df.drop("class", axis=1).columns
y = df["class"].values

# === Symbolic transformation (ln(|x|)) ===
X_transformed = np.log(np.abs(X) + epsilon)
feature_names_sym = [f"ln(abs({col})+ε)" for col in feature_names]

# === Polynomial expansion (can try degree=2 or 3 for more complexity) ===
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_transformed)

# Expanded feature names
poly_feature_names = poly.get_feature_names_out(feature_names_sym)

# === Standardize features ===
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# === Logistic Regression ===
log_reg = LogisticRegression(max_iter=2000, solver="lbfgs")
log_reg.fit(X_poly_scaled, y)

# Predictions
y_pred = log_reg.predict(X_poly_scaled)

# Evaluate
print("\n📊 Classification Report:")
print(classification_report(y, y_pred))
print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")

# === Build symbolic logistic regression equation ===
coef = log_reg.coef_[0]
intercept = log_reg.intercept_[0]

equation_terms = []
for c, fname in zip(coef, poly_feature_names):
    if abs(c) > 1e-6:  # filter small coefficients
        equation_terms.append(f"{c:.10f} * {fname}")

equation = " + ".join(equation_terms)
logit_eq = f"logit(p) = {intercept:.10f} + " + equation

print("\n🔑 Symbolic Logistic Regression Equation (with scaling):")
print(logit_eq)  # preview first 2000 chars
