# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 20:19:01 2025

@author: Asus
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# Small epsilon to prevent log(0)
epsilon = 1e-10

# Load data from CSV
df = pd.read_csv("E:/LHC-CERN/MultiJetRun2010B.csv",encoding='latin1')  # Change "data.csv" to your actual filename
for column in df.select_dtypes(include=['object']):
    df[column] = df[column].astype(str)

label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']):
    df[column] = label_encoder.fit_transform(df[column])

# Automatically detect feature columns (all except the last column)
#feature_columns = df.columns[:-1]  # Assumes last column is the target (y)
#target_column = df.columns[-1]  # Assumes last column is y
target_column = "Lumi"  # Set your target column explicitly
feature_columns = [col for col in df.columns if col != target_column] 
# Extract features (X) and target (y)
X = df[feature_columns].values  # Feature matrix
y = df[target_column].values  # Target variable

# Take absolute values and add epsilon for stability
y_positive = np.abs(y) + epsilon
X_positive = np.abs(X) + epsilon  # Apply to all feature columns

# Logarithmic transformations
ln_y = np.log(y_positive).reshape(-1, 1)  # Log-transformed target
ln_X = np.log(X_positive)  # Log-transformed features

# Create polynomial features (degree can be adjusted)
degree = 2  # Adjust this value for higher-degree polynomial regression
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(ln_X)

# Perform multivariate linear regression on transformed polynomial data
model = LinearRegression()
model.fit(X_poly, ln_y.ravel())  # Use .ravel() to flatten ln_y

# Extract coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Predict ln(y) values using the model
ln_y_pred = model.predict(X_poly)

# Calculate R-squared score
r2 = r2_score(ln_y, ln_y_pred)

# Generate equation dynamically based on input features
feature_names = poly.get_feature_names_out([f"ln(|{col}|)" for col in feature_columns])
equation = f"ln(y) = {intercept:.5f}"
for coef, name in zip(coefficients, feature_names):
    equation += f" + {coef:.5f} * {name}"

# Print the results
print("Equation in transformed space (log-linearized with polynomial features):")
print(equation)
print(f"R-squared (transformed space): {r2:.5f}")
