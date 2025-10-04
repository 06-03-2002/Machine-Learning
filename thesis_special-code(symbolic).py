# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 23:35:53 2025

@author: Asus
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.impute import SimpleImputer

# Small epsilon to prevent log(0)
epsilon = 1e-10

# Load dataset
data_path = r"E:/URP/thesis/train file-kuznet curve.csv"
df = pd.read_csv(data_path, encoding='latin1')

# Ensure target column exists and drop rows where target is NaN
target_column = "gdp"
df = df.dropna(subset=[target_column]).reset_index(drop=True)

# Encode categorical features
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']):
    df[column] = df[column].astype(str)
    df[column] = label_encoder.fit_transform(df[column])

# Automatically detect feature columns (all except target)
feature_columns = [col for col in df.columns if col != target_column]

# Handle missing values in features using mean imputation
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df[feature_columns])
y = df[target_column].values

# Take absolute values and add epsilon for stability
y_positive = np.abs(y) + epsilon
X_positive = np.abs(X) + epsilon

# Logarithmic transformations
ln_y = np.log(y_positive).reshape(-1, 1)
ln_X = np.log(X_positive)

# Create polynomial features
degree = 2  # Change degree for higher-order polynomials
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(ln_X)

# Fit linear regression on transformed data
model = LinearRegression()
model.fit(X_poly, ln_y.ravel())

# Extract coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Predict ln(y)
ln_y_pred = model.predict(X_poly)

# Calculate R-squared
r2 = r2_score(ln_y, ln_y_pred)

# Generate dynamic equation
feature_names = poly.get_feature_names_out([f"ln(|{col}|)" for col in feature_columns])
equation = f"ln(gdp) = {intercept:.5f}"
for coef, name in zip(coefficients, feature_names):
    equation += f" + {coef:.5f} * {name}"

# Print results
print("Equation in transformed space (log-linearized with polynomial features):")
print(equation)
print(f"R-squared (transformed space): {r2:.5f}")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.impute import SimpleImputer

# Small epsilon to prevent log(0)
epsilon = 1e-10

# Load dataset
data_path = r"E:/URP/thesis/train file-kuznet curve.csv"
df = pd.read_csv(data_path, encoding='latin1')

# Ensure target column exists and drop rows where target is NaN
target_column = "gdp"
df = df.dropna(subset=[target_column]).reset_index(drop=True)

# Encode categorical features
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']):
    df[column] = df[column].astype(str)
    df[column] = label_encoder.fit_transform(df[column])

# Automatically detect feature columns (all except target)
feature_columns = [col for col in df.columns if col != target_column]

# Handle missing values in features using mean imputation
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df[feature_columns])
y = df[target_column].values

# Conditional logarithmic transformation for features
def signed_log(x):
    """If x > 0: ln(x), if x < 0: -ln(|x|), if x=0: 0"""
    x = np.array(x)
    result = np.zeros_like(x, dtype=float)
    pos_mask = x > 0
    neg_mask = x < 0
    result[pos_mask] = np.log(x[pos_mask] + epsilon)
    result[neg_mask] = -np.log(np.abs(x[neg_mask]) + epsilon)
    return result

ln_X = signed_log(X)
ln_y = signed_log(y).reshape(-1, 1)

# Create polynomial features
degree = 2  # Adjust degree as needed
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(ln_X)

# Fit linear regression on transformed data
model = LinearRegression()
model.fit(X_poly, ln_y.ravel())

# Extract coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Predict ln(y)
ln_y_pred = model.predict(X_poly)

# Calculate R-squared
r2 = r2_score(ln_y, ln_y_pred)

# Generate dynamic equation
feature_names = poly.get_feature_names_out([f"sln({col})" for col in feature_columns])
equation = f"sln(gdp) = {intercept:.5f}"
for coef, name in zip(coefficients, feature_names):
    equation += f" + {coef:.5f} * {name}"

# Print results
print("Equation in transformed space (signed-log with polynomial features):")
print(equation)
print(f"R-squared (transformed space): {r2:.5f}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('E:/URP/thesis/train file-kuznet curve.csv', encoding='latin1')

# Convert categorical columns to string type
for column in data.select_dtypes(include=['object']):
    data[column] = data[column].astype(str)

# Encode categorical variables if any
columns_to_encode = []
encoder = LabelEncoder()
for column in columns_to_encode:
    data[column] = encoder.fit_transform(data[column])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(data.drop("gdp", axis=1))

# Convert back to DataFrame
df_plot = pd.DataFrame(features_imputed, columns=data.columns[:-1])
df_plot['gdp'] = data['gdp']

# Get all features
selected_features = df_plot.columns[:-1]  # Exclude 'class'

# Downsample dataset for visualization (keep original data for modeling)
df_sample = df_plot.sample(2000, random_state=42)  # Adjust sample size for speed

# Plot each feature separately on a new page
for feature in selected_features:
    plt.figure(figsize=(8, 6))  # Create a new figure for each feature
    
    # Scatter plot of feature vs. class
    sns.scatterplot(x=df_sample[feature], y=df_sample["gdp"], alpha=0.5)
    
    plt.xlabel(feature)
    plt.ylabel("gdp")
    plt.title(f"{feature} vs gdp")
    
    plt.show()  # Show one figure at a time (separate page)
