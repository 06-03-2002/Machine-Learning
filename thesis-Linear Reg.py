# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 14:39:16 2025

@author: Asus
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import numpy as np
# Read the CSV file
data = pd.read_csv("E:/URP/thesis/KUZNET CURVE-ENVIRONMENTS.csv",encoding='latin1')
# Convert non-numeric columns to categorical type
for column in data.select_dtypes(include=['object']):
    data[column] = data[column].astype(str)
# Encode categorical variables
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']):
    data[column] = label_encoder.fit_transform(data[column])

# Define columns to encode
columns_to_encode = []

# Initialize LabelEncoder
encoder = LabelEncoder()

# Encode categorical variables
for column in columns_to_encode:
    data[column] = encoder.fit_transform(data[column])

# Define features and target variable
features = data.drop(" GHG_ktCO2e", axis=1)
target = data[" GHG_ktCO2e"]

# Display the transformed dataframe
print(data.head())
# Handle missing values in the features
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Handle missing values in the target variable
target_imputed = imputer.fit_transform(target.values.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target_imputed, test_size=0.1)
# Apply polynomial features transformation
poly = PolynomialFeatures(degree=1, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Create an instance of the LinearRegression model
regressor = LinearRegression()

# Fit the model on the polynomial features
regressor.fit(x_train_poly, y_train)

# Retrieve the coefficients and intercept
coefficients = regressor.coef_
intercept = regressor.intercept_

# Retrieve the original feature names
original_feature_names = features.columns

# Generate the polynomial feature names
feature_names = list(original_feature_names)
for feature_idx in poly.powers_:
    if np.sum(feature_idx) > 1:
        feature_name = "*".join(
            [
                f"{name}^{power}"
                for name, power in zip(original_feature_names, feature_idx)
                if power > 0
            ]
        )
        feature_names.append(feature_name)

# Create the equation
equation = "GHG_ktCO2e= "
for i, coefficient in enumerate(coefficients):
    if i == 0:
        equation += f"{intercept:f}"
    else:
        equation += f" + {coefficient:f} * {feature_names[i]}"
print('coefficient',coefficients) 
print('intercept',intercept)       
print("Equation:", equation)
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Read the CSV file and select desired columns
# Read the CSV file and select desired columns

# Encode categorical variables
label_encoder = LabelEncoder()

# Convert non-numeric columns to categorical type
for column in data.select_dtypes(include=['object']):
    data[column] = data[column].astype(str)
for column in data.select_dtypes(include=['object']):
    data[column] = label_encoder.fit_transform(data[column])
# Define features and target variable


# Apply OrdinalEncoder to encode categorical variables
encoder = OrdinalEncoder()
features_encoded = encoder.fit_transform(features)

# Handle missing values in the encoded features
imputer = SimpleImputer(strategy='most_frequent')
features_imputed = imputer.fit_transform(features_encoded)

# Handle missing values in the target variable
target_imputed = imputer.fit_transform(target.values.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target_imputed, test_size=0.1)

# Create an instance of the LinearRegression model
linear = linear_model.LinearRegression()

# Fit the model on the training data
linear.fit(x_train, y_train)

# Evaluate the model's accuracy
acc = linear.score(x_test, y_test)
print("Accuracy:", acc)



import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# Read the CSV file and select desired columns

# Encode categorical variables
label_encoder = LabelEncoder()
data_encoded = data.apply(label_encoder.fit_transform)
label_encoder = LabelEncoder()
data_encoded = data.apply(label_encoder.fit_transform)
label_encoder = LabelEncoder()

features = data_encoded.drop(" GHG_ktCO2e", axis=1)
target = data_encoded[" GHG_ktCO2e"]


# Handle missing values in the features
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.1)

# Apply polynomial features transformation
poly = PolynomialFeatures(degree=1, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Create an instance of the LinearRegression model
regressor = LinearRegression()

# Fit the model on the polynomial features
regressor.fit(x_train_poly, y_train)

# Retrieve the coefficients and intercept
coefficients = regressor.coef_
intercept = regressor.intercept_

# Retrieve the original feature names
original_feature_names = features.columns

# Generate the polynomial feature names
feature_names = list(original_feature_names)
for feature_idx in poly.powers_:
    if np.sum(feature_idx) > 1:
        feature_name = "*".join(
            [
                f"{name}^{power}"
                for name, power in zip(original_feature_names, feature_idx)
                if power > 0
            ]
        )
        feature_names.append(feature_name)

# Create the equation
equation = "GHG_ktCO2e= "
for i, coefficient in enumerate(coefficients):
    if i == 0:
        equation += f"{intercept:.2f}"
    else:
        equation += f" + {coefficient:.2f} * {feature_names[i]}"
        



# Predict 'G1' values using the trained model
y_train_pred = regressor.predict(x_train_poly)
y_test_pred = regressor.predict(x_test_poly)

# Print the predicted 'G1' values

# Predict on the test data
import matplotlib.pyplot as plt
y_pred = regressor.predict(x_test_poly)

# Plot the actual G1 values and the predicted G1 values
plt.scatter(y_test, y_pred)
plt.plot([np.min(x_test), np.max(x_test)], [np.min(y_test), np.max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual GHG_ktCO2e")
plt.ylabel("Predicted GHG_ktCO2e")
plt.title("Linear Regression: Actual vs Predicted GHG_ktCO2e")
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Read the CSV file and select desired columns

# Encode categorical variables
label_encoder = LabelEncoder()
data_encoded = data.apply(label_encoder.fit_transform)

# Retrieve the column names
column_names = data_encoded.columns

# Plot 'G1' against each column
for column in column_names:
    if column != ' GHG_ktCO2e':
        plt.scatter(data_encoded[column], data_encoded[' GHG_ktCO2e'])
        plt.xlabel(column)
        plt.ylabel('GHG_ktCO2e')
        plt.title(f'GHG_ktCO2e vs {column}')
        plt.show()
        
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# =============================
# 1. Load & Preprocess Data
# =============================
data = pd.read_csv("E:/URP/thesis/KUZNET CURVE-ENVIRONMENTS.csv", encoding='latin1')

# Convert categorical columns to string, then encode
for column in data.select_dtypes(include=['object']):
    data[column] = data[column].astype(str)
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])

# Features and Target
features = data.drop(" GHG_ktCO2e", axis=1)
target = data[" GHG_ktCO2e"]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)
target_imputed = imputer.fit_transform(target.values.reshape(-1, 1)).flatten()

# =============================
# 2. Train-Test Split
# =============================
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target_imputed, test_size=0.1, random_state=42)

# =============================
# 3. Polynomial Linear Regression
# =============================
poly = PolynomialFeatures(degree=1, include_bias=False)  # change degree if needed
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

regressor = LinearRegression()
regressor.fit(x_train_poly, y_train)

# Predictions
y_pred = regressor.predict(x_test_poly)

# =============================
# 4. Evaluation Metrics
# =============================
#mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"R² Score: {r2:.3f}")

# =============================
# 5. Plot Actual vs Predicted
# =============================
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Ideal Fit")
plt.xlabel("Actual GHG_ktCO2e")
plt.ylabel("Predicted GHG_ktCO2e")
plt.title("Actual vs Predicted GHG Emissions")
plt.legend()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load your dataset
data = pd.read_csv("E:/URP/thesis/KUZNET CURVE-ENVIRONMENTS.csv", encoding="latin1")

# Identify categorical vs numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns
numeric_cols = data.select_dtypes(exclude=['object']).columns

# Ensure target column exists
target_col = ' GHG_ktCO2e'

for column in data.columns:
    if column == target_col:
        continue
    
    # If categorical -> bar plot
    if column in categorical_cols:
        plt.figure(figsize=(8,5))
        grouped = data.groupby(column)[target_col].mean().sort_values(ascending=False)  # mean GHG per category
        grouped.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.ylabel("Average GHG_ktCO2e")
        plt.title(f"Average GHG_ktCO2e by {column}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    
    # If numerical -> scatter plot
    else:
        plt.figure(figsize=(6,4))
        plt.scatter(data[column], data[target_col], alpha=0.6)
        plt.xlabel(column)
        plt.ylabel("GHG_ktCO2e")
        plt.title(f"GHG_ktCO2e vs {column}")
        plt.show()
