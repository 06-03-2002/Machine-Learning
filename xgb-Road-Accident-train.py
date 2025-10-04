# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 21:20:19 2025

@author: Asus
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import warnings

warnings.filterwarnings("ignore")

# --- Load dataset ---
df = pd.read_csv("E:/ML-PROJECT/RTA Dataset-Train2.csv")
print("Data loaded successfully.")

# --- Label encode categorical features ---
categorical_features = [
    "Type_of_vehicle",
    "Defect_of_vehicle",
    "Lanes_or_Medians",
    "Road_allignment",
    "Types_of_Junction",
    "Road_surface_type",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions",
    "Vehicle_movement",
    "Cause_of_accident"
]

# Dictionary to store label encoders for each column
label_encoders = {}

for col in categorical_features + ["Accident_severity"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for future use

# --- Define features and target ---
X = df.drop("Accident_severity", axis=1)
y = df["Accident_severity"]

# --- Optional: Apply MinMaxScaler ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# --- Split train-test ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

xgb=XGBClassifier()
xgb.fit(X_train,y_train)
y_pred7=xgb.predict(X_test)
y_pred7
# --- Evaluate ---
print(classification_report(y_test, y_pred7))

# --- Save the trained KNN model ---
joblib.dump(xgb, "E:/ML-PROJECT/xgb_model.pkl")

# --- Save the label encoders ---
joblib.dump(label_encoders, "E:/ML-PROJECT/label_encoders-xgb.pkl")

# --- Save the scaler ---
joblib.dump(scaler, "E:/ML-PROJECT/minmax_scaler-xgb.pkl")

print("✅  model, label encoders, and scaler saved successfully.")
# --- 1. Plot classification report as heatmap ---
import numpy as np

from sklearn.metrics import confusion_matrix

# Get classification report as dictionary
report_dict = classification_report(y_test, y_pred7, output_dict=True)
report_df = pd.DataFrame(report_dict).iloc[:-1, :-1]  # Exclude avg/total row

plt.figure(figsize=(10, 6))
sns.heatmap(report_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Classification Report Heatmap")
plt.show()


# --- 2. Plot Accident_severity vs other categorical features (bar plots) ---
# Use original raw values for plotting
# Inverse transform the label-encoded values
df_raw = df.copy()
for col, le in label_encoders.items():
    df_raw[col] = le.inverse_transform(df[col])

# List of features to compare with Accident_severity
features_to_plot = [
    "Type_of_vehicle",
    "Defect_of_vehicle",
    "Lanes_or_Medians",
    "Road_allignment",
    "Types_of_Junction",
    "Road_surface_type",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions",
    "Vehicle_movement",
    "Cause_of_accident"
]

for feature in features_to_plot:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=feature, hue="Accident_severity", data=df_raw, palette="Set2")
    plt.title(f"Accident Severity vs {feature}")
    plt.xticks(rotation=45)
    plt.legend(title="Accident Severity")
    plt.tight_layout()
    plt.show()