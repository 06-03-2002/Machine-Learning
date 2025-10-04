# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 23:56:09 2025

@author: Asus
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error, r2_score

# Read the CSV file
FILENAME='E:/URP/thesis/KUZNET CURVE-ENVIRONMENTS.csv'
data = pd.read_csv(FILENAME, encoding='latin1')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Track encoded columns and their mappings
encoded_mappings = {}

# Encode categorical variables
for column in data.select_dtypes(include=['object']):
    # Display the unique values before encoding
    print(f"Original values in '{column}': {data[column].unique()}")
    
    # Fit the encoder and transform the column
    data[column] = label_encoder.fit_transform(data[column].astype(str))
    
    # Store the mapping of original values to encoded labels
    encoded_mappings[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    # Display the mapping and the encoded values
    print(f"Mapping for '{column}': {encoded_mappings[column]}")
    print(f"Encoded values in '{column}': {data[column].unique()}\n")



