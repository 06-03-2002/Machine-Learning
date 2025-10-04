# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 00:44:53 2025

@author: Asus
"""

import pandas as pd

# Input and output paths
input_file = r"E:/LHC-CERN/Book4-cern-di_mioun.xlsx"
output_file = r"E:\LHC-CERN\Book4-cern-zmumu.csv"  # keep .csv extension

# Read Excel file
df = pd.read_excel(input_file)

# Save as CSV with semicolon separator instead of comma
df.to_csv(output_file, sep=';', index=False)

print(f"✅ File converted successfully and saved at: {output_file}")
