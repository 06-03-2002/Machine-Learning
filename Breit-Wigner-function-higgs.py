# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 10:26:58 2025

@author: Asus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load dataset ===
file_path = r"E:/LHC-CERN/Z_boson-2.csv"
df = pd.read_csv(file_path)

# === Lepton masses (in GeV) ===
m_mu = 0.1056583745   # muon
m_e  = 0.00051099895  # electron

# Infer lepton mass from 'class' column (Zmumu = muons, Zee = electrons)
def lepton_mass_from_class(cl):
    if isinstance(cl, str) and "Zmumu" in cl:
        return m_mu
    if isinstance(cl, str) and "Zee" in cl:
        return m_e
    return m_mu  # default fallback

df["lep_mass"] = df["class"].apply(lepton_mass_from_class)

# === Reconstruct 4-vectors ===
df["px1"] = df["pt1"] * np.cos(df["phi1"])
df["py1"] = df["pt1"] * np.sin(df["phi1"])
df["pz1"] = df["pt1"] * np.sinh(df["eta1"])
df["p1"]  = np.sqrt(df["pt1"]**2 + df["pz1"]**2)
df["E1"]  = np.sqrt(df["p1"]**2 + df["lep_mass"]**2)

df["px2"] = df["pt2"] * np.cos(df["phi2"])
df["py2"] = df["pt2"] * np.sin(df["phi2"])
df["pz2"] = df["pt2"] * np.sinh(df["eta2"])
df["p2"]  = np.sqrt(df["pt2"]**2 + df["pz2"]**2)
df["E2"]  = np.sqrt(df["p2"]**2 + df["lep_mass"]**2)

# === Combine ===
df["E_tot"]  = df["E1"] + df["E2"]
df["px_tot"] = df["px1"] + df["px2"]
df["py_tot"] = df["py1"] + df["py2"]
df["pz_tot"] = df["pz1"] + df["pz2"]

# Invariant mass (GeV)
df["inv_mass"] = np.sqrt(
    np.maximum(0, df["E_tot"]**2 - (df["px_tot"]**2 + df["py_tot"]**2 + df["pz_tot"]**2))
)

# === Z boson window (66–116 GeV) ===
df["is_Z_window"] = df["inv_mass"].between(60, 120)

# === Save output ===
out_path = r"E:/LHC-CERN/Z_boson-2_with_invmass4.csv"
#df.to_csv(out_path, index=False)

#print("✅ Invariant mass added and saved to:", out_path)

# === Plot percentage of True/False in is_Z_window ===
counts = df["is_Z_window"].value_counts(normalize=True) * 100
counts.plot(kind="bar", color=["red", "green"])
plt.xticks([0,1], ["True","False",], rotation=0)
plt.ylabel("Percentage of events (%)")
plt.title("Events inside Z boson mass window (60–120 GeV)")
for i, v in enumerate(counts):
    plt.text(i, v + 1, f"{v:.2f}%", ha="center", fontsize=10)
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === path to your saved file ===
#file_path = r"E:/LHC-CERN/Z_boson-2_with_invmass3.csv"
file_path = out_path
# === load ===
df = pd.read_csv(file_path)

# === robust conversion of is_Z_window to boolean ===
s = df['is_Z_window']

# handle mixed types (True/False booleans, "True"/"False" strings, 1/0 numbers)
def to_bool(x):
    # True for typical truthy representations
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    xs = str(x).strip().lower()
    if xs in ("true", "t", "1", "yes", "y"):
        return True
    if xs in ("false", "f", "0", "no", "n"):
        return False
    # fallback: try numeric
    try:
        return float(xs) != 0.0
    except:
        return False

df['is_Z_window'] = df['is_Z_window'].apply(to_bool)

# === compute percentages ===
pct_true = df['is_Z_window'].mean() * 100.0
pct_false = 100.0 - pct_true
print(f"Fraction inside Z window: True = {pct_true:.2f}%, False = {pct_false:.2f}%")

# === bar plot (explicit order: False, True) ===
labels = ['False', 'True']
values = [pct_false, pct_true]
colors = ['tab:red', 'tab:green']

fig, ax = plt.subplots(figsize=(5,4))
ax.bar(labels, values, color=colors)
for i, v in enumerate(values):
    ax.text(i, v + 1.0, f"{v:.2f}%", ha='center', fontsize=10)
ax.set_ylim(0, 100)
ax.set_ylabel('Percentage of events (%)')
ax.set_title('Events inside Z mass window (60–120 GeV)')
plt.tight_layout()
plt.show()

# === histogram of inv_mass with Z window shading ===
if 'inv_mass' in df.columns:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(df['inv_mass'].dropna(), bins=120, histtype='stepfilled', alpha=0.6)
    ax.axvspan(66, 116, color='orange', alpha=0.2, label='Z window (60–120 GeV)')
    ax.set_xlabel('Invariant mass (GeV)')
    ax.set_ylabel('Counts')
    ax.set_title('Invariant mass distribution (all events)')
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("inv_mass column not found in file.")
