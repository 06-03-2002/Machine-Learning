# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 04:56:39 2025

@author: Asus
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import AdaBoostClassifier

# ------------------------------
# Config
# ------------------------------
train_path = r"E:/URP/transportation/flights_sample_3m.csv/Flight cancellation sample- training dataset.csv"
model_folder_name = "adaboost_trained_model_no_graphviz"
random_state = 42
test_size = 0.1

# Surrogate tree configuration
SURROGATE_MAX_DEPTH = 4
SURROGATE_MIN_SAMPLES_LEAF = 20

# ------------------------------
# Load data
# ------------------------------
data = pd.read_csv(train_path, encoding="latin1")

# ------------------------------
# Encode categoricals
# ------------------------------
encoders = {}
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = data[col].astype(str)
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# ------------------------------
# Features / Target
# ------------------------------
if "CANCELLED" not in data.columns:
    raise ValueError("Target column 'CANCELLED' not found.")

features = data.drop("CANCELLED", axis=1)
target = data["CANCELLED"].astype(int)

# ------------------------------
# Impute missing values
# ------------------------------
imputer = SimpleImputer(strategy="most_frequent")
X = imputer.fit_transform(features)
y = target.values

# ------------------------------
# Train / Test split
# ------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# ------------------------------
# AdaBoost Classifier
# ------------------------------
base_tree = DecisionTreeClassifier(max_depth=3, random_state=random_state)
ada = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=200,
    learning_rate=0.1,
    random_state=random_state
)
ada.fit(x_train, y_train)

# ------------------------------
# Evaluate AdaBoost
# ------------------------------
y_pred = ada.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ AdaBoost Accuracy: {acc:.4f}")

if hasattr(ada, "predict_proba"):
    y_proba = ada.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AdaBoost ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - AdaBoost")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

# ------------------------------
# Feature importances (AdaBoost)
# ------------------------------
if hasattr(ada, "feature_importances_"):
    fi = pd.Series(ada.feature_importances_, index=features.columns).sort_values(ascending=True)
    plt.figure(figsize=(10, max(6, 0.25 * len(fi))))
    sns.barplot(x=fi.values, y=fi.index, orient="h")
    plt.title("Feature Importances (AdaBoost)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# ------------------------------
# Bar plots: CANCELLED vs each feature
# ------------------------------
plot_df = pd.DataFrame(X, columns=features.columns)
plot_df["CANCELLED"] = y

for col in features.columns:
    plt.figure(figsize=(8, 5))
    if pd.api.types.is_numeric_dtype(plot_df[col]):
        try:
            bins = pd.qcut(plot_df[col], q=10, duplicates="drop")
        except Exception:
            bins = pd.cut(plot_df[col], bins=10)
        grouped = plot_df.groupby([bins, "CANCELLED"]).size().reset_index(name="count")
        grouped["bin_str"] = grouped.iloc[:, 0].astype(str)
        sns.barplot(data=grouped, x="bin_str", y="count", hue="CANCELLED")
        plt.xlabel(f"{col} (binned)")
        plt.xticks(rotation=45, ha="right")
    else:
        grouped = plot_df.groupby([col, "CANCELLED"]).size().reset_index(name="count")
        sns.barplot(data=grouped, x=col, y="count", hue="CANCELLED")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(col)
    plt.title(f"CANCELLED vs {col}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# ------------------------------
# Surrogate Decision Tree (to approximate AdaBoost)
# ------------------------------
ada_train_pred = ada.predict(x_train)

surrogate = DecisionTreeClassifier(max_depth=SURROGATE_MAX_DEPTH,
                                   min_samples_leaf=SURROGATE_MIN_SAMPLES_LEAF,
                                   random_state=random_state)
surrogate.fit(x_train, ada_train_pred)

surrogate_pred_on_test = surrogate.predict(x_test)
ada_pred_on_test = ada.predict(x_test)
fidelity = (surrogate_pred_on_test == ada_pred_on_test).mean()
print(f"\n🔁 Surrogate fidelity (matches AdaBoost on test): {fidelity:.3f}")

surrogate_acc = accuracy_score(y_test, surrogate.predict(x_test))
print(f"🌳 Surrogate Decision Tree accuracy (vs true CANCELLED): {surrogate_acc:.3f}")

# ------------------------------
# Plot surrogate decision tree
# ------------------------------
plt.figure(figsize=(20, 12))
plot_tree(
    surrogate,
    feature_names=features.columns,
    class_names=["Not Cancelled", "Cancelled"],
    filled=True,
    rounded=True,
    fontsize=10,
    impurity=False,
    proportion=False
)
plt.title(f"Surrogate Decision Tree (max_depth={SURROGATE_MAX_DEPTH}) — approximates AdaBoost", fontsize=16)
plt.tight_layout()
plt.show()

# ------------------------------
# Print surrogate tree rules
# ------------------------------
print("\n📜 Surrogate Decision Tree Rules (human-readable):\n")
rules_text = export_text(surrogate, feature_names=list(features.columns))
print(rules_text)

# ------------------------------
# Save artifacts
# ------------------------------
base_dir = os.path.dirname(train_path)
model_dir = os.path.join(base_dir, model_folder_name)
os.makedirs(model_dir, exist_ok=True)

ada_model_file = os.path.join(model_dir, "adaboost_cancel_model.pkl")
surrogate_file = os.path.join(model_dir, "surrogate_decision_tree.pkl")
features_file = os.path.join(model_dir, "features_list.pkl")
encoders_file = os.path.join(model_dir, "label_encoders.pkl")
imputer_file = os.path.join(model_dir, "imputer.pkl")

joblib.dump(ada, ada_model_file)
joblib.dump(surrogate, surrogate_file)
joblib.dump(list(features.columns), features_file)
joblib.dump(encoders, encoders_file)
joblib.dump(imputer, imputer_file)

print("\n📁 Saved:")
print(f" - AdaBoost model: {ada_model_file}")
print(f" - Surrogate Decision Tree: {surrogate_file}")
print(f" - Features list: {features_file}")
print(f" - Label encoders: {encoders_file}")
print(f" - Imputer: {imputer_file}")

with open(os.path.join(model_dir, "surrogate_tree_rules.txt"), "w", encoding="utf8") as f:
    f.write(rules_text)

print(f"\n📄 Wrote surrogate rules into {model_dir}")
