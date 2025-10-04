# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 20:35:12 2025

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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

# ------------------------------
# Config
# ------------------------------
train_path = r"E:/URP/transportation/flights_sample_3m.csv/Flight cancellation sample- training dataset.csv"
model_folder_name = "adaboost_trained_model"
random_state = 42
test_size = 0.1

# ------------------------------
# Load data
# ------------------------------
data = pd.read_csv(train_path, encoding="latin1")

# ------------------------------
# Encode categoricals (LabelEncoder per column)
# NOTE: We also save encoders to reuse at inference time.
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
# Train / Test split (stratified)
# ------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# ------------------------------
# AdaBoost (use 'estimator' for modern scikit-learn)
# A decision stump (max_depth=1) is the standard base learner.
# ------------------------------
base_stump = DecisionTreeClassifier(max_depth=1, random_state=random_state)
ada = AdaBoostClassifier(
    estimator=base_stump,          # <-- use 'estimator' (NOT base_estimator)
    n_estimators=100,
    learning_rate=1.0,
    algorithm="SAMME.R",
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
# Bar plots: CANCELLED vs each feature (training data)
# ------------------------------
plot_df = pd.DataFrame(X, columns=features.columns)
plot_df["CANCELLED"] = y

for col in features.columns:
    plt.figure(figsize=(8, 5))
    if pd.api.types.is_numeric_dtype(plot_df[col]):
        # use qcut; if it fails (constant/duplicates), fallback to cut
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
# Train a Decision Tree for interpretation (conditions-only plot)
# ------------------------------
dt = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=random_state)
dt.fit(x_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(x_test))
print(f"🌳 Decision Tree Accuracy: {dt_acc:.4f}")

# Top features by DT importance
dt_fi = pd.Series(dt.feature_importances_, index=features.columns)
top_n = 10
top_features = dt_fi.nlargest(top_n).index.tolist()
print(f"📊 Top {top_n} DT features:", top_features)

# Plot conditions-only tree (no gini/proportions)
plt.figure(figsize=(20, 12))
plot_tree(
    dt,
    feature_names=features.columns,
    class_names=["Not Cancelled", "Cancelled"],
    filled=True,
    rounded=True,
    fontsize=10,
    impurity=False,
    proportion=False
)
plt.title("Decision Tree (Conditions Only)", fontsize=16)
plt.tight_layout()
plt.show()

# Optional: smaller tree using only top features (cleaner)
if len(top_features) > 0:
    top_idx = [list(features.columns).index(f) for f in top_features]
    dt_small = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=random_state)
    dt_small.fit(x_train[:, top_idx], y_train)

    plt.figure(figsize=(14, 10))
    plot_tree(
        dt_small,
        feature_names=top_features,
        class_names=["Not Cancelled", "Cancelled"],
        filled=True,
        rounded=True,
        fontsize=10,
        impurity=False,
        proportion=False
    )
    plt.title(f"Decision Tree (Top {len(top_features)} Features, Conditions Only)", fontsize=14)
    plt.tight_layout()
    plt.show()


    # Print readable rules (text) for the smaller tree
    print("\n📜 Decision Tree (Top features) rules:\n")
    print(export_text(dt_small, feature_names=top_features))
    # Full decision tree using all features (for complete rules)
dt_full = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=random_state)
dt_full.fit(x_train, y_train)

# --- Plot full decision tree ---
plt.figure(figsize=(20, 12))
plot_tree(
    dt_full,
    feature_names=features.columns,
    class_names=["Not Cancelled", "Cancelled"],
    filled=True,
    rounded=True,
    fontsize=10,
    impurity=False,
    proportion=False
)
plt.title("Decision Tree (All Features - Conditions Only)", fontsize=16)
plt.tight_layout()
plt.show()

# --- Print all tree rules in text format ---
print("\n📜 Decision Tree Rules (All Features):\n")
tree_rules = export_text(dt_full, feature_names=list(features.columns))
print(tree_rules)
# Full decision tree using all features (for complete rules)
dt_full = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=random_state)
dt_full.fit(x_train, y_train)

# --- Plot full decision tree ---
plt.figure(figsize=(20, 12))
plot_tree(
    dt_full,
    feature_names=features.columns,
    class_names=["Not Cancelled", "Cancelled"],
    filled=True,
    rounded=True,
    fontsize=10,
    impurity=False,
    proportion=False
)
plt.title("Decision Tree (All Features - Conditions Only)", fontsize=16)
plt.tight_layout()
plt.show()

# --- Print all tree rules in text format ---
print("\n📜 Decision Tree Rules (All Features):\n")
tree_rules = export_text(dt_full, feature_names=list(features.columns))
print(tree_rules)
# ------------------------------
# Save artifacts: AdaBoost model, Decision Tree, features, encoders, imputer
# ------------------------------
base_dir = os.path.dirname(train_path)
model_dir = os.path.join(base_dir, model_folder_name)
os.makedirs(model_dir, exist_ok=True)

ada_model_file = os.path.join(model_dir, "adaboost_cancel_model.pkl")
dt_model_file = os.path.join(model_dir, "decision_tree_cancel_model.pkl")
features_file = os.path.join(model_dir, "features_list.pkl")
encoders_file = os.path.join(model_dir, "label_encoders.pkl")
imputer_file = os.path.join(model_dir, "imputer.pkl")

#joblib.dump(ada, ada_model_file)
#joblib.dump(dt, dt_model_file)
#joblib.dump(list(features.columns), features_file)
#joblib.dump(encoders, encoders_file)
#joblib.dump(imputer, imputer_file)

#print("\n📁 Saved:")
#print(f" - AdaBoost model: {ada_model_file}")
#print(f" - Decision Tree model: {dt_model_file}")
#print(f" - Features list: {features_file}")
#print(f" - Label encoders: {encoders_file}")
#print(f" - Imputer: {imputer_file}")
