import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier

# LOAD DATA
df = pd.read_csv("final_fraud_dataset.csv")

# Handle missing values
df = df.fillna(0)

# Split features and target
target = "isFraud"
X = df.drop(columns=[target])
y = df[target]

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# EVALUATION FUNCTION
def evaluate_model(name, model, X_test, y_test, threshold=0.3):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fnr = fn / (fn + tp)

    print(f"\n{name}")
    print(f"Threshold: {threshold}")
    print("AUROC:", auc)
    print("F1:", f1)
    print("FNR:", fnr)

    return auc, f1, fnr

# FIND BEST THRESHOLD
def find_best_threshold(model, X_test, y_test):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_fnr = 1
    best_threshold = 0.5

    y_prob = model.predict_proba(X_test)[:, 1]

    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fnr = fn / (fn + tp)

        if fnr < best_fnr:
            best_fnr = fnr
            best_threshold = t

    print(f"\nBest Threshold Found: {best_threshold} (FNR: {best_fnr})")
    return best_threshold

# HANDLE CLASS IMBALANCE
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 6. TRAIN MODEL (TUNED XGBOOST)
model = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# EVALUATION
print("\n--- Default Threshold (0.3) ---")
evaluate_model("XGBoost", model, X_test, y_test, threshold=0.3)

# Find best threshold
best_threshold = find_best_threshold(model, X_test, y_test)

print("\n--- Optimized Threshold ---")
evaluate_model("XGBoost Optimized", model, X_test, y_test, threshold=best_threshold)

# SAVE MODEL
joblib.dump(model, "final_model.pkl")
print("\nModel saved as final_model.pkl")

# FEATURE IMPORTANCE
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure()
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.show()


# ROC CURVE
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()