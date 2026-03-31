import joblib
import pandas as pd

model = joblib.load("final_model.pkl")

# Example: load one row
df = pd.read_csv("final_fraud_dataset.csv").iloc[:1]
df = df.fillna(0)

X = df.drop(columns=["isFraud"])

prob = model.predict_proba(X)[0][1]
prediction = 1 if prob > 0.3 else 0

print("Fraud probability:", prob)
print("Prediction:", prediction)