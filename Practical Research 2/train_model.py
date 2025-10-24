

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

CSV_PATH = "data/features_labels.csv"
MODEL_OUT = "models/rf_npk.pkl"

# Ensure input CSV exists
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Input CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Example: feature columns (if you extracted using features.py)
feature_cols = [
    "mean_R","mean_G","mean_B","mean_exg","mean_vari","mean_gli",
    "area_px","perimeter_px","aspect_ratio","solidity",
    "lap_var","entropy"
]
# target column(s)
target_col = "N_percent"   # change to your ground truth column

# Check if all required columns exist
missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

# drop rows with NaN
df = df.dropna(subset=feature_cols+[target_col])

# Check if data is available after dropping NaNs
if df.empty:
    raise ValueError("No data left after dropping NaNs. Check your CSV file.")

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

# Ensure output directory exists
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# save model and feature list
joblib.dump({"model": rf, "features": feature_cols}, MODEL_OUT)
print("Saved model to", MODEL_OUT)
