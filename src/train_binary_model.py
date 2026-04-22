import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_dataset.csv")
MODELS_PATH = os.path.join(BASE_DIR, "models")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df["Binary_Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

X = df.drop(["Label", "Binary_Label"], axis=1)
y = df["Binary_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Binary Accuracy:",
      accuracy_score(y_test, model.predict(X_test_scaled)))

os.makedirs(MODELS_PATH, exist_ok=True)

joblib.dump(model, os.path.join(MODELS_PATH, "binary_ids.pkl"))
joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))