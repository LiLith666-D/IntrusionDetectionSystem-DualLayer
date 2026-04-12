import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_dataset.csv")
MODELS_PATH = os.path.join(BASE_DIR, "models")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df = df[df["Label"] != "BENIGN"]

X = df.drop("Label", axis=1)
y_raw = df["Label"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

print("Multiclass Accuracy:",
      accuracy_score(y_test, model.predict(X_test)))

os.makedirs(MODELS_PATH, exist_ok=True)

joblib.dump(model, os.path.join(MODELS_PATH, "random_forest_ids.pkl"))
joblib.dump(label_encoder, os.path.join(MODELS_PATH, "label_encoder.pkl"))