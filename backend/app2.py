import os
import random
import pandas as pd
import joblib
from flask import Flask, jsonify
from flask_cors import CORS

# -----------------------------
# PATH CONFIGURATION
# -----------------------------
BASE_PATH = "/home/vadapav/College/miniproject/MIDS/sentryai"

DATA_PATH = os.path.join(BASE_PATH, "data", "cleaned_dataset.csv")
MODEL_PATH = os.path.join(BASE_PATH, "models")

# -----------------------------
# LOAD DATASET
# -----------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)

# Separate features and label
X_full = df.drop(columns=["Label"], errors="ignore")

# -----------------------------
# LOAD MODELS
# -----------------------------
print("Loading models...")

binary_model = joblib.load(os.path.join(MODEL_PATH, "binary_ids.pkl"))
multi_model = joblib.load(os.path.join(MODEL_PATH, "random_forest_ids.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))

print("Models loaded successfully.")

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["GET"])
def predict():
    # Random row from dataset
    idx = random.randint(0, len(X_full) - 1)
    sample = X_full.iloc[[idx]]

    # Scale
    sample_scaled = scaler.transform(sample)

    # Binary Prediction
    binary_pred = binary_model.predict(sample_scaled)[0]

    if binary_pred == 0:
        return jsonify({
            "level1": "BENIGN",
            "level2": None,
            "confidence": 100,
            "flow": {}
        })

    # Multiclass Prediction
    multi_pred = multi_model.predict(sample_scaled)[0]
    multi_proba = multi_model.predict_proba(sample_scaled).max()

    attack_label = label_encoder.inverse_transform([multi_pred])[0]

    return jsonify({
        "level1": "ATTACK",
        "level2": attack_label,
        "confidence": round(float(multi_proba * 100), 2),
        "flow": {
            "flow_bytes_per_s": float(sample.iloc[0].get("Flow Bytes/s", 0)),
            "flow_pkts_per_s": float(sample.iloc[0].get("Flow Packets/s", 0)),
            "flow_duration": float(sample.iloc[0].get("Flow Duration", 0)),
            "dst_port": int(sample.iloc[0].get("Destination Port", 0))
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)