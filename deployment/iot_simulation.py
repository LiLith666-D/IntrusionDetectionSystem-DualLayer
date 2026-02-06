import os
import pandas as pd
import joblib

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load models
    binary_model = joblib.load(os.path.join(base_path, "models", "binary_ids.pkl"))
    multiclass_model = joblib.load(os.path.join(base_path, "models", "random_forest_ids.pkl"))
    scaler = joblib.load(os.path.join(base_path, "models", "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(base_path, "models", "label_encoder.pkl"))

    # Load a small sample of traffic (simulate IoT traffic)
    data_path = os.path.join(base_path, "data", "processed", "cleaned_dataset.csv")
    df = pd.read_csv(data_path)

    df.columns = df.columns.str.strip()

    # Take ONE packet/flow as IoT input
    sample = df.sample(1)
    X = sample.drop("Label", axis=1)

    # Scale features
    X_scaled = scaler.transform(X)

    print("[*] Incoming IoT traffic received")

    # Stage 1: Binary IDS (IoT Gateway)
    binary_prediction = binary_model.predict(X_scaled)[0]

    if binary_prediction == 0:
        print("[*] Binary IDS: Traffic is BENIGN — Allowed at IoT Gateway")
    else:
        print("[*] Binary IDS: ATTACK detected — Forwarding to Edge Server")

        # Stage 2: Multi-class IDS (Edge)
        attack_pred = multiclass_model.predict(X_scaled)[0]
        attack_name = label_encoder.inverse_transform([attack_pred])[0]

        print(f"[*] Multi-class IDS: Attack Type → {attack_name}")

if __name__ == "__main__":
    main()
