import os
import pandas as pd
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = os.path.join(base_path, "data", "processed", "cleaned_dataset.csv")
    models_path = os.path.join(base_path, "models")

    print("[*] Loading cleaned dataset")
    df = pd.read_csv(data_path)

    # Fix column names
    df.columns = df.columns.str.strip()

    # Create binary label
    df["Binary_Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    X = df.drop(["Label", "Binary_Label"], axis=1)
    y = df["Binary_Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[*] Training Binary Random Forest IDS")
    start = time.time()

    rf_binary = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )

    rf_binary.fit(X_train, y_train)

    end = time.time()

    y_pred = rf_binary.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[✓] Binary IDS Accuracy: {acc:.4f}")
    print(f"[✓] Training Time: {end - start:.2f} seconds")

    os.makedirs(models_path, exist_ok=True)
    joblib.dump(rf_binary, os.path.join(models_path, "binary_ids.pkl"))

    print("[✓] Binary IDS model saved")


if __name__ == "__main__":
    main()
