import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


def load_cleaned_data(processed_path):

    if not os.path.exists(processed_path):
        raise FileNotFoundError("Cleaned dataset not found. Run data_cleaning.py first.")

    df = pd.read_csv(processed_path)
    df.columns = df.columns.str.strip()

    print("[*] Cleaned dataset loaded successfully")
    print("[*] Dataset shape:", df.shape)

    return df


def balance_dataset(df):

    print("[*] Balancing dataset to 50% Attack and 50% Benign")

    if "Label" not in df.columns:
        raise ValueError("Target column 'Label' not found")


    benign_df = df[df["Label"] == "BENIGN"]
    attack_df = df[df["Label"] != "BENIGN"]

    print("[*] Original BENIGN samples:", len(benign_df))
    print("[*] Original ATTACK samples:", len(attack_df))

    min_samples = min(len(benign_df), len(attack_df))

    benign_sampled = benign_df.sample(n=min_samples, random_state=42)
    attack_sampled = attack_df.sample(n=min_samples, random_state=42)

    balanced_df = pd.concat([benign_sampled, attack_sampled])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("[✓] Balanced dataset shape:", balanced_df.shape)
    print("[✓] Each class count:", min_samples)

    return balanced_df


def preprocess_data(df):

    if "Label" not in df.columns:
        raise ValueError("Target column 'Label' not found")

    X = df.drop("Label", axis=1)
    y = df["Label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("[*] Label encoding completed")
    print("[*] Classes:", list(label_encoder.classes_))

    return X, y_encoded, label_encoder


def split_and_scale(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[*] Train-test split and scaling completed")
    print("[*] Training samples:", X_train.shape[0])
    print("[*] Testing samples:", X_test.shape[0])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_splits(X_train, X_test, y_train, y_test, splits_path):

    os.makedirs(splits_path, exist_ok=True)

    pd.DataFrame(X_train).to_csv(os.path.join(splits_path, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(splits_path, "X_test.csv"), index=False)
    pd.DataFrame(y_train, columns=["Label"]).to_csv(os.path.join(splits_path, "y_train.csv"), index=False)
    pd.DataFrame(y_test, columns=["Label"]).to_csv(os.path.join(splits_path, "y_test.csv"), index=False)

    print(f"[✓] Train-test splits saved to: {splits_path}")


def save_encoders(scaler, label_encoder, models_path):

    os.makedirs(models_path, exist_ok=True)

    joblib.dump(scaler, os.path.join(models_path, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(models_path, "label_encoder.pkl"))

    print("[✓] Scaler and LabelEncoder saved")


def main():

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    processed_path = os.path.join(base_path, "data", "processed", "cleaned_dataset.csv")
    splits_path = os.path.join(base_path, "data", "splits")
    models_path = os.path.join(base_path, "models")

    print("[*] Starting preprocessing pipeline")

    df = load_cleaned_data(processed_path)

    df = balance_dataset(df)

    X, y, label_encoder = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    save_splits(X_train, X_test, y_train, y_test, splits_path)
    save_encoders(scaler, label_encoder, models_path)

    print("[✓] Preprocessing completed successfully")


if __name__ == "__main__":
    main()