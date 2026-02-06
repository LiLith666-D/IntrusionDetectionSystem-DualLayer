import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def load_data(splits_path):
    """
    Load train-test splits
    """
    X_train = pd.read_csv(os.path.join(splits_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(splits_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(splits_path, "y_train.csv"))["Label"]
    y_test = pd.read_csv(os.path.join(splits_path, "y_test.csv"))["Label"]

    print("[*] Train-test data loaded successfully")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier
    """
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )

    print("[*] Training Random Forest model...")
    start_time = time.time()
    rf.fit(X_train, y_train)
    end_time = time.time()

    print(f"[✓] Training completed in {end_time - start_time:.2f} seconds")
    return rf


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model accuracy
    """
    print("[*] Evaluating model on test data...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[✓] Test Accuracy: {acc:.4f}")
    return acc


def save_model(model, models_path):
    """
    Save trained model
    """
    os.makedirs(models_path, exist_ok=True)
    model_path = os.path.join(models_path, "random_forest_ids.pkl")
    joblib.dump(model, model_path)
    print(f"[✓] Trained model saved to: {model_path}")


def main():
    """
    Main training pipeline
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    splits_path = os.path.join(base_path, "data", "splits")
    models_path = os.path.join(base_path, "models")

    print("[*] Starting Random Forest training pipeline")

    X_train, X_test, y_train, y_test = load_data(splits_path)
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)
    save_model(rf_model, models_path)

    print("[✓] Random Forest training pipeline completed successfully")


if __name__ == "__main__":
    main()
