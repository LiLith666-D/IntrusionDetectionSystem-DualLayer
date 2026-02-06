import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_test_data(splits_path):
    """
    Load test dataset
    """
    X_test = pd.read_csv(os.path.join(splits_path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(splits_path, "y_test.csv"))["Label"]

    print("[*] Test data loaded successfully")
    return X_test, y_test


def load_model_and_encoder(models_path):
    """
    Load trained model and label encoder
    """
    model = joblib.load(os.path.join(models_path, "random_forest_ids.pkl"))
    label_encoder = joblib.load(os.path.join(models_path, "label_encoder.pkl"))

    print("[*] Model and label encoder loaded successfully")
    return model, label_encoder


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Generate evaluation metrics
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"[✓] Test Accuracy: {accuracy:.4f}")

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    return y_pred, report, accuracy


def save_classification_report(report, results_path):
    """
    Save classification report as text file
    """
    os.makedirs(results_path, exist_ok=True)

    report_df = pd.DataFrame(report).transpose()
    report_file = os.path.join(results_path, "classification_report.csv")
    report_df.to_csv(report_file)

    print(f"[✓] Classification report saved to: {report_file}")


def plot_confusion_matrix(y_test, y_pred, label_encoder, results_path):
    """
    Plot and save confusion matrix
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        annot=False,
        cmap="Blues"
    )

    plt.title("Confusion Matrix - Random Forest IDS")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    cm_path = os.path.join(results_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"[✓] Confusion matrix saved to: {cm_path}")


def save_summary(accuracy, results_path):
    """
    Save evaluation summary
    """
    summary_path = os.path.join(results_path, "evaluation_summary.txt")

    with open(summary_path, "w") as f:
        f.write("Random Forest Intrusion Detection System Evaluation\n")
        f.write("=================================================\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")

    print(f"[✓] Evaluation summary saved to: {summary_path}")


def main():
    """
    Main evaluation pipeline
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    splits_path = os.path.join(base_path, "data", "splits")
    models_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, "results")

    print("[*] Starting model evaluation pipeline")

    X_test, y_test = load_test_data(splits_path)
    model, label_encoder = load_model_and_encoder(models_path)

    y_pred, report, accuracy = evaluate_model(
        model, X_test, y_test, label_encoder
    )

    save_classification_report(report, results_path)
    plot_confusion_matrix(y_test, y_pred, label_encoder, results_path)
    save_summary(accuracy, results_path)

    print("[✓] Model evaluation completed successfully")


if __name__ == "__main__":
    main()
