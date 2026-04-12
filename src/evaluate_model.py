import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def load_data_and_model(base_path, model_type="multiclass"):

    splits_path = os.path.join(base_path, "data", "splits")
    models_path = os.path.join(base_path, "models")

    if model_type == "multiclass":
        model_path = os.path.join(models_path, "random_forest_ids.pkl")
        X_test = pd.read_csv(os.path.join(splits_path, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(splits_path, "y_test.csv"))["Label"]
        label_encoder = joblib.load(os.path.join(models_path, "label_encoder.pkl"))

        class_names = label_encoder.classes_

    elif model_type == "binary":
        model_path = os.path.join(models_path, "binary_ids.pkl")

        # Reload dataset for binary split
        data_path = os.path.join(base_path, "data", "processed", "cleaned_dataset.csv")
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()

        df["Binary_Label"] = df["Label"].apply(
            lambda x: 0 if x == "BENIGN" else 1
        )

        from sklearn.model_selection import train_test_split

        X = df.drop(["Label", "Binary_Label"], axis=1)
        y = df["Binary_Label"]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        class_names = ["BENIGN", "ATTACK"]

    else:
        raise ValueError("Invalid model type")

    model = joblib.load(model_path)

    return model, X_test, y_test, class_names


def evaluate_model(model, X_test, y_test, class_names, results_path, model_type):

    print(f"\n[*] Evaluating {model_type.upper()} model")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"[✓] Accuracy : {acc:.4f}")
    print(f"[✓] Precision: {precision:.4f}")
    print(f"[✓] Recall   : {recall:.4f}")
    print(f"[✓] F1 Score : {f1:.4f}")

    os.makedirs(results_path, exist_ok=True)

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        os.path.join(results_path, f"{model_type}_classification_report.csv")
    )

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"Confusion Matrix - {model_type.upper()} IDS")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(os.path.join(results_path, f"{model_type}_confusion_matrix.png"))
    plt.close()

    print(f"[✓] Results saved in: {results_path}")


def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_path, "results")

    model, X_test, y_test, class_names = load_data_and_model(
        base_path,
        model_type="multiclass"
    )
    evaluate_model(
        model,
        X_test,
        y_test,
        class_names,
        results_path,
        "multiclass"
    )

    model, X_test, y_test, class_names = load_data_and_model(
        base_path,
        model_type="binary"
    )
    evaluate_model(
        model,
        X_test,
        y_test,
        class_names,
        results_path,
        "binary"
    )

    print("\n[✓] Evaluation for both models completed successfully")


if __name__ == "__main__":
    main()