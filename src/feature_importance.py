import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt


def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(base_path, "models", "random_forest_ids.pkl")
    data_path = os.path.join(base_path, "data", "processed", "cleaned_dataset.csv")
    results_path = os.path.join(base_path, "results")

    print("[*] Loading model and data")

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    df.columns = df.columns.str.strip()

    X = df.drop("Label", axis=1)

    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    os.makedirs(results_path, exist_ok=True)

    feature_df.to_csv(
        os.path.join(results_path, "feature_importance.csv"),
        index=False
    )

    print("[✓] Feature importance saved")

    # Plot top 10 features
    top_features = feature_df.head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Important Features - Random Forest IDS")
    plt.xlabel("Importance Score")
    plt.tight_layout()

    plt.savefig(os.path.join(results_path, "top_10_features.png"))
    plt.close()

    print("[✓] Feature importance plot saved")


if __name__ == "__main__":
    main()
