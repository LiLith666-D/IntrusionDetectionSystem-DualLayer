import os
import glob
import pandas as pd
import numpy as np


def load_and_merge_csv(raw_data_path):
    """
    Load and merge all CSV files from the raw data directory
    """
    csv_files = glob.glob(os.path.join(raw_data_path, "*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in data/raw directory")

    df_list = []
    for file in csv_files:
        print(f"[+] Loading: {os.path.basename(file)}")
        df = pd.read_csv(file, low_memory=False)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def clean_dataset(df):
    """
    Perform data cleaning operations
    """
    print("[*] Initial dataset shape:", df.shape)

    # Drop non-informative / leakage columns
    drop_columns = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Source Port",
        "Destination Port",
        "Timestamp"
    ]
    df.drop(columns=drop_columns, errors="ignore", inplace=True)

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    print("[*] Cleaned dataset shape:", df.shape)
    return df


def save_cleaned_data(df, output_path):
    """
    Save cleaned dataset to processed directory
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✓] Cleaned dataset saved to: {output_path}")


def main():
    """
    Main data cleaning pipeline
    """

    # Dynamically determine project root
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    raw_data_path = os.path.join(base_path, "data", "raw")
    output_path = os.path.join(base_path, "data", "processed", "cleaned_dataset.csv")

    print("[*] Starting data cleaning pipeline")
    print(f"[*] Reading raw data from: {raw_data_path}")

    df = load_and_merge_csv(raw_data_path)
    cleaned_df = clean_dataset(df)
    save_cleaned_data(cleaned_df, output_path)

    print("[✓] Data cleaning completed successfully")


if __name__ == "__main__":
    main()
