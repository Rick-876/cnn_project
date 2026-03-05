import os
import pandas as pd

# Path to the dataset directory
DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets--Meyerger--ASAG2024", "snapshots", "9a7a179de4e227f5e78625bc9ded2d8761f1ff09")
parquet_path = os.path.join(DATASET_DIR, "train.parquet")

if os.path.exists(parquet_path):
    df = pd.read_parquet(parquet_path)
    # Try to find the question column
    question_col = None
    for col in df.columns:
        if 'question' in col.lower():
            question_col = col
            break
    if question_col:
        print(f"Number of questions in train.parquet: {df[question_col].nunique()}")
    else:
        print(f"Number of rows in train.parquet: {len(df)} (question column not found)")
else:
    print("train.parquet not found in the dataset directory.")
