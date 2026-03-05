"""
save_test_set.py
Reads asag2024_all.csv, performs the same 80/20 train/test split used in
asag_cnn_pipeline.py (random_state=42), and saves the test portion to test_set.csv.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full dataset
df = pd.read_csv("asag2024_all.csv")

# Rename columns to match pipeline conventions
df.columns = ["question", "provided_answer", "reference_answer", "normalized_grade"]

# Fill missing values
df = df.fillna({"question": "", "reference_answer": "", "provided_answer": "", "normalized_grade": 0.0})

# Build combined text input (mirrors asag_cnn_pipeline.py)
df["text"] = (
    "Question: " + df["question"] + " "
    "Reference: " + df["reference_answer"] + " "
    "Student: " + df["provided_answer"]
)

# 80/20 split with the same seed
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save test set
test_df[["text", "normalized_grade"]].to_csv("test_set.csv", index=False, encoding="utf-8")

print(f"Total rows   : {len(df)}")
print(f"Train rows   : {len(train_df)}")
print(f"Test rows    : {len(test_df)}")
print("Saved        : test_set.csv")
