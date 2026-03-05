import os
import pandas as pd
from collections import Counter

# Path to the dataset directory
DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets--Meyerger--ASAG2024", "snapshots", "9a7a179de4e227f5e78625bc9ded2d8761f1ff09")
parquet_path = os.path.join(DATASET_DIR, "train.parquet")

subjects = [
    "biology", "chemistry", "physics", "mathematics", "math", "language", "reading", "comprehension", "social studies"
]
subject_map = {
    "biology": "Science: Biology",
    "chemistry": "Science: Chemistry",
    "physics": "Science: Physics",
    "mathematics": "Mathematics",
    "math": "Mathematics",
    "language": "Language and Reading",
    "reading": "Language and Reading",
    "comprehension": "Comprehension",
    "social studies": "Social Studies"
}

if os.path.exists(parquet_path):
    df = pd.read_parquet(parquet_path)
    # Combine all questions and answers text, handling NaN values
    text_data = " ".join(df.apply(lambda row: " ".join([str(x) if pd.notnull(x) else "" for x in row]), axis=1))
    # Count subject keywords
    counts = Counter()
    for subj in subjects:
        counts[subj] = text_data.lower().count(subj)
    # Map to display names and sort
    mapped_counts = {subject_map[k]: v for k, v in counts.items()}
    sorted_counts = sorted(mapped_counts.items(), key=lambda x: x[1], reverse=True)
    print("Subject frequency in questions and answers:")
    for subject, count in sorted_counts:
        print(f"{subject}: {count}")
else:
    print("train.parquet not found in the dataset directory.")
