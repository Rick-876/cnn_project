def show_questions_and_answers(parquet_file):
    import pandas as pd
    from tabulate import tabulate
    df = pd.read_parquet(parquet_file)
    # Try to find the relevant columns by common names
    col_map = {
        'Question': None,
        'Student Answer': None,
        'Reference Answer': None,
        'Human Score/Grade': None
    }
    # Try to match columns by lower-case substrings
    for col in df.columns:
        lcol = col.lower()
        if 'question' in lcol and not col_map['Question']:
            col_map['Question'] = col
        elif ('student' in lcol or 'answer' in lcol) and not col_map['Student Answer']:
            col_map['Student Answer'] = col
        elif ('reference' in lcol or 'ref' in lcol) and not col_map['Reference Answer']:
            col_map['Reference Answer'] = col
        elif 'score' in lcol or 'grade' in lcol:
            col_map['Human Score/Grade'] = col
    # Fallback: try to find second/third answer columns
    if not col_map['Student Answer']:
        for col in df.columns:
            if 'answer' in col.lower():
                col_map['Student Answer'] = col
                break
    if not col_map['Reference Answer']:
        for col in df.columns:
            if 'reference' in col.lower() or 'ref' in col.lower():
                col_map['Reference Answer'] = col
                break
    # Prepare the output table
    display_cols = [col_map['Question'], col_map['Student Answer'], col_map['Reference Answer'], col_map['Human Score/Grade']]
    display_cols = [c for c in display_cols if c]
    if display_cols:
        table = df[display_cols]
        table.columns = [k for k, v in col_map.items() if v in display_cols]
        print(tabulate(table.head(10), headers='keys', tablefmt='github', showindex=False))
        # Output to CSV
        csv_path = os.path.join(os.path.dirname(__file__), 'asag2024_all.csv')
        table.to_csv(csv_path, index=False)
        # Output to Excel
        excel_path = os.path.join(os.path.dirname(__file__), 'asag2024_all.xlsx')
        table.to_excel(excel_path, index=False)
        print(f"\nAll questions output saved to: {csv_path}\nAll questions Excel output saved to: {excel_path}")
    else:
        print("Could not find the required columns. Available columns:", df.columns)

if __name__ == "__main__":
    import os
    # Path to the dataset directory
    DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets--Meyerger--ASAG2024", "snapshots", "9a7a179de4e227f5e78625bc9ded2d8761f1ff09")
    parquet_path = os.path.join(DATASET_DIR, "train.parquet")
    if os.path.exists(parquet_path):
        show_questions_and_answers(parquet_path)
    else:
        print("train.parquet not found in the dataset directory.")
