#!/usr/bin/env python
"""Simple test to verify both scripts can load data and initialize."""
import pandas as pd
import sys

# Test CSV loading and column mapping (as done in both scripts)
df = pd.read_csv('asag2024_all.csv')
print(f"✓ CSV loaded: {df.shape[0]} rows")

# Test column mapping
if 'Question' in df.columns:
    df = df.rename(columns={
        'Question': 'question',
        'Student Answer': 'provided_answer',
        'Reference Answer': 'reference_answer',
        'Human Score/Grade': 'normalized_grade'
    })
    print("✓ Columns renamed")

# Test grade normalization
if df['normalized_grade'].max() > 1.0:
    df['normalized_grade'] = df['normalized_grade'] / df['normalized_grade'].max()
    print("✓ Grades normalized")

required = ['question', 'reference_answer', 'provided_answer', 'normalized_grade']
df = df[required].fillna({
    "question": "", 
    "reference_answer": "", 
    "provided_answer": "", 
    "normalized_grade": 0.0
})
print(f"✓ Data cleaned: {df.shape}")

# Test tokenizer (for hybrid_pipeline)
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
    sample = "Question: test Student answer: test answer"
    encoded = tokenizer(sample, max_length=256, truncation=True, padding='max_length')
    print(f"✓ Tokenizer works ({len(encoded['input_ids'])} tokens)")
except Exception as e:
    print(f"✗ Tokenizer failed: {e}")
    sys.exit(1)

print("\n✓ All checks passed! Both scripts should run.")
