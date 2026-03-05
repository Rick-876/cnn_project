import pandas as pd
import torch
from transformers import AutoTokenizer

# Test hybrid_pipeline data loading
df = pd.read_csv('asag2024_all.csv')
df = df.rename(columns={
    'Question': 'question',
    'Student Answer': 'provided_answer',
    'Reference Answer': 'reference_answer',
    'Human Score/Grade': 'normalized_grade'
})
if df['normalized_grade'].max() > 1.0:
    df['normalized_grade'] = df['normalized_grade'] / df['normalized_grade'].max()
print(f'√ Data loaded: {df.shape[0]} samples')
print(f'  Grade range: [{df["normalized_grade"].min():.3f}, {df["normalized_grade"].max():.3f}]')

# Test tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
print(f'√ Tokenizer loaded: {tokenizer.__class__.__name__}')

# Test a sample encode
sample = 'Question: test Student answer: test answer'
encoded = tokenizer(sample, max_length=256, truncation=True, padding='max_length')
print(f'√ Tokenization works: {len(encoded["input_ids"])} tokens')
