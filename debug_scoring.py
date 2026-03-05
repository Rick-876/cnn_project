"""Debug scoring to understand what's happening."""
import pandas as pd
import re
import numpy as np
from collections import Counter

def tokenize(text):
    return re.findall(r"\w+", text.lower())

# Load data
df = pd.read_csv('asag2024_all.csv')
df.columns = ['question', 'provided_answer', 'reference_answer', 'normalized_grade']

# Find DHCP question
matches = df[df['question'].str.contains('Dynamic Host Configuration', case=False, na=False)]
print(f"Found {len(matches)} DHCP questions")
print(f"\nReference answer:")
ref = matches['reference_answer'].iloc[0]
print(ref)
print(f"\nGrade distribution for DHCP:")
print(matches['normalized_grade'].value_counts().sort_index())

# Test student answer
student = """The Dynamic Host Configuration Protocol (DHCP) is a network management protocol used in Internet Protocol (IP) networks, whereby a DHCP server dynamically assigns an IP address and other network configuration parameters to each device on the network. DHCP has largely replaced RARP (and BOOTP). Uses of DHCP are: Simplifies installation and configuration of end systems, Allows for manual and automatic IP address assignment, May provide additional configuration information (DNS server, netmask, default router, etc.)"""

# Compare
STOPWORDS = {
    "what","is","the","a","an","of","in","to","and","or","for",
    "on","at","with","this","that","are","it","as","be","from",
    "by","was","were","has","have","had","its","do","does","did",
    "how","why","when","where","which","who","can","could","would",
    "should","may","might","will","shall","used","use","using",
    "also","about","between","into","they","them","their","your",
    "our","we","you","he","she","i","me","my","his","her","not",
    "no","so","if","than","then","such","any","all","each",
}

r_words = {w for w in tokenize(ref) if w not in STOPWORDS and len(w) > 2}
s_words = {w for w in tokenize(student) if w not in STOPWORDS and len(w) > 2}

print(f"\nReference key terms ({len(r_words)}): {sorted(r_words)[:20]}")
print(f"\nStudent key terms ({len(s_words)}): {sorted(s_words)[:20]}")
print(f"\nOverlap: {len(s_words & r_words)} / {len(r_words)} = {len(s_words & r_words) / len(r_words):.3f}")
print(f"\nMissing from student: {r_words - s_words}")
print(f"\nExtra in student: {list((s_words - r_words))[:20]}")

# Check actual grades for similar answers
print("\n\nSample high-scoring DHCP answers:")
high_score = matches[matches['normalized_grade'] >= 0.7]
for idx, row in high_score.head(3).iterrows():
    print(f"\nGrade: {row['normalized_grade']}")
    print(f"Answer: {row['provided_answer'][:200]}...")
