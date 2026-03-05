"""
preprocessing.py
Enhanced text preprocessing utilities for ASAG.
Includes lemmatization, stemming, stopword removal, and text normalization.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize tools
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))

# Extended stop words for academic context
EXTENDED_STOPWORDS = STOP_WORDS.union({
    'question', 'answer', 'student', 'reference', 'please', 'explain', 
    'describe', 'define', 'state', 'list', 'give', 'discuss', 'think',
    'consider', 'include', 'know', 'ask', 'show', 'tell', 'say'
})


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, remove extra whitespace, fix unicode."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[\n\r\t]+', ' ', text)  # Replace newlines/tabs with space
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = text.strip()
    return text


def tokenize_simple(text: str) -> list:
    """Simple word tokenization (alphanumeric and underscore only)."""
    if not text:
        return []
    return re.findall(r'\w+', text.lower())


def tokenize_nltk(text: str) -> list:
    """NLTK-based tokenization (preserves contractions)."""
    if not text:
        return []
    text = normalize_text(text)
    try:
        return word_tokenize(text)
    except:
        return tokenize_simple(text)


def remove_stopwords(tokens: list, extended: bool = False) -> list:
    """Remove stopwords from token list."""
    stop_set = EXTENDED_STOPWORDS if extended else STOP_WORDS
    return [t for t in tokens if t.lower() not in stop_set and len(t) > 0]


def lemmatize_tokens(tokens: list) -> list:
    """Apply lemmatization to tokens."""
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def stem_tokens(tokens: list) -> list:
    """Apply stemming to tokens."""
    return [STEMMER.stem(t) for t in tokens]


def preprocess_text(text: str, 
                   lemmatize: bool = True, 
                   stem: bool = False,
                   remove_stops: bool = True,
                   extended_stops: bool = False) -> str:
    """
    Full preprocessing pipeline.
    
    Args:
        text: Input text to preprocess
        lemmatize: Apply lemmatization
        stem: Apply stemming (overrides lemmatization if True)
        remove_stops: Remove stopwords
        extended_stops: Use extended stopword list
    
    Returns:
        Preprocessed text as space-separated tokens
    """
    if not text:
        return ""
    
    # Normalize
    text = normalize_text(text)
    
    # Tokenize
    tokens = tokenize_nltk(text)
    
    # Remove stopwords
    if remove_stops:
        tokens = remove_stopwords(tokens, extended=extended_stops)
    
    # Apply morphological processing
    if stem:
        tokens = stem_tokens(tokens)
    elif lemmatize:
        tokens = lemmatize_tokens(tokens)
    
    # Filter out very short tokens
    tokens = [t for t in tokens if len(t) > 1]
    
    return ' '.join(tokens)


def preprocess_for_model(text: str, 
                        context: str = "training",
                        use_extended_stops: bool = False) -> str:
    """
    Preprocess text optimized for model input.
    For training: aggressive preprocessing (lemmatization + extended stopwords)
    For inference: conservative preprocessing (lemmatization only)
    
    Args:
        text: Input text
        context: 'training' or 'inference'
        use_extended_stops: Whether to use extended stopword list
    
    Returns:
        Preprocessed text
    """
    aggressive = context == "training"
    return preprocess_text(
        text,
        lemmatize=True,
        stem=False,
        remove_stops=True,
        extended_stops=use_extended_stops or aggressive
    )


def preprocess_answer_pair(question: str, 
                           student_answer: str,
                           reference_answer: str = None) -> dict:
    """
    Preprocess a question-answer pair consistently.
    
    Returns:
        Dict with preprocessed texts and original texts
    """
    result = {
        'question_original': question,
        'student_original': student_answer,
        'reference_original': reference_answer,
        'question': preprocess_for_model(question, context='training'),
        'student': preprocess_for_model(student_answer, context='training'),
    }
    
    if reference_answer:
        result['reference'] = preprocess_for_model(reference_answer, context='training')
    
    return result


def get_content_words(text: str, min_length: int = 3) -> set:
    """Extract content words (non-stopwords) from text."""
    tokens = tokenize_nltk(normalize_text(text))
    tokens = remove_stopwords(tokens, extended=True)
    return {t for t in tokens if len(t) >= min_length}


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    words1 = get_content_words(text1)
    words2 = get_content_words(text2)
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def compute_text_stats(text: str) -> dict:
    """Compute text statistics."""
    tokens = tokenize_simple(text)
    content_words = get_content_words(text)
    
    return {
        'length': len(text),
        'tokens': len(tokens),
        'content_words': len(content_words),
        'avg_token_length': sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
        'unique_tokens': len(set(tokens)),
    }
