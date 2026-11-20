"""
Preprocess Module
=================

Implements regular-expression semantic entity masking, text normalization,
stopword pruning, and Porter morphological stemming for SMS spam detection.

Entity Masking Specifications:
- URLs: `https?://\\S+|www\\.\\S+` -> `httpaddr`
- Email Addresses: `\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b` -> `emailaddr`
- Currency Symbols: `[£$€₹¥]` & textual currencies -> `moneysymb`
- Phone Numbers: Sequences of 5+ digits or telephone formats -> `phonenum`
- General Numbers: `\\b\\d+\\b` -> `numbr`
"""

import os
import re
import string
import sys
from typing import Iterable, List, Optional, Set

# Attempt to load NLTK resources gracefully with offline fallbacks
try:
    import nltk
    # Check local workspace data first
    local_nltk_data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nltk_data")
    if os.path.exists(local_nltk_data) and local_nltk_data not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_data)

    from nltk.corpus import stopwords
    try:
        NLTK_STOPWORDS = set(stopwords.words("english"))
    except (LookupError, OSError):
        # Attempt silent download if online
        try:
            nltk.download("stopwords", quiet=True)
            NLTK_STOPWORDS = set(stopwords.words("english"))
        except Exception:
            NLTK_STOPWORDS = None
except ImportError:
    NLTK_STOPWORDS = None

try:
    from nltk.stem import PorterStemmer
    STEMMER = PorterStemmer()
except ImportError:
    STEMMER = None

# Comprehensive fallback English stopwords list (standard 198 NLTK terms)
FALLBACK_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn",
    "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll",
    "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't",
    "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn",
    "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

# Resolve active stop words
ACTIVE_STOPWORDS: Set[str] = NLTK_STOPWORDS if NLTK_STOPWORDS else FALLBACK_STOPWORDS


def mask_entities(text: str) -> str:
    """
    Apply regular-expression semantic entity masking to preserve vital fraud indicators.

    Replaces:
    - Web URLs with 'httpaddr'
    - Email addresses with 'emailaddr'
    - Currency symbols with 'moneysymb'
    - Phone sequences (5+ digits or phone formatted) with 'phonenum'
    - Numeric values with 'numbr'

    Args:
        text (str): Raw input SMS text string.

    Returns:
        str: Entity-masked text string.
    """
    if not isinstance(text, str):
        return ""

    # 1. URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " httpaddr ", text, flags=re.IGNORECASE)

    # 2. Email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " emailaddr ",
        text,
        flags=re.IGNORECASE,
    )

    # 3. Currency symbols and textual abbreviations
    text = re.sub(r"[£$€₹¥¢]", " moneysymb ", text)
    text = re.sub(
        r"\b(?:rs\.?|inr|usd|eur|gbp|pounds?|bucks?)\b",
        " moneysymb ",
        text,
        flags=re.IGNORECASE,
    )

    # 4. Telephone sequences: matches 5+ contiguous digits or formatted dial sequences
    text = re.sub(
        r"\b(?:\+?\d{1,4}[-.\s]?)?(?:\(?\d{2,5}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{3,5}\b|\b\d{5,}\b",
        " phonenum ",
        text,
    )

    # 5. General remaining numbers
    text = re.sub(r"\b\d+\b", " numbr ", text)

    return text


def preprocess_text(text: str, stem: bool = True) -> str:
    """
    Comprehensive text preprocessing pipeline for SMS messages:
    1. Regex semantic entity masking (httpaddr, emailaddr, moneysymb, phonenum, numbr)
    2. Case normalization (lowercasing)
    3. Punctuation removal
    4. Stopword filtering
    5. Morphological Porter stemming

    Args:
        text (str): Raw input SMS message.
        stem (bool): Whether to apply Porter stemming (default: True).

    Returns:
        str: Fully sanitized and normalized token string.
    """
    if not isinstance(text, str):
        return ""

    # Stage 1: Semantic Entity Masking
    text = mask_entities(text)

    # Stage 2: Lowercasing
    text = text.lower()

    # Stage 3: Punctuation removal (preserving whitespace separation)
    text = re.sub(r"[" + re.escape(string.punctuation) + "]", " ", text)

    # Stage 4: Tokenization & whitespace cleanup
    tokens = text.split()

    # Stage 5: Stopword pruning & token filtering
    filtered_tokens: List[str] = [
        t for t in tokens if t not in ACTIVE_STOPWORDS and len(t) > 1
    ]

    # Stage 6: Morphological Porter Stemming
    if stem and STEMMER is not None:
        cleaned_tokens = [STEMMER.stem(t) for t in filtered_tokens]
    else:
        cleaned_tokens = filtered_tokens

    return " ".join(cleaned_tokens)


# Scikit-Learn Transformer Compatibility
try:
    from sklearn.base import BaseEstimator, TransformerMixin

    class TextPreprocessor(BaseEstimator, TransformerMixin):
        """
        Scikit-learn compatible transformer for seamless integration into
        sklearn Pipeline objects.
        """

        def __init__(self, stem: bool = True):
            self.stem = stem

        def fit(self, X: Iterable[str], y: Optional[Iterable] = None):
            return self

        def transform(self, X: Iterable[str]) -> List[str]:
            return [preprocess_text(text, stem=self.stem) for text in X]

except ImportError:
    # If scikit-learn is not available in standalone execution
    class TextPreprocessor:  # type: ignore
        def __init__(self, stem: bool = True):
            self.stem = stem

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return [preprocess_text(text, stem=self.stem) for text in X]


if __name__ == "__main__":
    sample_text = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "CONGRATULATIONS!! You won £1000 cash or a £2000 prize. Call 08000930705 or visit http://claim-prize.com to collect."
    )
    print("=" * 60)
    print("SMS PREPROCESSING DEMONSTRATION")
    print("=" * 60)
    print(f"Raw Input:\n  {sample_text}\n")
    print(f"Entity Masked:\n  {mask_entities(sample_text)}\n")
    print(f"Fully Preprocessed (Masked + Lower + NoPunct + Stopwords + Stemmed):\n  {preprocess_text(sample_text)}\n")
    print("=" * 60)
