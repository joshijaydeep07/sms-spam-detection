"""
SMS Spam Message Detection Package
==================================

An end-to-end Machine Learning pipeline for SMS spam and phishing ('smishing')
classification based on regular-expression semantic entity masking, sublinear
n-gram TF-IDF vectorization, and lightweight statistical classifiers (LinearSVC,
Naive Bayes variants).

Developed at Unitedworld Institute of Technology (UIT), Karnavati University.
"""

from src.preprocess import (
    mask_entities,
    preprocess_text,
    TextPreprocessor,
)
from src.predict import (
    predict_message,
    load_artifacts,
)

__version__ = "1.0.0"
__all__ = [
    "mask_entities",
    "preprocess_text",
    "TextPreprocessor",
    "predict_message",
    "load_artifacts",
]
