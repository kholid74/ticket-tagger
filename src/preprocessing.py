"""
Text preprocessing pipeline for ticket classification.
Handles cleaning, tokenization, stopword removal, lemmatization, and TF-IDF vectorization.
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib

# Download required NLTK resources (run once)
def download_nltk_resources():
    resources = ["stopwords", "wordnet", "punkt", "punkt_tab"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean raw ticket text:
    - Lowercase
    - Remove template placeholders like {product_purchased}
    - Remove URLs
    - Remove special characters and numbers
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"\{[^}]+\}", "", text)             # remove {placeholder} templates
    text = re.sub(r"http\S+|www\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)              # keep only letters
    text = re.sub(r"\s+", " ", text).strip()          # normalize whitespace
    return text


def tokenize_and_filter(text: str) -> list[str]:
    """Tokenize text and remove stopwords."""
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Apply lemmatization to each token."""
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline: clean → tokenize → lemmatize → join."""
    cleaned = clean_text(text)
    tokens = tokenize_and_filter(cleaned)
    lemmas = lemmatize_tokens(tokens)
    return " ".join(lemmas)


def preprocess_series(series: pd.Series) -> pd.Series:
    """Apply full preprocessing to a pandas Series of texts."""
    return series.apply(preprocess_text)


def build_tfidf_vectorizer(
    max_features: int = 10_000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> TfidfVectorizer:
    """Build a TF-IDF vectorizer with sensible defaults."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )


def load_banking77() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load bundled customer support ticket dataset (no internet required).
    200 realistic examples across 5 categories, split 80/20 train/test.
    Returns (train_df, test_df) each with 'text' and 'label' columns.
    """
    from src.sample_data import TICKETS
    from sklearn.model_selection import train_test_split

    rows = [
        {"text": text, "label": label}
        for label, texts in TICKETS.items()
        for text in texts
    ]
    df = pd.DataFrame(rows)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load the customer support ticket CSV and return a cleaned DataFrame
    with 'text' and 'label' columns.

    Combines ticket_subject + ticket_description when both are available,
    since ticket_subject is more informative and description may contain
    noisy/template text.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Identify label column
    label_candidates = ["ticket_type", "type", "category", "label", "tag"]
    label_col = next((c for c in label_candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"Could not find label column. Available: {list(df.columns)}")

    # Combine subject + description for richer signal
    subject_col = "ticket_subject" if "ticket_subject" in df.columns else None
    desc_candidates = ["ticket_description", "description", "body", "text", "message", "issue"]
    desc_col = next((c for c in desc_candidates if c in df.columns), None)

    if subject_col and desc_col:
        # Repeat subject 3x to give it higher weight in TF-IDF
        df["text"] = (
            (df[subject_col].fillna("") + " ") * 3
            + df[desc_col].fillna("")
        )
        print(f"      Using combined columns: '{subject_col}' (3x) + '{desc_col}'")
    elif subject_col:
        df["text"] = df[subject_col].fillna("")
        print(f"      Using text column: '{subject_col}'")
    elif desc_col:
        df["text"] = df[desc_col].fillna("")
        print(f"      Using text column: '{desc_col}'")
    else:
        raise ValueError(f"Could not find text column. Available: {list(df.columns)}")

    df = df[["text", label_col]].copy()
    df.columns = ["text", "label"]
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(str).str.strip()
    return df
