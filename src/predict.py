"""
Inference/prediction module for the Auto Tagging Ticket Support System.

Usage:
    from src.predict import predict_tag

    result = predict_tag("My payment was charged twice but no order was placed.")
    # Returns: {"tag": "Billing", "confidence": 0.92, "all_scores": {...}}
"""

import sys
import numpy as np
import joblib
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import preprocess_text

ROOT = Path(__file__).parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "model.pkl"

_model_cache = {}


def load_model(model_path: str = None):
    """Load and cache the trained sklearn pipeline."""
    path = str(model_path or DEFAULT_MODEL_PATH)
    if path not in _model_cache:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model not found at '{path}'. "
                "Run `python src/train.py` first to train the model."
            )
        _model_cache[path] = joblib.load(path)
    return _model_cache[path]


def predict_tag(text: str, model_path: str = None) -> dict:
    """
    Predict the support ticket tag for given text.

    Args:
        text: Raw support ticket text
        model_path: Optional path to model file (defaults to models/model.pkl)

    Returns:
        dict with keys:
            - tag (str): Predicted category/tag
            - confidence (float): Confidence of top prediction (0–1)
            - all_scores (dict): Confidence scores for all classes
            - preprocessed_text (str): Cleaned input text
    """
    pipeline = load_model(model_path)

    # Preprocess input
    processed = preprocess_text(text)

    # Predict
    tag = pipeline.predict([processed])[0]
    proba = pipeline.predict_proba([processed])[0]
    classes = pipeline.classes_

    # Build scores dict
    all_scores = {cls: round(float(prob), 4) for cls, prob in zip(classes, proba)}
    confidence = round(float(np.max(proba)), 4)

    return {
        "tag": tag,
        "confidence": confidence,
        "all_scores": all_scores,
        "preprocessed_text": processed,
    }


def predict_batch(texts: list[str], model_path: str = None) -> list[dict]:
    """
    Predict tags for a list of ticket texts.

    Args:
        texts: List of raw ticket text strings
        model_path: Optional path to model file

    Returns:
        List of prediction dicts (same format as predict_tag)
    """
    pipeline = load_model(model_path)
    processed = [preprocess_text(t) for t in texts]

    tags = pipeline.predict(processed)
    probas = pipeline.predict_proba(processed)
    classes = pipeline.classes_

    results = []
    for text, raw, tag, proba in zip(texts, processed, tags, probas):
        all_scores = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        results.append(
            {
                "tag": tag,
                "confidence": round(float(np.max(proba)), 4),
                "all_scores": all_scores,
                "preprocessed_text": raw,
            }
        )
    return results


if __name__ == "__main__":
    # Quick smoke test
    sample_tickets = [
        "My internet connection keeps dropping every few minutes.",
        "I was charged twice for the same order. Please refund.",
        "How do I reset my password? I can't log in.",
        "The package was supposed to arrive yesterday but it's still not here.",
        "I want to cancel my subscription immediately.",
    ]

    print("Running prediction smoke test...\n")
    for ticket in sample_tickets:
        result = predict_tag(ticket)
        print(f"Ticket : {ticket}")
        print(f"  Tag  : {result['tag']} (confidence: {result['confidence']:.2%})")
        print()
