"""
Training pipeline for the Auto Tagging Ticket Support System.

Usage:
    python src/train.py                    # uses Banking77 (default)
    python src/train.py --csv              # uses local CSV in data/raw/

Outputs:
    models/model.pkl      — Trained sklearn Pipeline (TF-IDF + LogReg)
    models/metrics.json   — Evaluation metrics
    data/processed/       — Cleaned dataset CSV
"""

import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    load_banking77,
    load_and_prepare_data,
    preprocess_series,
    build_tfidf_vectorizer,
)

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw" / "customer_support_tickets.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "tickets_clean.csv"
MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"


def train(use_csv: bool = False):
    # ── Step 1: Load data ──────────────────────────────────────────
    if use_csv:
        print(f"[1/6] Loading data from CSV: {DATA_RAW}")
        df = load_and_prepare_data(str(DATA_RAW))
        print(f"      Loaded {len(df):,} rows | {df['label'].nunique()} unique labels")

        print("[2/6] Preprocessing text...")
        df["text_clean"] = preprocess_series(df["text"])

        DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PROCESSED, index=False)

        print("[3/6] Splitting dataset (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            df["text_clean"], df["label"],
            test_size=0.2, random_state=42, stratify=df["label"],
        )
        all_labels = sorted(df["label"].unique())

    else:
        print("[1/6] Loading Banking77 dataset from HuggingFace...")
        train_df, test_df = load_banking77()
        print(f"      Train: {len(train_df):,} | Test: {len(test_df):,} | Labels: {train_df['label'].nunique()}")

        print("[2/6] Preprocessing text...")
        train_df["text_clean"] = preprocess_series(train_df["text"])
        test_df["text_clean"] = preprocess_series(test_df["text"])

        # Save combined processed data for EDA/dashboard
        DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat([train_df, test_df], ignore_index=True)
        combined.to_csv(DATA_PROCESSED, index=False)

        print("[3/6] Using Banking77 built-in train/test split...")
        X_train, y_train = train_df["text_clean"], train_df["label"]
        X_test, y_test = test_df["text_clean"], test_df["label"]
        all_labels = sorted(combined["label"].unique())

    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"      Saved processed data → {DATA_PROCESSED}")

    # ── Step 4: Build & train pipeline ────────────────────────────
    print("[4/6] Building and training pipeline (TF-IDF + LogisticRegression)...")
    pipeline = Pipeline([
        ("tfidf", build_tfidf_vectorizer(max_features=10_000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(
            solver="lbfgs",
            multi_class="auto",
            max_iter=1000,
            C=1.0,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    print("      Training complete.")

    # ── Step 5: Evaluate ──────────────────────────────────────────
    print("[5/6] Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)

    print(f"\n{'─'*50}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  F1-score  : {f1:.4f} (weighted)")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred))

    if f1 < 0.75:
        print(f"  ⚠  WARNING: F1-score ({f1:.4f}) is below target of 0.75")
    else:
        print(f"  ✓  F1-score meets target of 0.75")

    # ── Step 6: Save artifacts ────────────────────────────────────
    print("[6/6] Saving model artifacts...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"      Model saved → {MODEL_PATH}")

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": all_labels,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "num_classes": len(all_labels),
        "model_type": "TF-IDF + Logistic Regression",
        "dataset": "Banking77 (HuggingFace)" if not use_csv else "Custom CSV",
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"      Metrics saved → {METRICS_PATH}")

    print("\nTraining pipeline completed successfully.")
    return pipeline, metrics


if __name__ == "__main__":
    use_csv = "--csv" in sys.argv
    train(use_csv=use_csv)
