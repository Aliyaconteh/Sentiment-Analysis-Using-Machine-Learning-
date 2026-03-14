import re
from typing import Tuple

import numpy as np
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk
from tkinter import scrolledtext, ttk


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop rows with missing text or sentiment
    df = df.dropna(subset=["text", "sentiment"])
    # Remove duplicate rows
    df = df.drop_duplicates(subset=["text", "sentiment"])
    return df


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Lowercasing
    - Removing URLs, mentions, hashtags, numbers
    - Removing punctuation and extra whitespace
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)  # URLs
    text = re.sub(r"@[\w_]+", " ", text)  # @mentions
    text = re.sub(r"#[\w_]+", " ", text)  # hashtags
    text = re.sub(r"[^a-z\s]", " ", text)  # keep letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    # Remove rows where cleaning removed almost everything
    df = df[df["clean_text"].str.len() > 0]
    return df


def vectorize_text(
    train_text: pd.Series, test_text: pd.Series
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    return X_train, X_test, vectorizer


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> str:
    models = {
        "Naive Bayes (MultinomialNB)": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    lines = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report_dict = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        accuracy = report_dict.get("accuracy", 0.0)
        conf = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))

        lines.append("=" * 80)
        lines.append(f"Model: {name}")
        lines.append(f"Accuracy: {accuracy:.4f}")
        lines.append("")
        lines.append("Classification report:")
        lines.append(classification_report(y_test, y_pred, zero_division=0))
        lines.append("Confusion matrix:")
        lines.append(str(conf))
        lines.append("")

    return "\n".join(lines)


def show_results_window(log_text: str, results_text: str) -> None:
    root = tk.Tk()
    root.title("Sentiment Analysis Results")
    root.geometry("900x700")
    root.configure(bg="#d4d0c8")

    header = tk.Label(
        root,
        text="Sentiment Analysis Results",
        bg="#d4d0c8",
        fg="#000000",
        font=("MS Sans Serif", 12, "bold"),
    )
    header.pack(pady=(10, 6))

    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both", padx=12, pady=(0, 12))

    log_frame = tk.Frame(notebook, bg="#d4d0c8")
    results_frame = tk.Frame(notebook, bg="#d4d0c8")

    notebook.add(log_frame, text="Run Log")
    notebook.add(results_frame, text="Model Results")
    notebook.select(results_frame)

    log_area = scrolledtext.ScrolledText(
        log_frame,
        wrap=tk.WORD,
        font=("Consolas", 10),
        bg="#ffffff",
        fg="#000000",
        relief="sunken",
        bd=2,
    )
    log_area.pack(expand=True, fill="both", padx=10, pady=10)
    log_area.insert(tk.END, log_text)
    log_area.configure(state="disabled")

    results_label = tk.Label(
        results_frame,
        text="Model Results (includes accuracy and classification report)",
        bg="#d4d0c8",
        fg="#000000",
        font=("MS Sans Serif", 9, "bold"),
        anchor="w",
    )
    results_label.pack(fill="x", padx=10, pady=(10, 4))

    results_area = scrolledtext.ScrolledText(
        results_frame,
        wrap=tk.WORD,
        font=("Consolas", 10),
        bg="#ffffff",
        fg="#000000",
        relief="sunken",
        bd=2,
    )
    results_area.pack(expand=True, fill="both", padx=10, pady=(0, 10))
    results_area.insert(tk.END, results_text)
    results_area.configure(state="disabled")

    root.mainloop()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="sklearn.metrics._classification",
    )
    data_path = "data/sample_reviews.csv"

    log_lines = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    log(f"Loading dataset from: {data_path}")
    df = load_dataset(data_path)
    log(f"Number of rows after cleaning missing/duplicates: {len(df)}")
    log("\nSample of raw data:")
    log(df.head().to_string(index=True))
    df_clean = preprocess_dataset(df)
    log("\nApplying text preprocessing...")
    log(f"Number of rows after text cleaning: {len(df_clean)}")
    log("\nSample of cleaned text:")
    log(df_clean[["text", "clean_text", "sentiment"]].head().to_string(index=True))
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df_clean["clean_text"],
        df_clean["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df_clean["sentiment"],
    )
    log("\nVectorizing text with TF-IDF...")
    X_train, X_test, _ = vectorize_text(X_train_text, X_test_text)
    log(f"Training data shape: {X_train.shape}")
    log(f"Test data shape: {X_test.shape}")
    log("\nTraining and evaluating models...")
    results_text = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    log_text = "\n".join(log_lines)
    show_results_window(log_text, results_text)


if __name__ == "__main__":
    main()
