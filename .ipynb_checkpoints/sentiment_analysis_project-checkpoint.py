import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


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
) -> None:
    models = {
        "Naive Bayes (MultinomialNB)": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    for name, model in models.items():
        print("=" * 80)
        print(f"Model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("\\nClassification report:")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique())))
        print()


def main() -> None:
    data_path = "data/sample_reviews.csv"
    print(f"Loading dataset from: {data_path}")
    df = load_dataset(data_path)
    print(f"Number of rows after cleaning missing/duplicates: {len(df)}")
    print("\\nSample of raw data:")
    print(df.head())

    print("\\nApplying text preprocessing...")
    df_clean = preprocess_dataset(df)
    print(f"Number of rows after text cleaning: {len(df_clean)}")
    print("\\nSample of cleaned text:")
    print(df_clean[["text", "clean_text", "sentiment"]].head())

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df_clean["clean_text"],
        df_clean["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df_clean["sentiment"],
    )

    print("\\nVectorizing text with TF-IDF...")
    X_train, X_test, _ = vectorize_text(X_train_text, X_test_text)
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    print("\\nTraining and evaluating models...")
    train_and_evaluate_models(X_train, X_test, y_train, y_test)
if __name__ == "__main__":
    main()

