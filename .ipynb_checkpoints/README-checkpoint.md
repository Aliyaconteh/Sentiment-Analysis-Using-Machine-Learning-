## Sentiment Analysis Exam Project (Data Mining & Machine Learning)

This project implements a simple **sentiment analysis** system using **machine learning** as required by your exam question:

- Use **logistic regression** and **Naive Bayes** to classify sentiment in text data.
- Apply and explain **data preprocessing** steps such as tokenization, cleaning, and feature extraction.

### Project Structure

- `requirements.txt` – Python dependencies for the project.
- `data/sample_reviews.csv` – Small real-world-style dataset of reviews with sentiment labels (`positive`, `negative`, `neutral`).
- `sentiment_analysis_project.py` – Python script with:
  - Data loading
  - Data preprocessing (cleaning, tokenization, vectorization)
  - Training **Naive Bayes** and **Logistic Regression** classifiers
  - Model evaluation and comparison
- `report.md` – Written explanation of preprocessing steps and how they improve data quality, plus a short discussion of results.

### How to Run (from this folder)

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the sentiment analysis script**:

   ```bash
   python sentiment_analysis_project.py
   ```

   This will:
   - Load the dataset
   - Apply preprocessing
   - Vectorize the text with TF-IDF
   - Train Naive Bayes and Logistic Regression
   - Print evaluation metrics to the terminal

You can use `report.md` directly (or adjust the wording) as your exam write-up. If you prefer Jupyter Notebooks, you can also copy the code from `sentiment_analysis_project.py` into a notebook cell-by-cell.


