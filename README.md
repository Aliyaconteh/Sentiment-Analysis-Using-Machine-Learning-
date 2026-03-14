## Sentiment Analysis Project

A small, end-to-end sentiment analysis project that cleans text reviews, vectorizes them with TF-IDF, and trains two classic classifiers (Multinomial Naive Bayes and Logistic Regression). Results are shown in a simple Tkinter GUI with a run log and model metrics.

## Features
- Cleans and normalizes review text
- TF-IDF vectorization with unigrams + bigrams
- Trains and evaluates two models: Naive Bayes and Logistic Regression
- Displays accuracy, classification report, and confusion matrix
- Tkinter UI for viewing logs and results

## Project Structure
- sentiment_analysis_project.py — main script (load, preprocess, train, evaluate, GUI)
- `data/sample_reviews.csv — example dataset with text and sentiment columns
- `requirements.txt — Python dependencies
- `report.md — project write-up
- `documentation.md — extended documentation
- `explanation_simple.md — simplified explanation

## Requirements
- Python 3.8+

Install dependencies:

pip install -r requirements.txt


## How to Run
From the project folder:

python sentiment_analysis_project.py


## What You’ll See
- A **Run Log** tab with dataset stats and preprocessing details
- A **Model Results** tab with accuracy, classification reports, and confusion matrices

## Dataset Format
`data/sample_reviews.csv should contain:
- text: the review text
- sentiment: label such as positive, negative, or neutral

## Notes
- The script uses a fixed train/test split with random_state=42 and stratification.
- If you want to use your own dataset, keep the same column names (text, sentiment).

## License
This project is for educational use.
