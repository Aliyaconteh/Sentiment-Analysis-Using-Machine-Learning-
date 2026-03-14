## Simple Explanation of My Sentiment Analysis Project

### 1. Problem and Goal

- **Problem**: We want a computer to read short text reviews and decide if each one is **positive**, **negative**, or **neutral**.
- **Goal**: Build a **machine learning model** that can automatically classify the **sentiment** of each review.

### 2. Dataset

- I created a small dataset of **24 short reviews** (for products, apps, services, etc.).
- Each review has:
  - `text`: the review sentence.
  - `sentiment`: one of three labels → `positive`, `negative`, or `neutral`.
- Example:
  - Text: *"I love this phone, the battery life is amazing"*
  - Sentiment: `positive`

### 3. Data Preprocessing Steps

Before we can use machine learning, we must **clean and prepare** the data. These are the main steps:

1. **Remove missing and duplicate rows**
   - Drop any review with missing `text` or `sentiment`.
   - Remove duplicate reviews.
   - **Why**: Ensures the data is **complete** and **not repeated**, which improves reliability.

2. **Text cleaning**
   - Convert all text to **lowercase**.
   - Remove:
     - URLs (e.g. `http://...`)
     - Mentions (e.g. `@user`)
     - Hashtags (e.g. `#topic`)
     - Punctuation and special characters.
     - Extra spaces.
   - **Why**: Removes **noise** and keeps only useful words, making patterns easier for the model to learn.

3. **Tokenization and stopwords (via TF‑IDF)**
   - We split text into **tokens** (words).
   - We use Scikit‑Learn’s `TfidfVectorizer` which:
     - Tokenizes the text.
     - Removes common **English stopwords** (like *the, is, and*).
     - Builds features from **unigrams and bigrams** (single words and word pairs).
   - **Why**: Focuses on more meaningful words and word combinations, and reduces irrelevant words.

4. **Feature extraction: TF‑IDF**
   - TF‑IDF turns each review into a **numeric vector**.
   - Each position in the vector represents a word (or word pair) and its importance in that review.
   - **Why**: Machine learning algorithms **cannot work directly on text**, they need **numbers**. TF‑IDF provides a good numeric representation for text mining.

5. **Train–test split**
   - Split the cleaned data into:
     - **Training set** (about 80% of reviews).
     - **Test set** (about 20% of reviews).
   - Use `train_test_split` with `stratify` on the labels so each class keeps a similar proportion.
   - **Why**: We train on one part of the data and evaluate on unseen data to measure **generalization**.

### 4. Machine Learning Algorithms Used

I used two **supervised learning** algorithms from Scikit‑Learn:

1. **Naive Bayes (MultinomialNB)**
   - A probabilistic classifier often used for text.
   - Assumes that words are **independent** given the class (the “naive” assumption).
   - **Advantages**: Very fast and simple; works well as a baseline for text classification.

2. **Logistic Regression**
   - A linear model that outputs **probabilities** for each class.
   - Works well on **high‑dimensional** data like TF‑IDF features.
   - **Advantages**: Often gives strong performance and interpretable coefficients.

### 5. Model Evaluation

- After training, I evaluated the models on the **test set** using:
  - **Accuracy**: Percentage of correctly classified reviews.
  - **Precision, Recall, F1‑score** for each class.
  - **Confusion matrix**: Shows how many positive, negative, and neutral reviews were correctly or incorrectly classified.
- Because the dataset and test set are **small**, the scores are not very high, and some classes may have **0 precision or recall**. Scikit‑Learn prints a **warning** for this, but the code still runs correctly.
- **Important**: For this exam project, the main focus is to **show the full pipeline** (preprocessing + algorithms + evaluation), not to reach very high accuracy.

### 6. How Preprocessing Improves Data Quality for Mining

In this project, preprocessing improves data quality in several ways:

- **Cleaning missing and duplicate data** → ensures reliable and non‑redundant examples.
- **Removing noise (URLs, punctuation, etc.)** → reduces irrelevant information and helps the model focus on meaningful words.
- **Lowercasing and stopword handling** → reduces vocabulary size and removes very common words that do not add much meaning.
- **TF‑IDF vectorization** → converts unstructured text into structured numerical features that capture word importance.
- **Train–test split** → allows a fair evaluation of model performance on unseen data.

Together, these steps turn raw, messy text into a **clean, structured dataset** suitable for **data mining and machine learning**.

### 7. Short Conclusion

- I implemented a **sentiment analysis** system that classifies reviews as positive, negative, or neutral.
- I used two **machine learning algorithms**: **Naive Bayes** and **Logistic Regression**.
- The project demonstrates the **complete machine learning pipeline**:
  - Data collection
  - Data preprocessing
  - Feature extraction
  - Model training
  - Model evaluation
- The work clearly shows that **good preprocessing** is essential to improve data quality and to make effective use of machine learning algorithms in data mining.

