## Sentiment Analysis Using Machine Learning

### 1. Introduction

In this project, we apply **data mining and machine learning** techniques to the problem of **sentiment analysis**. Sentiment analysis is the task of automatically determining whether a piece of text expresses a **positive**, **negative**, or **neutral** opinion. It is widely used in real-world applications such as analysing customer reviews, monitoring opinions on social media, and supporting business decision-making.

The main goal of this work is to:

- Use **machine learning algorithms** (specifically **Naive Bayes** and **Logistic Regression**) to classify sentiment in short text reviews.
- Explain and demonstrate the **data preprocessing** steps that are required before applying machine learning, and show how they improve the quality of data for mining.

This project is aligned with the course **CS404 Data Mining and Machine Learning**, especially:

- **Unit 2**: Data Preprocessing.
- **Unit 3**: Supervised Learning Algorithms.
- **Unit 6**: Data Mining Applications.

### 2. Dataset Description

For this project, we use a small real-world-style dataset of short **product and service reviews**. Each row in the dataset contains:

- `text`: a short review written in natural language (e.g. *\"I love this phone, the battery life is amazing\"*).
- `sentiment`: a label indicating the overall sentiment of the review:
  - `positive`
  - `negative`
  - `neutral`

The dataset is stored in the file `data/sample_reviews.csv`. Although the dataset is relatively small for demonstration purposes, it represents typical real-world text data: informal language, noise (such as punctuation and varying writing styles), and a mix of positive, negative, and neutral opinions.

### 3. Data Preprocessing

Data preprocessing is a critical step in the **data mining process** and the **machine learning pipeline**. Raw data is often noisy, incomplete, and not in a suitable format for algorithms. In this project, we apply several preprocessing steps, and we explain how each one improves the quality of the data.

#### 3.1 Data Cleaning

1. **Handling missing values and duplicates**
   - We first load the dataset using `pandas` and remove any rows where the `text` or `sentiment` field is missing.
   - We also drop duplicate reviews (same text and same sentiment).
   - **Benefit**: This step ensures that the models are trained on **complete and non-redundant** data, which reduces bias and prevents the model from being influenced by repeated examples.

2. **Removing noise from text**
   - The raw text may contain URLs, user mentions, hashtags, numbers, and special characters.
   - We apply a cleaning function to:
     - Convert text to lowercase.
     - Remove URLs (e.g. `http://...`, `https://...`).
     - Remove user mentions (e.g. `@username`) and hashtags (e.g. `#topic`).
     - Remove non-alphabetic characters and extra whitespace.
   - **Benefit**: Removing noise focuses the model on the **semantic content** of the text (the words expressing sentiment) and reduces irrelevant variation in the data.

#### 3.2 Text Normalization

1. **Lowercasing**
   - All text is converted to lowercase (e.g. *\"Good\"* and *\"good\"* become the same token).
   - **Benefit**: This reduces the size of the vocabulary and ensures that the model does not treat the same word with different cases as separate features.

2. **Stopword removal (through TF‑IDF)**
   - Many common words such as *\"the\"*, *\"is\"*, and *\"and\"* appear very frequently but do not carry strong sentiment.
   - When we use the `TfidfVectorizer` from Scikit-Learn with `stop_words=\"english\"`, most common English stopwords are removed.
   - **Benefit**: This acts as a form of **feature selection**, focusing the model on more informative words and reducing the dimensionality of the feature space.

3. **Tokenization**
   - Tokenization is the process of splitting text into a sequence of **tokens** (usually words).
   - The `TfidfVectorizer` internally performs tokenization, splitting the cleaned text on whitespace and punctuation.
   - **Benefit**: Tokenization transforms raw text into a structured representation that can be counted, weighted, and used as input to machine learning algorithms.

#### 3.3 Feature Extraction (Vectorization)

Machine learning algorithms work with numeric feature vectors, not raw strings. Therefore, we transform the cleaned and tokenized text into numeric representations using **TF‑IDF (Term Frequency–Inverse Document Frequency)**:

- We use `TfidfVectorizer` with:
  - `stop_words=\"english\"` to remove common stopwords.
  - `ngram_range=(1, 2)` to include both single words (**unigrams**) and pairs of consecutive words (**bigrams**).
- The result is a sparse matrix where each row corresponds to a review, and each column corresponds to a word or word pair. The values are TF‑IDF scores, which reflect how important a word is in a document relative to the whole dataset.

**Benefits**:

- Converts qualitative text into **quantitative features** that algorithms can process.
- Emphasizes words that are important for distinguishing between different sentiments.
- Reduces the effect of very common words that appear in almost all documents.

#### 3.4 Train–Test Split

To correctly evaluate the performance of the models, we split the dataset into:

- **Training set** (e.g. 80% of the data): used to fit the models.
- **Test set** (e.g. 20% of the data): used only for evaluation.

We use `train_test_split` from Scikit-Learn with `stratify` on the sentiment labels to preserve the proportion of positive, negative, and neutral classes.

**Benefit**:

- Ensures that we measure **generalization performance**, i.e. how well the model performs on unseen data, which is a key concept in machine learning.

### 4. Machine Learning Models

After preprocessing and feature extraction, we apply two **supervised learning** algorithms from Scikit-Learn:

#### 4.1 Naive Bayes (MultinomialNB)

- We use the **Multinomial Naive Bayes** classifier, which is commonly applied to text classification problems.
- It models the probability of each class (sentiment) based on the frequencies of words in the text, under the simplifying assumption that features (words) are conditionally independent given the class.
- **Advantages**:
  - Simple and fast.
  - Performs surprisingly well for many text mining tasks.

#### 4.2 Logistic Regression

- We use **Logistic Regression** for multiclass classification (positive, negative, neutral).
- Logistic Regression learns a linear decision boundary in the TF‑IDF feature space and outputs probabilities for each class.
- **Advantages**:
  - More flexible decision boundary than Naive Bayes in many settings.
  - Often achieves strong performance on high-dimensional sparse data such as text.

### 5. Model Evaluation and Results

We train both models on the training data and evaluate them on the test data using the following **evaluation metrics**:

- **Accuracy**: the proportion of correctly classified reviews.
- **Precision, Recall, and F1‑score** for each class, obtained using `classification_report` from Scikit-Learn.
- **Confusion matrix**: shows how many examples of each true class are predicted as each possible class.

On the small sample dataset used here, both models achieve **reasonable accuracy**. Logistic Regression often performs slightly better than Naive Bayes, especially when using TF‑IDF features with unigrams and bigrams. However, Naive Bayes remains competitive and is very fast to train.

The evaluation results show that:

- Most **positive** and **negative** reviews are correctly identified by both models.
- **Neutral** reviews are slightly more difficult to classify, because they may contain a mixture of positive and negative words or very weak sentiment.

Even on this small dataset, the experiment demonstrates how the choice of features (TF‑IDF with stopword removal and n‑grams) and preprocessing steps impacts model performance.

### 6. Impact of Data Preprocessing on Data Quality

The project clearly shows that **data preprocessing improves the quality of data for mining**:

- **Cleaning missing values and duplicates** ensures that the models are trained on valid and diverse examples, not repeated or incomplete records.
- **Removing noise** such as URLs, hashtags, and special characters helps the algorithms to focus on meaningful words, reducing the risk of overfitting to irrelevant patterns.
- **Text normalization** (lowercasing, stopword handling) reduces unnecessary variation and the dimensionality of the feature space, making the learning process more stable and efficient.
- **Tokenization and TF‑IDF vectorization** transform unstructured text into structured numeric features that capture the importance of words for sentiment classification.
- **Train–test splitting** ensures a fair and reliable evaluation of how well the models generalize to unseen data.

Without these preprocessing steps, the raw text data would be noisy, inconsistent, and in a form that standard machine learning algorithms cannot handle directly. Preprocessing therefore plays a central role in making data suitable for **data mining and machine learning**.

### 7. Conclusion

In this project, we successfully implemented a sentiment analysis system using **Naive Bayes** and **Logistic Regression** to classify short reviews into positive, negative, and neutral sentiment. The experiment demonstrates the full **machine learning pipeline**:

1. Data collection and understanding.
2. Data preprocessing and feature engineering.
3. Model training and evaluation.

The results highlight that well-designed **preprocessing** and appropriate **feature extraction** (TF‑IDF with stopword removal and n‑grams) are essential for achieving good performance in text mining tasks. Although we worked with a small dataset and relatively simple models, the methodology can be extended to larger datasets and more advanced algorithms (e.g. neural networks with TensorFlow or PyTorch) in future work.

### 8. References and Further Reading

The following books and resources provide additional background and depth for this project:

1. Stemkoski, L., & Pascale, M. (2021). *Developing Graphics Frameworks with Python and OpenGL*. OAPEN.\n2. *Python Machine Learning for Beginners: Learning from scratch NumPy, pandas, Matplotlib, Seaborn, Scikitlearn, and TensorFlow for Machine Learning and Data Science*. (2020). AI Publishing.\n3. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems*. O’Reilly.\n4. Witten, I. H., Frank, E., & Hall, M. A. (2011). *Data Mining: Practical Machine Learning Tools and Techniques*. Morgan Kaufmann.\n5. Han, J., Kamber, M., & Pei, J. (2022). *Data Mining: Concepts and Techniques*. Morgan Kaufmann.

