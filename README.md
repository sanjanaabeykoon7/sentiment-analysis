# Sentiment Analysis on Movie Reviews

A machine learning project that classifies IMDB movie reviews as positive or negative using Natural Language Processing (NLP) techniques.

## Project Overview

This project demonstrates sentiment analysis on the IMDB dataset containing 50,000 movie reviews. The goal is to build a classifier that can accurately predict whether a review is positive or negative.

## Dataset

- Source: IMDB Movie Reviews Dataset
- Size: 50,000 reviews
- Distribution: 25,000 positive, 25,000 negative (perfectly balanced)
- Format: CSV file with two columns (review, sentiment)

## Methodology

### 1. Data Preprocessing
- Converted text to lowercase
- Removed HTML tags, URLs, and special characters
- Removed stopwords and short words
- Cleaned 50,000 reviews for analysis

### 2. Feature Engineering
- Used TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Created 5,000 features from the most important words
- Included both unigrams and bigrams

### 3. Model Training
- Split data: 80% training (40,000 reviews), 20% testing (10,000 reviews)
- Trained two models:
  - Logistic Regression
  - Multinomial Naive Bayes

## Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~89% |
| Naive Bayes | ~85% |

Both models achieved strong performance with high confidence scores on test reviews.

## Key Findings

- Most positive words: great, excellent, perfect, amazing, wonderful
- Most negative words: worst, awful, bad, waste, boring
- Model struggles with neutral/mixed sentiment reviews
- High confidence (95%+) on clearly positive or negative reviews

## Technologies Used

- Python 3.x
- pandas, numpy (data manipulation)
- scikit-learn (machine learning)
- nltk (text preprocessing)
- matplotlib, seaborn (visualization)
- wordcloud (word cloud generation)

## Installation
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud
```

## Usage

1. Download the IMDB dataset
2. Place `IMDB Dataset.csv` in the project directory
3. Run the Jupyter notebook `sentiment_analysis.ipynb`
4. Test with your own reviews using the `analyze_review()` function

## Future Improvements

- Add support for neutral sentiment classification
- Implement deep learning models (LSTM, BERT)
- Create a web interface for real-time predictions
- Expand to other domains (product reviews, social media)
- Hyperparameter tuning for better accuracy


Just built as a machine learning learning project.

---
